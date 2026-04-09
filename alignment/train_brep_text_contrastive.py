#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler

from packaging import version
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dgl
from transformers import CLIPModel, CLIPProcessor

from brep_text_dataset import BrepTextDataset
from brep_encoder import BrepEncoder


# -------------------------- Utility Functions -------------------------- #
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_output_dir(root: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_name = f"brep_text_contrastive_{time.strftime('%Y%m%d_%H%M%S')}"
    out = root / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True, parents=True)
    (out / "weights").mkdir(exist_ok=True, parents=True)
    return out


def save_loss_curve(csv_path: Path, out_png: Path, title: str = "Training Loss"):
    steps, losses = [], []
    if not Path(csv_path).exists():
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            losses.append(float(row["loss"]))
    if not steps:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, linewidth=2)
    plt.xlabel("Global Step")
    plt.ylabel("CLIP Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_checkpoint(state: Dict[str, Any], ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(ckpt_path))


def load_checkpoint(ckpt_path: Path, map_location="cpu") -> Dict[str, Any]:
    return torch.load(str(ckpt_path), map_location=map_location)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _has_safetensors(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any(fn.endswith(".safetensors") for fn in os.listdir(p))


def _unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else m


# -------------------------- Trainer -------------------------- #
class BrepTextContrastiveTrainer:
    def __init__(self, args, rank: int = 0, world_size: int = 1, device: Optional[torch.device] = None):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Only the main process creates the output directory
        if self.is_main:
            self.out_dir = make_output_dir(args.output_dir)
            self.ckpt_dir = self.out_dir / "checkpoints"
            self.log_csv = self.out_dir / "train_log.csv"
            self.cfg_json = self.out_dir / "config.json"
            with open(self.cfg_json, "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False)
        else:
            # Non-main processes need to get the out_dir created by the main process
            self.out_dir = Path(args.output_dir) / "DUMMY_WILL_BE_OVERWRITTEN"
        if hasattr(args, "run_dir") and args.run_dir is not None:
            self.out_dir = Path(args.run_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_csv = self.out_dir / "train_log.csv"

        # Dataset & distributed sampler
        self.train_ds = BrepTextDataset(
            data_dir=args.data_dir,
            caption_file=args.caption_csv,
            center_and_scale=True,
            random_rotate=args.aug_rotate,
        )
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
            )
        else:
            self.train_sampler = None

        self.train_dl = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            collate_fn=BrepTextDataset.collate_fn,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True if args.num_workers > 0 else False,
            persistent_workers=(args.num_workers > 0),
        )

        # Model: BrepEncoder
        brep = BrepEncoder(
            srf_in_channels=4, crv_in_channels=4,
            face_emb_dim=32, edge_emb_dim=16,
            point_emb_dim=32, graph_emb_dim=128
        ).to(self.device)

        # DDP wrapper (only wrap trainable modules)
        if self.world_size > 1:
            self.brep_encoder = nn.parallel.DistributedDataParallel(
                brep, device_ids=[self.device.index], output_device=self.device.index, find_unused_parameters=False
            )
        else:
            self.brep_encoder = brep

        # Text side: Frozen CLIP Text Encoder (ViT-L/14, safetensors)
        self.clip_name = args.clip_name
        self.clip = self._load_clip_frozen(self.clip_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # Dimension alignment: project graph_emb(128) to CLIP embed_dim (ViT-L/14 -> 768)
        self.embed_dim = self.clip.config.projection_dim
        proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.embed_dim)
        ).to(self.device)

        if self.world_size > 1:
            self.brep_proj = nn.parallel.DistributedDataParallel(
                proj, device_ids=[self.device.index], output_device=self.device.index, find_unused_parameters=False
            )
        else:
            self.brep_proj = proj

        # Learnable logit_scale (in log form)
        init_temp = math.log(1.0 / 0.07)  # CLIP default temperature 0.07 -> initial logit_scale
        self.logit_scale = nn.Parameter(
            torch.tensor(init_temp, dtype=torch.float32, device=self.device)
        )
        # Optimizer / scheduler
        optim_params = list(_unwrap_ddp(self.brep_encoder).parameters()) + \
                       list(_unwrap_ddp(self.brep_proj).parameters()) + \
                       [self.logit_scale]
        self.optimizer = torch.optim.AdamW(
            optim_params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98)
        )

        self.total_steps = args.epochs * max(1, len(self.train_dl))
        self.warmup_steps = int(args.warmup_ratio * self.total_steps)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.scaler = GradScaler(enabled=args.amp)
        self.global_step = 0
        self.start_epoch = 0

        # Only main process writes CSV header
        if self.is_main and not (self.log_csv).exists():
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "loss", "lr"])
                writer.writeheader()

        # Resume from checkpoint (all ranks load to keep optimizer/scheduler state consistent)
        if args.resume:
            self._resume_from(args.resume)

        # Print parameter counts (main process only)
        if self.is_main:
            total1, train1 = count_params(_unwrap_ddp(self.brep_encoder))
            total2, train2 = count_params(_unwrap_ddp(self.brep_proj))
            print(f"[BrepEncoder] Total: {total1/1e6:.3f}M | Trainable: {train1/1e6:.3f}M")
            print(f"[BrepProj]     Total: {total2/1e6:.3f}M | Trainable: {train2/1e6:.3f}M")
            print(f"[Embed dim]    {self.embed_dim}")

    # ---------- CLIP loading (ViT-L/14, safetensors preferred) ---------- #
    def _load_clip_frozen(self, clip_name: str) -> CLIPModel:
        # Prefer safetensors to avoid torch<2.6 .bin limitation
        try:
            return CLIPModel.from_pretrained(clip_name, use_safetensors=True)
        except Exception as e:
            if Path(clip_name).exists() and not _has_safetensors(clip_name):
                raise RuntimeError(
                    f"Local directory {clip_name} does not contain *.safetensors, "
                    f"and the current environment does not allow .bin. "
                    f"Please use the official repo 'openai/clip-vit-large-patch14' (with safetensors), "
                    f"or convert the weights to safetensors. Original error: {e}"
                )
            raise

    # ---------- Resume from checkpoint ---------- #
    def _resume_from(self, ckpt_path: str):
        ckpt = load_checkpoint(Path(ckpt_path), map_location=self.device)
        _unwrap_ddp(self.brep_encoder).load_state_dict(ckpt["brep_encoder"])
        _unwrap_ddp(self.brep_proj).load_state_dict(ckpt["brep_proj"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.logit_scale.data = ckpt["logit_scale"].to(self.device)
        self.global_step = ckpt["global_step"]
        self.start_epoch = ckpt["epoch"] + 1
        if self.is_main:
            print(f"Resumed from {ckpt_path} @ epoch={self.start_epoch}, global_step={self.global_step}")

    # ---------- Contrastive loss ---------- #
    def clip_loss(self, brep_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        brep_emb = F.normalize(brep_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=50.0)
        logits_per_brep = logit_scale * brep_emb @ text_emb.t()  # [B,B]
        logits_per_text = logits_per_brep.t()
        labels = torch.arange(brep_emb.size(0), device=brep_emb.device)
        loss_i = F.cross_entropy(logits_per_brep, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2.0

    @torch.no_grad()
    def _encode_text(self, captions):
        inputs = self.clip_processor(
            text=captions, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        return self.clip.get_text_features(**inputs)  # [B, embed_dim]

    def _train_one_epoch(self, epoch: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.brep_encoder.train()
        self.brep_proj.train()
        self.clip.eval()  # Keep text tower frozen

        log_interval = max(1, self.args.log_interval)
        accum_loss = 0.0
        seen = 0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch+1}/{self.args.epochs}", dynamic_ncols=True, disable=not self.is_main)
        for it, batch in enumerate(pbar):
            if batch is None:
                continue

            g: dgl.DGLGraph = batch["graph"].to(self.device)
            captions = batch["captions"]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.args.amp):
                # Geometry side
                _, graph_emb = self.brep_encoder(g)     # [B,128]
                brep_feat = self.brep_proj(graph_emb)   # [B,embed_dim]

                # Text side (frozen)
                with torch.no_grad():
                    text_feat = self._encode_text(captions)  # [B,embed_dim]

                loss = self.clip_loss(brep_feat, text_feat)

            self.scaler.scale(loss).backward()
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(_unwrap_ddp(self.brep_encoder).parameters()) + list(_unwrap_ddp(self.brep_proj).parameters()),
                    self.args.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1
            accum_loss += loss.item()
            seen += 1

            # Only main process writes logs
            if self.is_main and self.global_step % log_interval == 0:
                avg = accum_loss / max(1, seen)
                lr = self.optimizer.param_groups[0]["lr"]
                with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "loss", "lr"])
                    writer.writerow({"global_step": self.global_step, "epoch": epoch, "loss": avg, "lr": lr})
                accum_loss = 0.0
                seen = 0

            # Update tqdm postfix (main process only)
            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{lr:.2e}",
                                 temp=f"{self.logit_scale.exp().item():.2f}")

        # End of epoch, only main process saves
        if self.is_main:
            self._save_ckpt(epoch, tag="last")
            self._save_brep_weights(epoch, tag="last")
            save_loss_curve(self.log_csv, self.out_dir / "loss_curve.png", title="Training Loss")

    def _save_ckpt(self, epoch: int, tag: str = "last"):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "brep_encoder": _unwrap_ddp(self.brep_encoder).state_dict(),
            "brep_proj": _unwrap_ddp(self.brep_proj).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "logit_scale": self.logit_scale.data.detach().cpu(),
            "clip_name": self.clip_name,
            "embed_dim": self.embed_dim,
        }
        ckpt_path = self.ckpt_dir / f"{tag}.ckpt"
        save_checkpoint(state, ckpt_path)

    def _save_brep_weights(self, epoch: int, tag: str = "epoch"):
        torch.save(_unwrap_ddp(self.brep_encoder).state_dict(), str(self.out_dir / "weights" / f"brep_encoder_{tag}_{epoch}.pt"))
        torch.save(_unwrap_ddp(self.brep_proj).state_dict(),    str(self.out_dir / "weights" / f"brep_projector_{tag}_{epoch}.pt"))

    def train(self):
        if self.is_main:
            print(f"Start training for {self.args.epochs} epochs. Output -> {self.out_dir}")
        for epoch in range(self.start_epoch, self.args.epochs):
            self._train_one_epoch(epoch)
        if self.is_main:
            print("Training done.")
            print(f"Logs:        {self.log_csv}")
            print(f"Checkpoints: {self.ckpt_dir}")
            print(f"Loss Curve:  {self.out_dir/'loss_curve.png'}")


# -------------------------- Distributed entry point -------------------------- #
def parse_args():
    p = argparse.ArgumentParser("BRep-Text Contrastive Training (train-only, DDP)")
    # Data
    p.add_argument("--data_dir", type=str, required=True, help="Root directory containing .bin files (organized by uid)")
    p.add_argument("--caption_csv", type=str, required=True, help="Training CSV (containing uid, abstract)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--aug_rotate", action="store_true", help="Random rotation augmentation")

    # Model / optimization
    p.add_argument("--clip_name", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Runtime
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path (e.g., .../checkpoints/last.ckpt)")
    p.add_argument("--output_dir", type=str, default="./output")

    # Multi-GPU selection
    p.add_argument("--gpus", type=str, default="", help='e.g. "0,1,2"; leave empty for auto-detect (single GPU fallback)')
    p.add_argument("--master_port", type=str, default="29512", help="DDP port (single-node multi-GPU)")
    return p.parse_args()


def _setup_visible_devices(gpus: str):
    if gpus and isinstance(gpus, str):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def _init_dist(rank: int, world_size: int, master_port: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _worker(rank: int, world_size: int, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    _init_dist(rank, world_size, args.master_port)
    try:
        # Only rank0 creates the output directory first, then writes path back to args for other ranks
        if rank == 0:
            trainer0 = BrepTextContrastiveTrainer(args, rank=0, world_size=world_size, device=device)
            args.run_dir = str(trainer0.out_dir)
            dist.barrier()
            trainer0.train()
        else:
            assert hasattr(args, "run_dir") and args.run_dir is not None, "rank0 did not set run_dir"
            trainer = BrepTextContrastiveTrainer(args, rank=rank, world_size=world_size, device=device)
            trainer.train()
    finally:
        _cleanup_dist()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Select GPUs
    _setup_visible_devices(args.gpus)
    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 0

    # Single GPU or no GPU: use standard path
    if (not use_cuda) or world_size <= 1:
        device = torch.device("cuda" if use_cuda else "cpu")
        trainer = BrepTextContrastiveTrainer(args, rank=0, world_size=1, device=device)
        args.run_dir = str(trainer.out_dir)
        torch.set_float32_matmul_precision('high')
        trainer.train()
        return

    # Multi-GPU: DDP
    torch.set_float32_matmul_precision('high')
    mp.spawn(_worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":

    main()

"""
# Single GPU
python train_brep_text_contrastive.py \
  --data_dir /path/to/bindata \
  --caption_csv /path/to/brepdata_train.csv \
  --output_dir ./output \
  --batch_size 128 --num_workers 4 --epochs 100

# Multi-GPU
python train_brep_text_contrastive.py \
  --gpus 0,1 \
  --data_dir /path/to/bindata \
  --caption_csv /path/to/train.csv \
  --output_dir ./output \
  --batch_size 8 --num_workers 4 --epochs 20
"""
