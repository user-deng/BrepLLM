#!/usr/bin/env python3
#brep_text_dataset.py
import pathlib, gc, torch, dgl, random
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import load_graphs
from tqdm import tqdm

# Depends on util.py in the same directory
try:
    from util import rotate_uvgrid, get_random_rotation
except ImportError:
    print("="*50)
    print("!! Warning: util.py not found !!")
    print("!! Random rotation (random_rotate=True) will not work. !!")
    print("="*50)
    # Define placeholder functions so the code can at least be loaded without util.py
    def get_random_rotation():
        return np.identity(3) # Return a 3x3 identity matrix
    def rotate_uvgrid(inp, rotation):
        print("Warning: util.py not loaded, rotate_uvgrid not executed.")
        return inp

class BrepTextDataset(Dataset):
    """
    BREP-Text alignment pre-training dataset (adapted for pre-padded .bin files)

     - Reads .bin graph data pre-processed and padded by step_to_bin.py.
     - .bin files should contain:
        - g.ndata['x_full']: [F, S, S, 11] (face features: pts, nrm, vis, mean, type, area, MASK)
        - g.ndata['x']:      [F, S, S, 4]  (face simple: pts, MASK)
        - g.edata['x_full']: [E, C, 9]   (edge features: pts, tan, type, len, MASK)
        - g.edata['x']:      [E, C, 4]   (edge simple: pts, MASK)
     - (Optional) Centers and scales all graph XYZ coordinates.
     - (Optional) Applies random rotation data augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        caption_file: str,
        center_and_scale: bool = True,
        random_rotate: bool = False,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.center_and_scale = center_and_scale
        self.random_rotate = random_rotate

        print(f"Loading caption CSV: {caption_file}")
        try:
            self.caption_df = pd.read_csv(caption_file)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

        print("Pre-loading all graphs ...")
        self.preloaded_data = []
        self._load_and_preprocess_all_data()
        print(f"Pre-loaded {len(self.preloaded_data)} brep-text pairs")

    # --------------------------------------------------------------------- #
    #                           INTERNAL HELPERS
    # --------------------------------------------------------------------- #

    def _center_scale_xyz_inplace(self, g: dgl.DGLGraph):
        """
        (Robust version) Normalizes only xyz coordinates; preserves other channels and mask.
        Face and edge share the same (center, scale) to ensure consistency within the graph.

        Assumptions:
        - 'x'      (ndata/edata) [..., :3] is xyz, [..., 3:4] is mask
        - 'x_full' (ndata) [..., :3] is xyz, [..., 10:11] is mask
        - 'x_full' (edata) [..., :3] is xyz, [..., 8:9] is mask
        """
        # Use 'x' (simple version) to compute statistics as it is more lightweight
        if "x" not in g.ndata or g.ndata["x"].numel() == 0:
            return

        face_x = g.ndata["x"]             # [F, Su, Sv, 4]
        f_xyz  = face_x[..., :3]
        f_m    = face_x[..., 3:4]

        # Find all valid face points (mask > 0.5)
        valid_f_mask = f_m > 0.5
        if not valid_f_mask.any():
            return # No valid points in the graph

        # 1. Compute center (mu) and scale
        # Use more robust bbox center
        valid_points = f_xyz[valid_f_mask.squeeze(-1)]
        if valid_points.numel() == 0:
            return # Avoid empty tensor

        v_min = valid_points.min(dim=0)[0]
        v_max = valid_points.max(dim=0)[0]

        mu = (v_min + v_max) / 2.0
        # Expand mu for broadcasting
        mu = mu.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, 3]

        # Scale: use max bbox edge length
        scale = (v_max - v_min).max().clamp(min=1e-6)

        # 2. Apply to faces (ndata)
        face_x[..., :3] = torch.where(valid_f_mask, (f_xyz - mu) / scale, f_xyz)
        g.ndata["x"] = face_x

        if "x_full" in g.ndata:
            fx = g.ndata["x_full"]                  # [F, Su, Sv, 11]
            fx_xyz = fx[..., :3]
            fx_m = fx[..., 10:11] > 0.5             # [F, Su, Sv, 1]
            fx[..., :3] = torch.where(fx_m, (fx_xyz - mu) / scale, fx_xyz)
            g.ndata["x_full"] = fx

        # 3. Apply to edges (edata)
        if "x" in g.edata and g.edata["x"].numel() > 0:
            e = g.edata["x"]                        # [E, Cu, 4]
            e_xyz, e_m = e[..., :3], e[..., 3:4] > 0.5
            # Reshape mu to match [E, Cu, 3]
            mu_edge = mu.squeeze(0).squeeze(0)      # [1, 3]
            e[..., :3] = torch.where(e_m, (e_xyz - mu_edge) / scale, e_xyz)
            g.edata["x"] = e

        if "x_full" in g.edata and g.edata["x_full"].numel() > 0:
            ef = g.edata["x_full"]                  # [E, Cu, 9]
            ef_xyz, ef_m = ef[..., :3], ef[..., 8:9] > 0.5
            mu_edge = mu.squeeze(0).squeeze(0)      # [1, 3]
            ef[..., :3] = torch.where(ef_m, (ef_xyz - mu_edge) / scale, ef_xyz)
            g.edata["x_full"] = ef


    def _load_and_preprocess_all_data(self):
        """
        Load all .bin files into memory.
        Data has already been processed and padded in the .bin files.
        Only (optional) centering/scaling is applied here.
        """
        valid_cnt = 0
        for _, row in tqdm(self.caption_df.iterrows(), total=len(self.caption_df), desc="Loading BINs"):
            uid = str(row["uid"])
            bin_path = self.data_dir / f"{uid}.bin"
            if not bin_path.exists():
                continue

            try:
                # load_graphs returns (graphs_list, labels_dict)
                g_list, _ = load_graphs(str(bin_path))
                if not g_list:
                    print(f"[Warn] {uid} -> load_graphs returned empty list.")
                    continue
                g = g_list[0]

                # Validate graph (has nodes and 'x_full' features)
                if g.num_nodes() == 0:
                    continue
                if "x_full" not in g.ndata or g.ndata["x_full"].numel() == 0:
                    print(f"[Warn] {uid} -> Graph has no 'x_full' node features, skipping.")
                    continue
                # Edge features can be 0 (e.g., single face)
                if g.num_edges() > 0 and ("x_full" not in g.edata or g.edata["x_full"].numel() == 0):
                    print(f"[Warn] {uid} -> Graph has edges but no 'x_full' edge features, skipping.")
                    continue

                # Convert all features to float32
                for key in g.ndata: g.ndata[key] = g.ndata[key].float()
                for key in g.edata: g.edata[key] = g.edata[key].float()


                # (Optional) Apply centering and scaling
                if self.center_and_scale:
                    self._center_scale_xyz_inplace(g)

                # Get caption text
                caption = row.get("beginner", "")
                if isinstance(caption, str) and caption.strip():
                    self.preloaded_data.append({"graph": g, "caption": caption, "uid": uid})
                    valid_cnt += 1

            except Exception as e:
                print(f"[Error] Failed to load or process {uid}: {e}")
                continue

        print(f"Successfully loaded {valid_cnt} valid samples")
        random.shuffle(self.preloaded_data)
        gc.collect()

    # --------------------------------------------------------------------- #
    #                           DUNDER METHODS
    # --------------------------------------------------------------------- #
    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, idx):
        sample = self.preloaded_data[idx]
        g = sample["graph"]
        caption = sample["caption"]
        uid = sample["uid"]

        # (Optional) Apply random rotation data augmentation
        if self.random_rotate:
            g = g.clone() # Must clone to avoid modifying pre-loaded data
            try:
                R = get_random_rotation()
                # Rotation modifies g.ndata and g.edata in-place
                # rotate_uvgrid in util.py handles both 'x' and 'x_full' formats

                # Rotate faces
                if "x" in g.ndata:
                    g.ndata["x"] = rotate_uvgrid(g.ndata["x"], R)
                if "x_full" in g.ndata:
                    g.ndata["x_full"] = rotate_uvgrid(g.ndata["x_full"], R)

                # Rotate edges
                if "x" in g.edata:
                    g.edata["x"] = rotate_uvgrid(g.edata["x"], R)
                if "x_full" in g.edata:
                    g.edata["x_full"] = rotate_uvgrid(g.edata["x_full"], R)

            except Exception as e:
                print(f"Warning: Failed to apply random rotation to {uid}: {e}")

        return {"graph": g, "caption": caption, "uid": uid}

    # --------------------------------------------------------------------- #
    #                         DATALOADER UTILITIES
    # --------------------------------------------------------------------- #
    @staticmethod
    def collate_fn(batch):
        if not batch:
            return None

        # Filter out None items (in case __getitem__ returns None on error)
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        graphs = [b["graph"] for b in batch]
        captions = [b["caption"] for b in batch]
        uids = [b["uid"] for b in batch]

        try:
            batched_graph = dgl.batch(graphs)
        except Exception as e:
            print(f"[Collate Error] Failed to batch graphs: {e}")
            # Try to identify the problematic graph
            for i, g in enumerate(graphs):
                print(f"  Graph {i} (uid: {uids[i]}): {g.num_nodes()} nodes, {g.num_edges()} edges")
                if "x_full" in g.ndata:
                    print(f"    ndata['x_full']: {g.ndata['x_full'].shape}")
                if "x_full" in g.edata and g.num_edges() > 0:
                    print(f"    edata['x_full']: {g.edata['x_full'].shape}")
            return None # Return None to skip this batch

        return {"graph": batched_graph, "captions": captions, "uids": uids}

    def get_dataloader(
        self,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=num_workers > 0,
        )

# --------------------------- quick test ---------------------------------- #
if __name__ == "__main__":

    # *** Make sure these paths point to your generated .bin data ***
    DATA_DIR = "/path/to/bindata"
    CAPTION_FILE = "/path/to/brepdata_train.csv"

    print(f"Testing Dataset Loader...")
    print(f"BIN Path: {DATA_DIR}")
    print(f"CSV Path: {CAPTION_FILE}")

    ds = BrepTextDataset(
        data_dir=DATA_DIR,
        caption_file=CAPTION_FILE,
        center_and_scale=True,
        random_rotate=True,
    )

    if len(ds) == 0:
        print("\n--- Error: Dataset is empty! ---")
        print("Please check:")
        print(f"1. Is {DATA_DIR} path correct?")
        print(f"2. Is {CAPTION_FILE} path correct?")
        print(f"3. Have .bin files been generated?")
        print(f"4. Do 'uid' values in CSV match .bin file names?")
    else:
        print(f"\nDataset loaded successfully, size: {len(ds)}")

        sample = ds[0]
        g0 = sample["graph"]
        print(f"\n--- Single sample (UID: {sample['uid']}) ---")
        print(f"Graph: {g0}")

        print("\n--- Feature shape check (from g.ndata/g.edata) ---")
        if "x" in g0.ndata:
            print(f"Face simple (x) shape : {g0.ndata['x'].shape}") # Should be [F, S, S, 4]
        if "x_full" in g0.ndata:
            print(f"Face full   (x_full): {g0.ndata['x_full'].shape}") # Should be [F, S, S, 11]

        if g0.num_edges() > 0:
            if "x" in g0.edata:
                print(f"Edge simple (x) shape : {g0.edata['x'].shape}") # Should be [E, C, 4]
            if "x_full" in g0.edata:
                print(f"Edge full   (x_full): {g0.edata['x_full'].shape}") # Should be [E, C, 9]
        else:
            print("Graph has 0 edges.")

        print("\n--- Dataloader batch test ---")
        dl = ds.get_dataloader(batch_size=4, shuffle=False)
        batch = next(iter(dl), None)

        if batch is None:
            print("Dataloader returned None, batching failed.")
        else:
            print(f"Dataloader batching succeeded.")
            print(f"Captions in batch: {len(batch['captions'])}")
            print(f"Batched graph: {batch['graph']}")

            # Check batched feature shapes
            b_ndata_x_full = batch['graph'].ndata['x_full']
            b_edata_x_full = batch['graph'].edata.get('x_full', None)

            print(f"Batched face full : {b_ndata_x_full.shape}") # [Total_F, S, S, 11]
            if b_edata_x_full is not None:
                print(f"Batched edge full : {b_edata_x_full.shape}") # [Total_E, C, 9]

            # Check normalization (center_and_scale=True)
            # xyz coordinates should be roughly in [-1, 1] range
            f_xyz = b_ndata_x_full[..., :3]
            f_mask = b_ndata_x_full[..., 10:11] > 0.5
            valid_f_xyz = f_xyz[f_mask.squeeze(-1)]

            if valid_f_xyz.numel() > 0:
                print("\n--- Normalization check (center_and_scale=True) ---")
                print(f"Valid XYZ Min: {valid_f_xyz.min().item():.4f}")
                print(f"Valid XYZ Max: {valid_f_xyz.max().item():.4f}")
                print(f"Valid XYZ Mean: {valid_f_xyz.mean().item():.4f}")
            else:
                print("No valid points found in batch.")
