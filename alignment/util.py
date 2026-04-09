# util.py
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ---------- helpers ----------
def _detect_mask(inp: torch.Tensor):
    """Infer mask position based on channel count; returns None otherwise."""
    C = inp.shape[-1]
    if C in (4, 9, 11):
        return C - 1  # In our pipeline, mask is the last dimension
    # Legacy data compatibility: if C>6, older code places mask at index 6
    if C > 6:
        return 6
    return None

# ---------- bbox / normalization (adaptive mask) ----------
def bounding_box_uvgrid(inp: torch.Tensor):
    """Compute bbox of valid points from uv-grid; adaptive mask channel."""
    pts = inp[..., :3].reshape(-1, 3)
    midx = _detect_mask(inp)
    if midx is not None:
        m = inp[..., midx].reshape(-1)
        pts = pts[m > 0.5]
    # If no mask, compute bbox over all points
    return bounding_box_pointcloud(pts)

def bounding_box_pointcloud(pts: torch.Tensor):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box, dtype=pts.dtype, device=pts.device)

def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    """Scale xyz to [-1,1] range using bbox; only modifies the first 3 dims, leaves other channels/mask untouched."""
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / float(torch.clamp(torch.max(diag), min=1e-6))
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] = (inp[..., :3] - center) * scale
    if return_center_scale:
        return inp, center, scale
    return inp

def get_random_rotation():
    """Randomly select an orthogonal rotation of 0/90/180/270 degrees."""
    axes = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)

# ---------- safe rotation (adaptive normals/tangents and mask) ----------
def rotate_uvgrid(inp: torch.Tensor, rotation) -> torch.Tensor:
    """
    Rotate geometric vectors in a uv-grid tensor:
      - Always rotates xyz (...,:3)
      - If normals/tangents exist (...,3:6), rotates them as well
      - If last dim is 4/9/11 (or mask position is inferred), only rotates elements where mask=1
    rotation: can be scipy.Rotation or 3x3 torch.Tensor
    """
    # Get rotation matrix on the same device/dtype as inp
    if isinstance(rotation, Rotation):
        R = torch.tensor(rotation.as_matrix(), dtype=inp.dtype, device=inp.device)
    else:
        R = torch.as_tensor(rotation, dtype=inp.dtype, device=inp.device)
    assert R.shape == (3, 3), f"Rotation matrix must be 3x3, got {R.shape}"

    out = inp.clone()
    C = inp.shape[-1]
    has_vec2 = C >= 6  # Whether [...,3:6] (normals/tangents) exists
    midx = _detect_mask(inp)
    mask = None if midx is None else (inp[..., midx:midx+1] > 0.5).type_as(inp)

    # Rotate xyz
    xyz = inp[..., :3]
    xyz_r = (xyz.reshape(-1, 3) @ R.T).reshape_as(xyz)
    out[..., :3] = xyz_r if mask is None else (xyz_r * mask + xyz * (1 - mask))

    # Rotate normals/tangents (if present)
    if has_vec2:
        vec = inp[..., 3:6]
        vec_r = (vec.reshape(-1, 3) @ R.T).reshape_as(vec)
        out[..., 3:6] = vec_r if mask is None else (vec_r * mask + vec * (1 - mask))

    return out

# ---------- font filtering ----------
INVALID_FONTS = [
    "Bokor",
    "Lao Muang Khong",
    "Lao Sans Pro",
    "MS Outlook",
    "Catamaran Black",
    "Dubai",
    "HoloLens MDL2 Assets",
    "Lao Muang Don",
    "Oxanium Medium",
    "Rounded Mplus 1c",
    "Moul Pali",
    "Noto Sans Tamil",
    "Webdings",
    "Armata",
    "Koulen",
    "Yinmar",
    "Ponnala",
    "Noto Sans Tamil",
    "Chenla",
    "Lohit Devanagari",
    "Metal",
    "MS Office Symbol",
    "Cormorant Garamond Medium",
    "Chiller",
    "Give You Glory",
    "Hind Vadodara Light",
    "Libre Barcode 39 Extended",
    "Myanmar Sans Pro",
    "Scheherazade",
    "Segoe MDL2 Assets",
    "Siemreap",
    "Signika SemiBold" "Taprom",
    "Times New Roman TUR",
    "Playfair Display SC Black",
    "Poppins Thin",
    "Raleway Dots",
    "Raleway Thin",
    "Segoe MDL2 Assets",
    "Segoe MDL2 Assets",
    "Spectral SC ExtraLight",
    "Txt",
    "Uchen",
    "Yinmar",
    "Almarai ExtraBold",
    "Fasthand",
    "Exo",
    "Freckle Face",
    "Montserrat Light",
    "Inter",
    "MS Reference Specialty",
    "MS Outlook",
    "Preah Vihear",
    "Sitara",
    "Barkerville Old Face",
    "Bodoni MT" "Bokor",
    "Fasthand",
    "HoloLens MDL2 Assests",
    "Libre Barcode 39",
    "Lohit Tamil",
    "Marlett",
    "MS outlook",
    "MS office Symbol Semilight",
    "MS office symbol regular",
    "Ms office symbol extralight",
    "Ms Reference speciality",
    "Segoe MDL2 Assets",
    "Siemreap",
    "Sitara",
    "Symbol",
    "Wingdings",
    "Metal",
    "Ponnala",
    "Webdings",
    "Souliyo Unicode",
    "Aguafina Script",
    "Yantramanav Black",
]

def valid_font(filename):
    for name in INVALID_FONTS:
        if name.lower() in str(filename).lower():
            return False
    return True
