# ============================================================
# CONFIG – edit these three lines only
# ============================================================
CSV_PATH   = r"E:\ignite tool kit he segmentation\Data\data_overview.csv"          # ← your CSV
DATA_ROOT  = r"E:\ignite tool kit he segmentation\Data"        # ← folder that contains images/ and annotations/
OUTPUT_DIR = r"E:\ignite tool kit he segmentation\Data"               # ← where results will be written
ORGAN      = "lung"                                   # set to "" if you want all organs
OVERWRITE  = False                                    # True = re-write existing files
# ============================================================

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm   # works nicely in Spyder

def load_annotation_as_mask(ann_path: Path) -> np.ndarray:
    """Convert annotation PNG → clean integer mask."""
    ann = np.array(Image.open(ann_path))

    if ann.ndim == 2:                       # already a label map
        mask = ann.astype(np.int32)
    elif ann.ndim == 3:                     # RGB colour map
        flat = ann.reshape(-1, ann.shape[-1])
        colours, inverse = np.unique(flat, axis=0, return_inverse=True)
        mask = inverse.reshape(ann.shape[:2]).astype(np.int32)
        # force pure black → background 0
        mask[np.all(ann == 0, axis=-1)] = 0
    else:
        raise ValueError(f"Unsupported annotation shape: {ann.shape}")
    return mask


# ------------------------------------------------------------------
# 1. Load & filter
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["task"] == "he_tissue_segmentation"].copy()

if ORGAN:
    df = df[df["organ"] == ORGAN]

print(f"Found {len(df)} H&E ROIs (organ = '{ORGAN or 'all'}')")
print(df["histological_subtype"].value_counts().to_string())
print()

# ------------------------------------------------------------------
# 2. Process every ROI
# ------------------------------------------------------------------
output = Path(OUTPUT_DIR)
output.mkdir(parents=True, exist_ok=True)

data_root = Path(DATA_ROOT)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
    if not subtype or subtype == "nan":
        subtype = "unknown"

    img_dir  = output / subtype / "images"
    mask_dir = output / subtype / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # ----- locate source files -----
    img_src = Path(row["image_path"])
    ann_src = Path(row["annotation_path"])

    # try absolute path first, otherwise relative to DATA_ROOT
    if not img_src.is_file():
        img_src = data_root / img_src.name if img_src.name else data_root / img_src
        # more robust: strip the leading /ignite_data_toolkit part
        try:
            rel = Path(*Path(row["image_path"]).parts[1:])   # drop drive / first folder
            img_src = data_root / rel
        except Exception:
            pass

    if not ann_src.is_file():
        try:
            rel = Path(*Path(row["annotation_path"]).parts[1:])
            ann_src = data_root / rel
        except Exception:
            pass

    if not img_src.is_file():
        print(f"[WARN] missing image: {row['image_path']}")
        continue
    if not ann_src.is_file():
        print(f"[WARN] missing annotation: {row['annotation_path']}")
        continue

    # ----- destination names -----
    stem     = Path(row["name"]).stem          # e.g. patient1_he_roi1
    img_dst  = img_dir  / f"{stem}.png"
    mask_dst = mask_dir / f"{stem}.png"

    if img_dst.exists() and not OVERWRITE:
        continue

    # copy image
    shutil.copy2(img_src, img_dst)

    # create clean mask
    mask = load_annotation_as_mask(ann_src)
    Image.fromarray(mask.astype(np.uint8)).save(mask_dst)

print(f"\nDone → {output.resolve()}")