from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

root = Path(r"D:\microscope project\Data")
df = pd.read_csv(root / "data_overview_patient_folds.csv")

df = df[
    (df["task"] == "he_tissue_segmentation")
    & (df["split"] == "train")
]

total_tissue = 0
total_background = 0
results = []
missing = []

for _, row in df.iterrows():
    subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
    if subtype == "nan":
        subtype = "unknown"

    stem = Path(row["name"]).stem
    path = root / "organised_he" / subtype / "masks" / f"{stem}.png"

    if not path.exists():
        missing.append(str(path))
        continue

    mask = np.array(Image.open(path).convert("L"))
    tissue = int(np.count_nonzero(mask > 0))
    background = int(mask.size - tissue)

    total_tissue += tissue
    total_background += background

    results.append({
        "image": stem,
        "subtype": subtype,
        "background_percent": 100 * background / mask.size
    })

print("Masks analyzed:", len(results))
print("Missing masks:", len(missing))

if results:
    summary = pd.DataFrame(results)
    print("\nBackground percentage statistics:")
    print(summary["background_percent"].describe().to_string())

    total = total_tissue + total_background
    print(f"\nTotal tissue pixels: {total_tissue:,}")
    print(f"Total background pixels: {total_background:,}")
    print(f"Tissue percentage: {100 * total_tissue / total:.4f}%")
    print(f"Background percentage: {100 * total_background / total:.4f}%")

    print("\nBy histological subtype:")
    print(summary.groupby("subtype")["background_percent"].agg(
        ["count", "mean", "min", "max"]
    ).to_string())

if missing:
    print("\nFirst missing masks:")
    print("\n".join(missing[:10]))
