from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import ignite_unet_pipeline as p

train_idx, val_idx = p.folds[1]
train_df = p.df.iloc[train_idx]

records = []

for _, row in train_df.iterrows():
    subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
    stem = Path(row["name"]).stem

    mask_path = Path(p.DATA_ROOT) / subtype / "masks" / f"{stem}.png"

    mask = np.asarray(Image.open(mask_path))
    bg_pct = 100.0 * np.mean(mask == 0)

    records.append({
        "name": row["name"],
        "subtype": subtype,
        "background_pct": bg_pct
    })

results = pd.DataFrame(records).sort_values(
    "background_pct", ascending=False
)

print("\nTOP 15 BACKGROUND-RICH TRAINING IMAGES")
print(results.head(15).to_string(index=False))

print("\nTraining images with >= 5% background:",
      int((results["background_pct"] >= 5).sum()))

print("Training images with >= 1% background:",
      int((results["background_pct"] >= 1).sum()))
