import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import ignite_unet_pipeline as p

records = []

for fold_idx, (_, val_idx) in enumerate(p.folds):
    val_df = p.df.iloc[val_idx]
    bg_pixels = 0
    total_pixels = 0
    image_bg = []

    for _, row in val_df.iterrows():
        subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
        stem = Path(row["name"]).stem
        mask_path = Path(p.DATA_ROOT) / subtype / "masks" / f"{stem}.png"

        mask = np.array(Image.open(mask_path))
        bg = int(np.count_nonzero(mask == 0))
        total = mask.size

        bg_pixels += bg
        total_pixels += total
        image_bg.append(100 * bg / total)

    records.append({
        "fold": fold_idx,
        "images": len(val_df),
        "background_pct": 100 * bg_pixels / total_pixels,
        "images_with_background": sum(x > 0 for x in image_bg),
        "images_above_5pct_bg": sum(x >= 5 for x in image_bg),
        "max_image_bg_pct": max(image_bg)
    })

results = pd.DataFrame(records)
print("\nBACKGROUND DISTRIBUTION BY FOLD")
print(results.to_string(index=False))

results.to_csv(
    Path(p.OUTPUT_DIR) / "fold_background_distribution.csv",
    index=False
)
