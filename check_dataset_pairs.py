from pathlib import Path
import pandas as pd

root = Path(r"D:\microscope project\Data")
df = pd.read_csv(root / "data_overview_patient_folds.csv")

df = df[
    (df["task"] == "he_tissue_segmentation")
    & (df["split"] == "train")
].copy()

valid = []
missing_images = []
missing_masks = []

for _, row in df.iterrows():
    subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
    if subtype == "nan":
        subtype = "unknown"

    stem = Path(row["name"]).stem
    image = root / "organised_he" / subtype / "images" / f"{stem}.png"
    mask = root / "organised_he" / subtype / "masks" / f"{stem}.png"

    if not image.is_file():
        missing_images.append(stem)

    if not mask.is_file():
        missing_masks.append(stem)

    if image.is_file() and mask.is_file():
        valid.append(row)

valid_df = pd.DataFrame(valid)

print("Manifest training entries:", len(df))
print("Valid image-mask pairs:", len(valid_df))
print("Missing images:", len(missing_images))
print("Missing masks:", len(missing_masks))

if len(valid_df):
    print("\nValid pairs by fold:")
    print(valid_df.groupby("validation_fold").size().to_string())

    print("\nValid pairs by subtype:")
    print(valid_df.groupby("histological_subtype").size().to_string())

print("\nFirst 10 missing images:", missing_images[:10])
