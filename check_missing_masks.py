from pathlib import Path
import pandas as pd

root = Path(r"D:\microscope project\Data")
df = pd.read_csv(root / "data_overview_patient_folds.csv")
df = df[
    (df["task"] == "he_tissue_segmentation")
    & (df["split"] == "train")
]

missing = []

for _, row in df.iterrows():
    subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
    if subtype == "nan":
        subtype = "unknown"

    stem = Path(row["name"]).stem
    expected = root / "organised_he" / subtype / "masks" / f"{stem}.png"

    if not expected.is_file():
        matches = list((root / "organised_he").rglob(f"{stem}.png"))
        mask_matches = [p for p in matches if p.parent.name == "masks"]

        missing.append({
            "image": stem,
            "expected_subtype": subtype,
            "found_masks": "; ".join(str(p) for p in mask_matches) or "NOT FOUND"
        })

result = pd.DataFrame(missing)
print(result.to_string(index=False))
print("\nMissing entries:", len(result))
print("Found in another folder:", sum(result["found_masks"] != "NOT FOUND"))
