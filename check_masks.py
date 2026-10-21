
from pathlib import Path
from PIL import Image
import numpy as np

root = Path(r"D:\microscope project\Data\organised_he")

if not root.exists():
    print("Dataset folder not found:", root)
    raise SystemExit(1)

total = 0

for folder in root.rglob("masks"):
    print("\nChecking:", folder)

    for path in sorted(folder.glob("*.png"))[:10]:
        mask = np.array(Image.open(path))
        binary = mask > 0

        tissue = binary.mean() * 100
        background = 100 - tissue

        print(
            f"{path.name}: "
            f"Tissue={tissue:.2f}% "
            f"Background={background:.2f}%"
        )
        total += 1

print("\nTotal masks checked:", total)
