import numpy as np
from pathlib import Path
from tqdm import tqdm

TARGET_DIR = Path("processed_data") / "Semitic"

def main():
    if not TARGET_DIR.exists():
        print("Folder not found:", TARGET_DIR)
        return

    print("Scanning:", TARGET_DIR.resolve())

    bad = []
    files = list(TARGET_DIR.glob("*.npy"))

    for file in tqdm(files):
        arr = np.load(file, mmap_mode='r')  # FAST
        if arr.shape != (128, 125):
            bad.append((file.name, arr.shape))

    print(f"\nTotal files scanned: {len(files)}")

    if bad:
        print("\n❌ Wrong shapes:")
        for name, shape in bad[:20]:
            print(f"  {name}: {shape}")
        if len(bad) > 20:
            print(f"... and {len(bad)-20} more")
    else:
        print("\n✅ All shapes correct")

if __name__ == "__main__":
    main()
