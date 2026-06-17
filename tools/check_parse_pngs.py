import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--max_show", type=int, default=20)
    args = parser.parse_args()

    parse_dir = Path(args.parse_dir)
    files = []
    for ext in IMG_EXTS:
        files.extend(parse_dir.glob(f"*{ext}"))
    files = sorted(files)

    assert files, f"No parse files found in {parse_dir}"

    global_counter = Counter()
    bad_files = []

    for idx, path in enumerate(files):
        img = Image.open(path)
        arr = np.array(img)

        unique = np.unique(arr)
        global_counter.update(unique.tolist())

        ok = (
            arr.ndim == 2
            and unique.min() >= 0
            and unique.max() < args.num_classes
            and np.all(unique.astype(int) == unique)
        )

        if not ok:
            bad_files.append((path.name, img.mode, arr.shape, unique[:50].tolist(), int(unique.min()), int(unique.max())))

        if idx < args.max_show:
            print(
                f"[{idx}] {path.name} | mode={img.mode} | shape={arr.shape} | "
                f"dtype={arr.dtype} | min={arr.min()} | max={arr.max()} | unique={unique.tolist()}"
            )

    print("\n====== Summary ======")
    print(f"num_files = {len(files)}")
    print(f"global_classes = {sorted(global_counter.keys())}")
    print(f"bad_files = {len(bad_files)}")

    if bad_files:
        print("\nBad examples:")
        for item in bad_files[:20]:
            print(item)

    assert len(bad_files) == 0, "Some parse png files are invalid."

if __name__ == "__main__":
    main()