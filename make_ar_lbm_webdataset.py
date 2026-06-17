import os
import re
import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
import webdataset as wds


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_image(folder: Path, stem: str):
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def pil_to_bytes(img: Image.Image, fmt: str):
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def natural_sort_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch_dir", type=str, required=True)
    parser.add_argument("--photo_dir", type=str, required=True)
    parser.add_argument("--out_pattern", type=str, required=True)
    parser.add_argument("--maxcount", type=int, default=1000)

    # 新增：face parsing label map 文件夹
    parser.add_argument(
        "--parse_dir",
        type=str,
        default=None,
        help="Folder containing face parsing label maps. Filenames should match sketch/photo stems.",
    )

    # 如果开启，则没有 parse 的样本直接跳过
    parser.add_argument(
        "--require_parse",
        action="store_true",
        help="Skip samples without parse map. Recommended when training Face Parsing Adapter.",
    )

    args = parser.parse_args()

    sketch_dir = Path(args.sketch_dir)
    photo_dir = Path(args.photo_dir)
    parse_dir = Path(args.parse_dir) if args.parse_dir is not None else None

    sketch_files = []
    for ext in IMG_EXTS:
        sketch_files.extend(sketch_dir.glob(f"*{ext}"))
    sketch_files = sorted(sketch_files, key=natural_sort_key)

    assert len(sketch_files) > 0, f"No images found in {sketch_dir}"

    if parse_dir is not None:
        assert parse_dir.exists(), f"parse_dir does not exist: {parse_dir}"

    out_parent = Path(args.out_pattern).parent
    out_parent.mkdir(parents=True, exist_ok=True)

    sink = wds.ShardWriter(args.out_pattern, maxcount=args.maxcount)


    count = 0
    skipped = 0
    skipped_parse = 0

    for sketch_path in sketch_files:
        stem = sketch_path.stem
        photo_path = find_image(photo_dir, stem)

        if photo_path is None:
            print(f"[Skip] No paired photo for {sketch_path.name}")
            skipped += 1
            continue

        parse_path = None
        if parse_dir is not None:
            parse_path = find_image(parse_dir, stem)

            if parse_path is None:
                msg = f"[Skip parse] No paired parse for {sketch_path.name}"
                if args.require_parse:
                    print(msg)
                    skipped_parse += 1
                    continue
                else:
                    print(msg + " | continue without parse")

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        sketch = sketch.resize((256, 256), Image.BICUBIC)
        photo = photo.resize((256, 256), Image.BICUBIC)

        mask = Image.new("L", (256, 256), 255)

        sample = {
            "__key__": stem,
            "jpg": pil_to_bytes(sketch, "JPEG"),
            "normal_aligned.png": pil_to_bytes(photo, "PNG"),
            "mask.png": pil_to_bytes(mask, "PNG"),
        }

        # 新增：parse label map
        # 注意：parse 是单通道类别图，像素值为 0~18，不能用 BICUBIC
        if parse_path is not None:
            parse = Image.open(parse_path).convert("L")
            parse = parse.resize((256, 256), Image.NEAREST)
            sample["parse.png"] = pil_to_bytes(parse, "PNG")

        sink.write(sample)
        count += 1

    sink.close()
    print(f"Done. written={count}, skipped_photo={skipped}, skipped_parse={skipped_parse}")


if __name__ == "__main__":
    main()