from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.data.augmentation import augment_pair
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("augment_dataset")
OUTPUT_FIELDS = [
    "source_image", "source_mask", "augmented_image", "augmented_mask",
    "copy_index", "source_leak_pixels", "augmented_leak_pixels",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def stable_seed(base_seed: int, relative: Path, copy_index: int) -> int:
    value = f"{base_seed}|{relative.as_posix()}|{copy_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paired offline augmentation for Train only.")
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--images", default="dataset/prepared/images/train")
    parser.add_argument("--masks", default="dataset/prepared/masks/train")
    parser.add_argument("--manifest", default="dataset/metadata/augmentation_result.csv")
    parser.add_argument("--copies", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = require(load_yaml(args.config), "augmentation")
    copies = args.copies if args.copies is not None else int(config["copies_per_image"])
    if copies < 1:
        LOG.error("copies must be positive")
        return 2
    image_root, mask_root = Path(args.images), Path(args.masks)
    if not image_root.is_dir() or not mask_root.is_dir():
        LOG.error("Train image or mask root does not exist")
        return 2
    originals = sorted(
        p for p in image_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and "__aug" not in p.stem
    )
    rows: list[dict[str, object]] = []
    failures = 0
    for image_path in originals:
        relative = image_path.relative_to(image_root)
        mask_path = mask_root / relative.with_suffix(".png")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            LOG.error("Invalid train pair: %s", relative.as_posix())
            failures += 1
            continue
        for copy_index in range(1, copies + 1):
            rng = np.random.default_rng(stable_seed(int(config["seed"]), relative, copy_index))
            augmented_image, augmented_mask = augment_pair(image, mask, rng, config)
            image_output = image_path.with_name(f"{image_path.stem}__aug{copy_index:02d}{image_path.suffix}")
            mask_output = mask_path.with_name(f"{mask_path.stem}__aug{copy_index:02d}.png")
            image_params = (
                [cv2.IMWRITE_JPEG_QUALITY, 95]
                if image_output.suffix.lower() in {".jpg", ".jpeg"}
                else []
            )
            if not cv2.imwrite(str(image_output), augmented_image, image_params):
                failures += 1
                continue
            if not cv2.imwrite(str(mask_output), augmented_mask):
                failures += 1
                continue
            rows.append({
                "source_image": relative.as_posix(),
                "source_mask": relative.with_suffix(".png").as_posix(),
                "augmented_image": image_output.relative_to(image_root).as_posix(),
                "augmented_mask": mask_output.relative_to(mask_root).as_posix(),
                "copy_index": copy_index,
                "source_leak_pixels": int(np.count_nonzero(mask)),
                "augmented_leak_pixels": int(np.count_nonzero(augmented_mask)),
            })
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info(
        "Original train pairs=%d augmented pairs=%d failures=%d",
        len(originals), len(rows), failures,
    )
    LOG.info("Augmentation manifest written to %s", manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
