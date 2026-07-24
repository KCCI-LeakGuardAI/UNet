from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.utils.logger import configure_logging

LOG = logging.getLogger("validate_dataset")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def keyed_files(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one-to-one image and binary-mask pairs.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--max-empty-ratio", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    image_dir, mask_dir = Path(args.images), Path(args.masks)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        LOG.error("Both --images and --masks must be existing directories")
        return 2
    images, masks = keyed_files(image_dir), keyed_files(mask_dir)
    errors: list[str] = []
    for stem in sorted(images.keys() - masks.keys()):
        errors.append(f"missing mask: {images[stem].name}")
    for stem in sorted(masks.keys() - images.keys()):
        errors.append(f"missing image: {masks[stem].name}")
    empty = 0
    ratios: list[float] = []
    for stem in sorted(images.keys() & masks.keys()):
        image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            errors.append(f"unreadable pair: {stem}")
            continue
        if mask.ndim != 2:
            errors.append(f"mask is not single-channel: {masks[stem].name}")
            continue
        if image.shape[:2] != mask.shape:
            errors.append(f"size mismatch: {stem} image={image.shape[:2]} mask={mask.shape}")
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            errors.append(f"mask is not binary: {masks[stem].name} values={sorted(values)[:10]}")
        ratio = float(np.count_nonzero(mask)) / mask.size * 100.0
        ratios.append(ratio)
        empty += int(ratio == 0)
    pair_count = len(images.keys() & masks.keys())
    empty_ratio = empty / pair_count if pair_count else 0.0
    if empty_ratio > args.max_empty_ratio:
        errors.append(f"empty-mask ratio {empty_ratio:.3f} exceeds {args.max_empty_ratio:.3f}")
    for error in errors:
        LOG.error(error)
    if ratios:
        LOG.info("pairs=%d empty=%d leak_ratio[min/mean/max]=%.4f/%.4f/%.4f%%",
                 pair_count, empty, min(ratios), float(np.mean(ratios)), max(ratios))
    else:
        LOG.info("pairs=0")
    LOG.info("Validation %s with %d error(s)", "FAILED" if errors else "PASSED", len(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

