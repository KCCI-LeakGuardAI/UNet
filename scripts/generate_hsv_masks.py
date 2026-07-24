from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from src.data.mask_processing import generate_hsv_mask, make_overlay
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("generate_hsv_masks")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HSV-based ground-truth mask candidates.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-images", default="dataset/images_unreviewed")
    parser.add_argument("--output-masks", default="dataset/masks_unreviewed")
    parser.add_argument("--output-overlays", default="dataset/overlays")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_yaml(args.config)
    hsv, morph = require(config, "hsv"), require(config, "morphology")
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        LOG.error("Input directory not found: %s", input_dir)
        return 2
    outputs = [Path(args.output_images), Path(args.output_masks), Path(args.output_overlays)]
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    failures = 0
    for image_path in files:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            LOG.error("Unreadable image: %s", image_path)
            failures += 1
            continue
        try:
            mask = generate_hsv_mask(
                image, list(hsv["lower"]), list(hsv["upper"]),
                int(morph["kernel_size"]), int(morph["open_iterations"]),
                int(morph["close_iterations"]), int(morph["min_component_pixels"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOG.error("Mask configuration error: %s", exc)
            return 3
        shutil.copy2(image_path, outputs[0] / image_path.name)
        if not cv2.imwrite(str(outputs[1] / f"{image_path.stem}.png"), mask):
            failures += 1
        if not cv2.imwrite(str(outputs[2] / f"{image_path.stem}.png"), make_overlay(image, mask)):
            failures += 1
    LOG.info("Processed %d images (%d failures)", len(files), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

