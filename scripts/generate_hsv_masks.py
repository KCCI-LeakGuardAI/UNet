from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.data.mask_processing import generate_hsv_mask, make_overlay
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("generate_hsv_masks")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MANIFEST_FIELDS = [
    "relative_image", "relative_mask", "relative_overlay",
    "width", "height", "leak_pixels", "leak_ratio",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HSV-based ground-truth mask candidates.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-images", default="dataset/images_unreviewed")
    parser.add_argument("--output-masks", default="dataset/masks_unreviewed")
    parser.add_argument("--output-overlays", default="dataset/overlays")
    parser.add_argument(
        "--manifest",
        default="dataset/metadata/mask_generation.csv",
        help="CSV containing generated image/mask pairs and mask area",
    )
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
    manifest_rows: list[dict[str, object]] = []
    for image_path in files:
        relative = image_path.relative_to(input_dir)
        relative_png = relative.with_suffix(".png")
        output_image = outputs[0] / relative
        output_mask = outputs[1] / relative_png
        output_overlay = outputs[2] / relative_png
        for parent in (output_image.parent, output_mask.parent, output_overlay.parent):
            parent.mkdir(parents=True, exist_ok=True)
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
                bool(morph.get("fill_holes", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOG.error("Mask configuration error: %s", exc)
            return 3
        shutil.copy2(image_path, output_image)
        if not cv2.imwrite(str(output_mask), mask):
            failures += 1
        if not cv2.imwrite(str(output_overlay), make_overlay(image, mask)):
            failures += 1
        leak_pixels = int(np.count_nonzero(mask))
        manifest_rows.append({
            "relative_image": relative.as_posix(),
            "relative_mask": relative_png.as_posix(),
            "relative_overlay": relative_png.as_posix(),
            "width": image.shape[1],
            "height": image.shape[0],
            "leak_pixels": leak_pixels,
            "leak_ratio": f"{leak_pixels / mask.size * 100.0:.6f}",
        })
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    LOG.info("Processed %d images (%d failures)", len(files), failures)
    LOG.info("Manifest written to %s", manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
