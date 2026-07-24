from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.data.roi import Roi
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("analyze_dataset")
OUTPUT_FIELDS = [
    "filename", "session_id", "drop_count", "class_name",
    "leak_pixels", "roi_pixels", "leak_ratio",
]


def class_name(drop_count: int) -> str:
    if drop_count == 0:
        return "NORMAL"
    if drop_count <= 2:
        return "SMALL_LEAK"
    if drop_count <= 5:
        return "WARNING"
    return "DANGER"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate leak-area statistics by drop-count group.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--roi-config", default="configs/roi.yaml")
    parser.add_argument("--output", default="artifacts/dataset_analysis.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    metadata, mask_dir = Path(args.metadata), Path(args.masks)
    if not metadata.is_file() or not mask_dir.is_dir():
        LOG.error("Metadata file or mask directory does not exist")
        return 2
    monitoring = Roi.from_mapping(require(load_yaml(args.roi_config), "monitoring_roi"))
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    with metadata.open("r", newline="", encoding="utf-8-sig") as stream:
        for item in csv.DictReader(stream):
            mask_path = mask_dir / f"{Path(item['filename']).stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                missing.append(str(mask_path))
                continue
            monitoring.validate(mask.shape[1], mask.shape[0])
            crop = mask[monitoring.y:monitoring.y + monitoring.height,
                        monitoring.x:monitoring.x + monitoring.width]
            pixels = int(np.count_nonzero(crop))
            drops = int(item["drop_count"])
            rows.append({
                "filename": item["filename"], "session_id": item["session_id"],
                "drop_count": drops, "class_name": class_name(drops),
                "leak_pixels": pixels, "roi_pixels": monitoring.pixels,
                "leak_ratio": f"{pixels / monitoring.pixels * 100.0:.6f}",
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    grouped: defaultdict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["drop_count"])].append(float(row["leak_ratio"]))
    for drops, values in sorted(grouped.items()):
        LOG.info("drops=%d n=%d min=%.4f mean=%.4f median=%.4f std=%.4f max=%.4f%%",
                 drops, len(values), min(values), float(np.mean(values)),
                 float(np.median(values)), float(np.std(values)), max(values))
    for item in missing:
        LOG.error("Missing mask: %s", item)
    LOG.info("Wrote %d rows to %s", len(rows), output)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

