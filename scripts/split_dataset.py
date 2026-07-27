from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("split_dataset")
SPLITS = ("train", "val", "test")
OUTPUT_FIELDS = [
    "relative_source", "split", "class_name", "pseudo_session",
    "image_path", "mask_path", "leak_ratio",
]


@dataclass(frozen=True)
class Sample:
    relative: Path
    class_name: str
    pseudo_session: str
    leak_ratio: str


def _number_and_family(stem: str) -> tuple[str, int]:
    match = re.search(r"(\d+)\)?$", stem)
    if not match:
        return stem, 0
    family = stem[:match.start()].strip(" ()-_") or "numeric"
    return family, int(match.group(1))


def read_approved_samples(
    review_csv: Path,
    pseudo_session_size: int,
) -> list[Sample]:
    samples: list[Sample] = []
    with review_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["status"].strip().lower() != "approved":
                continue
            relative = Path(row["relative_image"])
            class_name = relative.parts[0]
            family, number = _number_and_family(relative.stem)
            chunk = number // pseudo_session_size
            pseudo_session = f"{class_name}/{family}/chunk_{chunk:03d}"
            samples.append(Sample(
                relative=relative,
                class_name=class_name,
                pseudo_session=pseudo_session,
                leak_ratio=row.get("leak_ratio", ""),
            ))
    return samples


def assign_splits(
    samples: list[Sample],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    by_class_session: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_class_session[sample.class_name][sample.pseudo_session].append(sample)
    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for class_name, session_map in sorted(by_class_session.items()):
        sessions = list(session_map.items())
        rng.shuffle(sessions)
        sessions.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(items) for _, items in sessions)
        targets = {split: ratios[split] * total for split in SPLITS}
        counts = {split: 0 for split in SPLITS}
        for session, items in sessions:
            split = max(
                SPLITS,
                key=lambda name: (
                    targets[name] - counts[name],
                    -counts[name],
                    {"train": 2, "val": 1, "test": 0}[name],
                ),
            )
            assignments[session] = split
            counts[split] += len(items)
        LOG.info("%s split counts: %s", class_name, counts)
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description="Split approved image/mask pairs by pseudo-session.")
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--review", default="dataset/metadata/mask_review.csv")
    parser.add_argument("--images", default="dataset/images_unreviewed")
    parser.add_argument("--masks", default="dataset/masks_unreviewed")
    parser.add_argument("--output", default="dataset/prepared")
    parser.add_argument("--manifest", default="dataset/metadata/split_result.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_yaml(args.config)
    split_config = require(config, "split")
    ratios = {name: float(split_config[name]) for name in SPLITS}
    if abs(sum(ratios.values()) - 1.0) > 1e-6 or any(value <= 0 for value in ratios.values()):
        LOG.error("Split ratios must be positive and sum to 1")
        return 2
    pseudo_session_size = int(split_config["pseudo_session_size"])
    if pseudo_session_size < 1:
        LOG.error("pseudo_session_size must be positive")
        return 2
    review_path, image_root, mask_root = Path(args.review), Path(args.images), Path(args.masks)
    if not review_path.is_file() or not image_root.is_dir() or not mask_root.is_dir():
        LOG.error("Review CSV, image root, or mask root is missing")
        return 2
    samples = read_approved_samples(review_path, pseudo_session_size)
    if not samples:
        LOG.error("No approved samples found in %s", review_path)
        return 3
    assignments = assign_splits(samples, ratios, int(split_config["seed"]))
    output_root = Path(args.output)
    rows: list[dict[str, str]] = []
    failures = 0
    for sample in samples:
        source_image = image_root / sample.relative
        source_mask = mask_root / sample.relative.with_suffix(".png")
        split = assignments[sample.pseudo_session]
        relative_image = Path(split) / sample.relative
        relative_mask = Path(split) / sample.relative.with_suffix(".png")
        destination_image = output_root / "images" / relative_image
        destination_mask = output_root / "masks" / relative_mask
        if not source_image.is_file() or not source_mask.is_file():
            LOG.error("Missing approved pair: %s", sample.relative.as_posix())
            failures += 1
            continue
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.ndim != 2 or image.shape[:2] != mask.shape:
            LOG.error("Invalid approved pair: %s", sample.relative.as_posix())
            failures += 1
            continue
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)
        shutil.copy2(source_mask, destination_mask)
        rows.append({
            "relative_source": sample.relative.as_posix(),
            "split": split,
            "class_name": sample.class_name,
            "pseudo_session": sample.pseudo_session,
            "image_path": (Path("images") / relative_image).as_posix(),
            "mask_path": (Path("masks") / relative_mask).as_posix(),
            "leak_ratio": sample.leak_ratio,
        })
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    LOG.info("Approved=%d copied=%d failures=%d counts=%s", len(samples), len(rows), failures, split_counts)
    LOG.info("Split manifest written to %s", manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
