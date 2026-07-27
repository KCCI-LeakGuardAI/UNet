from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.utils.logger import configure_logging

LOG = logging.getLogger("review_masks")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
REPORT_FIELDS = [
    "relative_image", "relative_mask", "status", "review_note",
    "leak_pixels", "leak_ratio",
]


@dataclass
class Pair:
    relative_image: Path
    image_path: Path
    mask_path: Path
    overlay_path: Path
    leak_pixels: int
    leak_ratio: float
    issues: list[str]


def collect_pairs(
    image_root: Path,
    mask_root: Path,
    overlay_root: Path,
    empty_allowed_tokens: tuple[str, ...],
) -> tuple[list[Pair], list[str]]:
    pairs: list[Pair] = []
    global_errors: list[str] = []
    images = sorted(
        p for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in images:
        relative = image_path.relative_to(image_root)
        relative_png = relative.with_suffix(".png")
        mask_path = mask_root / relative_png
        overlay_path = overlay_root / relative_png
        issues: list[str] = []
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            global_errors.append(f"unreadable image: {relative.as_posix()}")
            continue
        if mask is None:
            global_errors.append(f"missing/unreadable mask: {relative_png.as_posix()}")
            continue
        if mask.ndim != 2:
            issues.append("mask_not_single_channel")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != mask.shape:
            issues.append("size_mismatch")
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            issues.append("mask_not_binary")
        leak_pixels = int(np.count_nonzero(mask))
        leak_ratio = leak_pixels / mask.size * 100.0
        group = relative.parts[0].lower() if len(relative.parts) > 1 else ""
        empty_allowed = any(token.lower() in group for token in empty_allowed_tokens)
        if leak_pixels == 0 and not empty_allowed:
            issues.append("unexpected_empty_mask")
        pairs.append(Pair(
            relative, image_path, mask_path, overlay_path,
            leak_pixels, leak_ratio, issues,
        ))
    return pairs, global_errors


def create_contact_sheets(pairs: list[Pair], output_dir: Path, columns: int = 4) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Pair]] = {}
    for pair in pairs:
        group = pair.relative_image.parts[0] if len(pair.relative_image.parts) > 1 else "root"
        grouped.setdefault(group, []).append(pair)
    for group, items in grouped.items():
        selected_indices = np.linspace(0, len(items) - 1, min(16, len(items)), dtype=int)
        tiles: list[np.ndarray] = []
        for index in selected_indices:
            pair = items[int(index)]
            image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
            overlay = cv2.imread(str(pair.overlay_path), cv2.IMREAD_COLOR)
            if overlay is None:
                overlay = image.copy()
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            mask_bgr = cv2.cvtColor(
                cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR,
            )
            overlay = cv2.resize(overlay, (224, 224), interpolation=cv2.INTER_AREA)
            tile = np.hstack((image, mask_bgr, overlay))
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 25), (0, 0, 0), -1)
            label = f"{pair.relative_image.name} area={pair.leak_ratio:.3f}%"
            cv2.putText(
                tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
            tiles.append(tile)
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
        cv2.imwrite(str(output_dir / f"{group}.jpg"), np.vstack(rows))


def write_report(path: Path, pairs: list[Pair], decisions: dict[str, tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for pair in pairs:
            key = pair.relative_image.as_posix()
            default_status = "needs_review" if pair.issues else "auto_checked"
            status, note = decisions.get(key, (default_status, ";".join(pair.issues)))
            writer.writerow({
                "relative_image": key,
                "relative_mask": pair.relative_image.with_suffix(".png").as_posix(),
                "status": status,
                "review_note": note,
                "leak_pixels": pair.leak_pixels,
                "leak_ratio": f"{pair.leak_ratio:.6f}",
            })


def interactive_review(pairs: list[Pair]) -> dict[str, tuple[str, str]]:
    decisions: dict[str, tuple[str, str]] = {}
    index = 0
    LOG.info("Keys: Enter/A approve, R reject, B previous, Q quit")
    while 0 <= index < len(pairs):
        pair = pairs[index]
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        overlay = cv2.imread(str(pair.overlay_path), cv2.IMREAD_COLOR)
        if overlay is None:
            overlay = image.copy()
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        panel = np.hstack((image, mask_bgr, overlay))
        label = (
            f"{index + 1}/{len(pairs)} {pair.relative_image.as_posix()} "
            f"area={pair.leak_ratio:.3f}%"
        )
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(panel, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Original | Binary mask | Overlay", panel)
        key = cv2.waitKey(0) & 0xFF
        item_key = pair.relative_image.as_posix()
        if key in (13, ord("a")):
            decisions[item_key] = ("approved", "")
            index += 1
        elif key == ord("r"):
            decisions[item_key] = ("rejected", "manual rejection")
            index += 1
        elif key == ord("b"):
            index = max(0, index - 1)
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and visually review HSV mask candidates.")
    parser.add_argument("--images", default="dataset/images_unreviewed")
    parser.add_argument("--masks", default="dataset/masks_unreviewed")
    parser.add_argument("--overlays", default="dataset/overlays")
    parser.add_argument("--report", default="dataset/metadata/mask_review.csv")
    parser.add_argument("--contact-sheets", default="artifacts/mask_review")
    parser.add_argument("--empty-allowed", nargs="*", default=["background", "spreadROI"])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    roots = [Path(args.images), Path(args.masks), Path(args.overlays)]
    if not roots[0].is_dir() or not roots[1].is_dir():
        LOG.error("Image or mask directory does not exist")
        return 2
    pairs, errors = collect_pairs(
        roots[0], roots[1], roots[2], tuple(args.empty_allowed)
    )
    for error in errors:
        LOG.error(error)
    decisions = interactive_review(pairs) if args.interactive else {}
    write_report(Path(args.report), pairs, decisions)
    create_contact_sheets(pairs, Path(args.contact_sheets))
    issue_count = sum(bool(pair.issues) for pair in pairs)
    LOG.info(
        "pairs=%d global_errors=%d pairs_needing_review=%d",
        len(pairs), len(errors), issue_count,
    )
    LOG.info("Review report written to %s", args.report)
    LOG.info("Contact sheets written to %s", args.contact_sheets)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
