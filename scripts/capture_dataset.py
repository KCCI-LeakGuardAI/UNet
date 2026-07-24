from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from src.data.roi import Roi
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("capture_dataset")
FIELDNAMES = [
    "filename", "session_id", "drop_count", "capture_time",
    "lighting_condition", "liquid_concentration", "note",
]
KEY_TO_DROPS = {ord("0"): 0, ord("2"): 2, ord("5"): 5, ord("8"): 8}


def append_metadata(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def next_sequence(folder: Path, prefix: str, extension: str) -> int:
    values = []
    for item in folder.glob(f"{prefix}_*{extension}"):
        try:
            values.append(int(item.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=0) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture fixed-camera leak dataset images.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--roi-config", default="configs/roi.yaml")
    parser.add_argument("--session", required=True, help="Session ID, e.g. S01")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--lighting")
    parser.add_argument("--concentration")
    parser.add_argument("--note", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    config, roi_config = load_yaml(args.config), load_yaml(args.roi_config)
    camera_cfg = require(config, "camera")
    width, height = int(camera_cfg["width"]), int(camera_cfg["height"])
    monitoring = Roi.from_mapping(require(roi_config, "monitoring_roi"))
    danger = Roi.from_mapping(require(roi_config, "danger_roi"))
    monitoring.validate(width, height)
    danger.validate(width, height)
    raw_root = Path(require(config, "paths.raw"))
    session_folder = raw_root / f"session_{args.session.removeprefix('S').lower()}"
    session_folder.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(require(config, "paths.metadata"))
    extension = str(camera_cfg.get("image_extension", ".jpg"))
    counters: defaultdict[int, int] = defaultdict(int)
    for drops in (0, 2, 5, 8):
        counters[drops] = next_sequence(session_folder, f"{args.session}_D{drops:02d}", extension)

    camera_index = args.camera if args.camera is not None else int(camera_cfg["index"])
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, int(camera_cfg.get("fps", 30)))
    if not cap.isOpened():
        LOG.error("Camera %s could not be opened", camera_index)
        return 2
    lighting = args.lighting or require(config, "capture.default_lighting_condition")
    concentration = args.concentration or require(config, "capture.default_liquid_concentration")
    current_drops = 0
    LOG.info("Keys: 0/2/5/8 select and capture, S captures current group, Q quits")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                LOG.error("Camera frame read failed")
                return 3
            actual_h, actual_w = frame.shape[:2]
            if (actual_w, actual_h) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            display = frame.copy()
            cv2.rectangle(display, (monitoring.x, monitoring.y),
                          (monitoring.x + monitoring.width, monitoring.y + monitoring.height), (0, 255, 0), 2)
            cv2.rectangle(display, (danger.x, danger.y),
                          (danger.x + danger.width, danger.y + danger.height), (0, 0, 255), 2)
            cv2.putText(display, f"{args.session} | drops={current_drops}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Dataset capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in KEY_TO_DROPS:
                current_drops = KEY_TO_DROPS[key]
            if key in KEY_TO_DROPS or key == ord("s"):
                seq = counters[current_drops]
                filename = f"{args.session}_D{current_drops:02d}_{seq:04d}{extension}"
                output = session_folder / filename
                params = [cv2.IMWRITE_JPEG_QUALITY, int(camera_cfg.get("jpeg_quality", 95))]
                if not cv2.imwrite(str(output), frame, params):
                    LOG.error("Failed to write image: %s", output)
                    continue
                append_metadata(metadata_path, {
                    "filename": filename, "session_id": args.session, "drop_count": current_drops,
                    "capture_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "lighting_condition": lighting, "liquid_concentration": concentration, "note": args.note,
                })
                counters[current_drops] += 1
                LOG.info("Saved %s", output)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

