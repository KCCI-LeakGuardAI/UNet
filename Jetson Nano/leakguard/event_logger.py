from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

from .analyzer import LeakStatus
from .predictor import Prediction
from .visualization import Telemetry

FIELDS = [
    "timestamp",
    "event",
    "level",
    "detected",
    "leak_ratio",
    "instant_ratio",
    "leak_pixels",
    "spreading",
    "spreading_delta",
    "danger_overlap",
    "danger_overlap_pixels",
    "components",
    "inference_ms",
    "fps",
    "frame_index",
]


class StatusCsvLogger:
    def __init__(self, path: Path, interval_seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.is_file() and path.stat().st_size > 0
        self._stream: TextIO = path.open(
            "a", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(self._stream, fieldnames=FIELDS)
        if not existed:
            self._writer.writeheader()
        self._interval = interval_seconds
        self._last_write = 0.0
        self._last_level: Optional[str] = None

    def write(
        self,
        status: LeakStatus,
        prediction: Prediction,
        telemetry: Telemetry,
    ) -> None:
        level_changed = (
            self._last_level is not None
            and self._last_level != status.level
        )
        if (
            not level_changed
            and status.timestamp - self._last_write < self._interval
        ):
            return
        event = (
            f"LEVEL_CHANGE:{self._last_level}->{status.level}"
            if level_changed
            else "STATUS"
        )
        self._writer.writerow(
            {
                "timestamp": datetime.fromtimestamp(
                    status.timestamp
                ).isoformat(timespec="milliseconds"),
                "event": event,
                "level": status.level,
                "detected": int(status.detected),
                "leak_ratio": f"{status.leak_ratio:.6f}",
                "instant_ratio": f"{status.instant_ratio:.6f}",
                "leak_pixels": status.leak_pixels,
                "spreading": int(status.spreading),
                "spreading_delta": f"{status.spreading_delta:.6f}",
                "danger_overlap": int(status.danger_overlap),
                "danger_overlap_pixels": status.danger_overlap_pixels,
                "components": prediction.component_count,
                "inference_ms": f"{prediction.inference_ms:.3f}",
                "fps": f"{telemetry.fps:.3f}",
                "frame_index": telemetry.frame_index,
            }
        )
        self._stream.flush()
        self._last_write = status.timestamp
        self._last_level = status.level

    def close(self) -> None:
        self._stream.close()
