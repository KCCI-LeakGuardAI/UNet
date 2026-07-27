from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from .config import AnalysisConfig, Rect

LEVELS = ("NORMAL", "SMALL_LEAK", "WARNING", "DANGER")


@dataclass(frozen=True)
class LeakStatus:
    detected: bool
    leak_ratio: float
    instant_ratio: float
    spreading: bool
    spreading_delta: float
    danger_overlap: bool
    leak_pixels: int
    monitoring_pixels: int
    danger_overlap_pixels: int
    level: str
    timestamp: float


class LeakAnalyzer:
    def __init__(
        self,
        config: AnalysisConfig,
        danger_minimum_overlap_pixels: int,
    ) -> None:
        self.config = config
        self.danger_minimum_overlap_pixels = danger_minimum_overlap_pixels
        self._ema_ratio: float | None = None
        self._detection_votes: Deque[bool] = deque(
            maxlen=config.detection_window_frames
        )
        self._ratio_history: Deque[float] = deque(
            maxlen=config.spreading_window_frames * 2
        )
        self._spreading_streak = 0

    def reset(self) -> None:
        self._ema_ratio = None
        self._detection_votes.clear()
        self._ratio_history.clear()
        self._spreading_streak = 0

    def analyze(
        self,
        mask: np.ndarray,
        monitoring_roi: Rect,
        danger_roi: Rect,
    ) -> LeakStatus:
        monitor = mask[
            monitoring_roi.y:monitoring_roi.y2,
            monitoring_roi.x:monitoring_roi.x2,
        ]
        monitoring_pixels = max(1, monitoring_roi.area)
        leak_pixels = int(np.count_nonzero(monitor))
        instant_ratio = leak_pixels / monitoring_pixels * 100.0
        if self._ema_ratio is None:
            self._ema_ratio = instant_ratio
        else:
            alpha = self.config.ema_alpha
            self._ema_ratio = alpha * instant_ratio + (1.0 - alpha) * self._ema_ratio

        self._detection_votes.append(
            instant_ratio >= self.config.leak_detect_ratio
        )
        detected = (
            sum(self._detection_votes)
            >= self.config.detection_required_frames
        )

        danger_mask = mask[
            danger_roi.y:danger_roi.y2,
            danger_roi.x:danger_roi.x2,
        ]
        danger_overlap_pixels = int(np.count_nonzero(danger_mask))
        danger_overlap = (
            danger_overlap_pixels >= self.danger_minimum_overlap_pixels
        )

        self._ratio_history.append(self._ema_ratio)
        spreading_delta = 0.0
        spreading_candidate = False
        window = self.config.spreading_window_frames
        if len(self._ratio_history) == window * 2:
            values = list(self._ratio_history)
            previous = float(np.mean(values[:window]))
            current = float(np.mean(values[window:]))
            spreading_delta = current - previous
            spreading_candidate = (
                spreading_delta >= self.config.spreading_delta_ratio
            )
        self._spreading_streak = (
            self._spreading_streak + 1 if spreading_candidate else 0
        )
        spreading = (
            detected
            and self._spreading_streak >= self.config.spreading_confirm_frames
        )

        level = self._level(
            detected=detected,
            ratio=self._ema_ratio,
            spreading=spreading,
            danger_overlap=danger_overlap,
        )
        return LeakStatus(
            detected=detected,
            leak_ratio=self._ema_ratio,
            instant_ratio=instant_ratio,
            spreading=spreading,
            spreading_delta=spreading_delta,
            danger_overlap=danger_overlap,
            leak_pixels=leak_pixels,
            monitoring_pixels=monitoring_pixels,
            danger_overlap_pixels=danger_overlap_pixels,
            level=level,
            timestamp=time.time(),
        )

    def _level(
        self,
        detected: bool,
        ratio: float,
        spreading: bool,
        danger_overlap: bool,
    ) -> str:
        if danger_overlap or ratio >= self.config.danger_ratio:
            return "DANGER"
        if not detected:
            return "NORMAL"
        if ratio >= self.config.warning_ratio:
            level = "WARNING"
        elif ratio >= self.config.leak_detect_ratio:
            level = "SMALL_LEAK"
        else:
            level = "NORMAL"
        if spreading and self.config.spreading_escalates:
            if level == "SMALL_LEAK":
                return "WARNING"
            if level == "WARNING":
                return "DANGER"
        return level
