from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analyzer import LeakStatus
from .config import Rect
from .predictor import Prediction

LEVEL_COLORS = {
    "NORMAL": (60, 190, 60),
    "SMALL_LEAK": (0, 220, 255),
    "WARNING": (0, 140, 255),
    "DANGER": (30, 30, 235),
}


@dataclass(frozen=True)
class Telemetry:
    fps: float
    end_to_end_ms: float
    frame_index: int
    device: str


def render_dashboard(
    frame: np.ndarray,
    prediction: Prediction,
    status: LeakStatus,
    monitoring_roi: Rect,
    danger_roi: Rect,
    telemetry: Telemetry,
    overlay_alpha: float,
    threshold: float,
    show_mask_preview: bool,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    display_mask = prediction.mask
    if source_width < 640:
        display_width = 640
        display_height = int(round(source_height * display_width / source_width))
        frame = cv2.resize(
            frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR
        )
        display_mask = cv2.resize(
            prediction.mask,
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        monitoring_roi = monitoring_roi.scale(
            source_width, source_height, display_width, display_height
        )
        danger_roi = danger_roi.scale(
            source_width, source_height, display_width, display_height
        )
    canvas = frame.copy()
    level_color = LEVEL_COLORS[status.level]
    selected = display_mask.astype(bool)
    if np.any(selected):
        color_layer = np.zeros_like(canvas)
        color_layer[selected] = level_color
        blended = cv2.addWeighted(
            canvas, 1.0 - overlay_alpha, color_layer, overlay_alpha, 0.0
        )
        canvas[selected] = blended[selected]
        contours, _ = cv2.findContours(
            display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, level_color, 2)

    cv2.rectangle(
        canvas,
        (monitoring_roi.x, monitoring_roi.y),
        (monitoring_roi.x2, monitoring_roi.y2),
        (80, 220, 80),
        2,
    )
    danger_color = (0, 0, 255) if status.danger_overlap else (255, 0, 180)
    cv2.rectangle(
        canvas,
        (danger_roi.x, danger_roi.y),
        (danger_roi.x2, danger_roi.y2),
        danger_color,
        2,
    )
    cv2.putText(
        canvas,
        "MONITORING ROI",
        (monitoring_roi.x + 4, max(18, monitoring_roi.y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 220, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "DANGER ROI",
        (danger_roi.x + 3, max(18, danger_roi.y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        danger_color,
        1,
        cv2.LINE_AA,
    )

    panel_height = min(145, canvas.shape[0])
    panel = canvas[:panel_height].copy()
    dark = np.zeros_like(panel)
    canvas[:panel_height] = cv2.addWeighted(panel, 0.25, dark, 0.75, 0)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 32), level_color, -1)
    cv2.putText(
        canvas,
        f"LEAKGUARD AI   LEVEL: {status.level}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    left = [
        f"Leak area: {status.leak_ratio:.3f}%  (instant {status.instant_ratio:.3f}%)",
        f"Detected: {_yes_no(status.detected)}   Pixels: {status.leak_pixels:,}",
        (
            f"Spreading: {_yes_no(status.spreading)}  "
            f"Delta: {status.spreading_delta:+.3f}%"
        ),
    ]
    right = [
        (
            f"Danger ROI: {_yes_no(status.danger_overlap)}  "
            f"Overlap: {status.danger_overlap_pixels}px"
        ),
        (
            f"Infer: {prediction.inference_ms:.1f}ms  "
            f"E2E: {telemetry.end_to_end_ms:.1f}ms  FPS: {telemetry.fps:.1f}"
        ),
        (
            f"Components: {prediction.component_count}  Threshold: {threshold:.2f}  "
            f"{telemetry.device}"
        ),
    ]
    for index, text in enumerate(left):
        _text(canvas, text, 10, 56 + index * 27)
    right_x = max(10, canvas.shape[1] // 2)
    for index, text in enumerate(right):
        _text(canvas, text, right_x, 56 + index * 27)

    if show_mask_preview:
        _add_mask_preview(canvas, display_mask)
    cv2.putText(
        canvas,
        "Q/ESC: quit   S: snapshot   R: reset analyzer",
        (8, canvas.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _add_mask_preview(canvas: np.ndarray, mask: np.ndarray) -> None:
    preview_width = min(160, max(80, canvas.shape[1] // 5))
    preview_height = int(preview_width * mask.shape[0] / mask.shape[1])
    preview = cv2.resize(
        mask * 255,
        (preview_width, preview_height),
        interpolation=cv2.INTER_NEAREST,
    )
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    x1 = canvas.shape[1] - preview_width - 8
    y1 = canvas.shape[0] - preview_height - 28
    if x1 < 0 or y1 < 145:
        return
    canvas[y1:y1 + preview_height, x1:x1 + preview_width] = preview_bgr
    cv2.rectangle(
        canvas,
        (x1 - 1, y1 - 1),
        (x1 + preview_width, y1 + preview_height),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        canvas,
        "BINARY MASK",
        (x1, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _text(image: np.ndarray, value: str, x: int, y: int) -> None:
    cv2.putText(
        image,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"
