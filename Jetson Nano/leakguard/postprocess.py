from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostprocessResult:
    mask: np.ndarray
    component_count: int


def postprocess_probability(
    probability: np.ndarray,
    threshold: float,
    min_component_pixels: int,
    kernel_size: int,
    open_iterations: int,
    close_iterations: int,
) -> PostprocessResult:
    if probability.ndim != 2:
        raise ValueError("Probability mask must be a two-dimensional array")
    mask = (probability >= threshold).astype(np.uint8)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        if open_iterations > 0:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations
            )
        if close_iterations > 0:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
            )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    kept = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_pixels:
            cleaned[labels == label] = 1
            kept += 1
    return PostprocessResult(mask=cleaned, component_count=kept)
