from __future__ import annotations

import cv2
import numpy as np


def validate_hsv_bounds(lower: list[int], upper: list[int]) -> None:
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("HSV lower and upper bounds must contain three values")
    limits = (179, 255, 255)
    for name, values in (("lower", lower), ("upper", upper)):
        if any(not isinstance(v, int) or v < 0 or v > limits[i] for i, v in enumerate(values)):
            raise ValueError(f"Invalid HSV {name} bounds: {values}")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("HSV lower bounds cannot exceed upper bounds")


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels:
            output[labels == label] = 255
    return output


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions fully enclosed by foreground liquid pixels."""
    binary = (mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    enclosed = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, enclosed)


def generate_hsv_mask(
    image_bgr: np.ndarray,
    lower: list[int],
    upper: list[int],
    kernel_size: int,
    open_iterations: int,
    close_iterations: int,
    min_component_pixels: int,
    fill_holes: bool = True,
) -> np.ndarray:
    validate_hsv_bounds(lower, upper)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Morphology kernel_size must be a positive odd integer")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if open_iterations:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    if close_iterations:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask = remove_small_components(mask, min_component_pixels)
    return fill_mask_holes(mask) if fill_holes else mask


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :, 2] = mask
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image_bgr, 0.55, colored, 0.45, 0)[selected]
    return overlay
