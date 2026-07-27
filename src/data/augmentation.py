from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


def _uniform(rng: np.random.Generator, config: Mapping[str, Any], low: str, high: str) -> float:
    return float(rng.uniform(float(config[low]), float(config[high])))


def _geometric_transform(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    angle = float(rng.uniform(-float(config["rotation_degrees"]), float(config["rotation_degrees"])))
    scale = _uniform(rng, config, "scale_min", "scale_max")
    tx = float(rng.uniform(-1, 1) * float(config["translation_fraction"]) * width)
    ty = float(rng.uniform(-1, 1) * float(config["translation_fraction"]) * height)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed_image, np.where(transformed_mask > 127, 255, 0).astype(np.uint8)


def _photometric_transform(
    image: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> np.ndarray:
    value = image.astype(np.float32)
    contrast = _uniform(rng, config, "contrast_factor_min", "contrast_factor_max")
    brightness = _uniform(rng, config, "brightness_factor_min", "brightness_factor_max")
    channel_mean = value.mean(axis=(0, 1), keepdims=True)
    value = (value - channel_mean) * contrast + channel_mean
    value *= brightness
    value = np.clip(value, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(value, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = _uniform(rng, config, "saturation_factor_min", "saturation_factor_max")
    hue_max = int(config["hue_shift_max"])
    hue_shift = int(rng.integers(-hue_max, hue_max + 1)) if hue_max else 0
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue_shift, 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    value = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < float(config["blur_probability"]):
        value = cv2.GaussianBlur(value, (3, 3), 0)
    if rng.random() < float(config["noise_probability"]):
        noise_std = float(rng.uniform(0.5, float(config["noise_std_max"])))
        noise = rng.normal(0.0, noise_std, value.shape)
        value = np.clip(value.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return value


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply matched geometry to image/mask and image-only photometric changes."""
    if image.ndim != 3 or mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError("Image and single-channel mask dimensions must match")
    original_pixels = int(np.count_nonzero(mask))
    attempts = 8
    transformed_image, transformed_mask = image, mask
    for _ in range(attempts):
        transformed_image, transformed_mask = _geometric_transform(image, mask, rng, config)
        if original_pixels == 0:
            break
        retention = np.count_nonzero(transformed_mask) / original_pixels
        if retention >= float(config["minimum_foreground_retention"]):
            break
    transformed_image = _photometric_transform(transformed_image, rng, config)
    return transformed_image, transformed_mask
