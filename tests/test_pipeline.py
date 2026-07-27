from __future__ import annotations

import numpy as np

from src.data.augmentation import augment_pair
from src.data.mask_processing import fill_mask_holes, generate_hsv_mask
from src.data.roi import Roi


def test_hsv_mask_removes_small_components() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:50, 20:50] = (255, 0, 0)
    image[2:4, 2:4] = (255, 0, 0)
    mask = generate_hsv_mask(image, [90, 80, 50], [135, 255, 255], 3, 0, 0, 100)
    assert mask[30, 30] == 255
    assert mask[2, 2] == 0
    assert set(np.unique(mask)).issubset({0, 255})


def test_roi_validation() -> None:
    roi = Roi(10, 20, 30, 40)
    roi.validate(100, 100)
    assert roi.pixels == 1200


def test_fill_mask_holes_preserves_outer_background() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[8:12, 8:12] = 0
    filled = fill_mask_holes(mask)
    assert filled[9, 9] == 255
    assert filled[0, 0] == 0


def test_augmentation_preserves_binary_mask_and_shape() -> None:
    image = np.full((64, 64, 3), 220, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    image[20:40, 20:40] = (30, 80, 140)
    mask[20:40, 20:40] = 255
    config = {
        "rotation_degrees": 5,
        "translation_fraction": 0.03,
        "scale_min": 0.95,
        "scale_max": 1.05,
        "brightness_factor_min": 0.9,
        "brightness_factor_max": 1.1,
        "contrast_factor_min": 0.95,
        "contrast_factor_max": 1.05,
        "saturation_factor_min": 0.95,
        "saturation_factor_max": 1.05,
        "hue_shift_max": 2,
        "blur_probability": 0.0,
        "noise_probability": 0.0,
        "noise_std_max": 3.0,
        "minimum_foreground_retention": 0.9,
    }
    augmented_image, augmented_mask = augment_pair(
        image, mask, np.random.default_rng(42), config
    )
    assert augmented_image.shape == image.shape
    assert augmented_mask.shape == mask.shape
    assert set(np.unique(augmented_mask)).issubset({0, 255})
    assert np.count_nonzero(augmented_mask) >= np.count_nonzero(mask) * 0.9
