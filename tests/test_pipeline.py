from __future__ import annotations

import numpy as np

from src.data.mask_processing import generate_hsv_mask
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

