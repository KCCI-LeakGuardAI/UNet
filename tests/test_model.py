from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.models.losses import bce_dice_loss, dice_coefficient
from src.models.metrics import dice_score, iou_score
from src.models.unet import build_unet


def test_unet_output_shape() -> None:
    model = build_unet(
        input_height=64, input_width=64, input_channels=3,
        base_channels=4, output_channels=1, dropout=0.1,
    )
    output = model(tf.zeros((2, 64, 64, 3)), training=False)
    assert tuple(output.shape) == (2, 64, 64, 1)
    assert np.isfinite(output.numpy()).all()


def test_dice_is_one_for_perfect_prediction() -> None:
    true_mask = tf.constant([[[[0.0], [1.0]], [[1.0], [0.0]]]])
    score = dice_coefficient(true_mask, true_mask)
    assert float(score.numpy()) == 1.0
    assert float(bce_dice_loss(true_mask, true_mask).numpy()) < 1e-5


def test_binary_metrics_handle_empty_masks() -> None:
    empty = tf.zeros((1, 4, 4, 1), dtype=tf.float32)
    assert float(dice_score(empty, empty).numpy()) == 1.0
    assert float(iou_score(empty, empty).numpy()) == 1.0

    one_false_positive = tf.tensor_scatter_nd_update(
        empty, indices=[[0, 0, 0, 0]], updates=[1.0]
    )
    assert float(dice_score(empty, one_false_positive).numpy()) == 0.0
    assert float(iou_score(empty, one_false_positive).numpy()) == 0.0
