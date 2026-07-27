from __future__ import annotations

import tensorflow as tf

from src.models.losses import dice_coefficient


def dice_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Return thresholded Dice, including a correct score for empty masks.

    The loss keeps using the differentiable soft Dice coefficient.  Model
    selection, however, must reflect the binary mask that is actually used at
    inference time.  Thresholding also makes an empty ground-truth/prediction
    pair score 1 instead of being penalized merely because sigmoid outputs are
    not mathematically equal to zero.
    """
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    score = tf.math.divide_no_nan(2.0 * intersection, denominator)
    both_empty = tf.equal(denominator, 0.0)
    score = tf.where(both_empty, tf.ones_like(score), score)
    return tf.reduce_mean(score)


def iou_score(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true + y_pred, axis=axes) - intersection
    score = tf.math.divide_no_nan(intersection, union)
    both_empty = tf.equal(union, 0.0)
    score = tf.where(both_empty, tf.ones_like(score), score)
    return tf.reduce_mean(score)


def training_metrics() -> list[tf.keras.metrics.Metric | object]:
    return [
        dice_score,
        iou_score,
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
    ]


def custom_objects() -> dict[str, object]:
    from src.models.losses import bce_dice_loss, dice_loss

    return {
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "dice_score": dice_score,
        "iou_score": iou_score,
    }
