from __future__ import annotations

import tensorflow as tf


def decode_and_preprocess_image(
    path: tf.Tensor,
    height: int,
    width: int,
) -> tf.Tensor:
    encoded = tf.io.read_file(path)
    image = tf.image.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        tf.cast(image, tf.float32),
        [height, width],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    return image / 255.0


def decode_and_preprocess_mask(
    path: tf.Tensor,
    height: int,
    width: int,
) -> tf.Tensor:
    encoded = tf.io.read_file(path)
    mask = tf.image.decode_png(encoded, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(
        tf.cast(mask, tf.float32),
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return tf.cast(mask >= 127.5, tf.float32)


def load_pair(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    height: int,
    width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        decode_and_preprocess_image(image_path, height, width),
        decode_and_preprocess_mask(mask_path, height, width),
    )
