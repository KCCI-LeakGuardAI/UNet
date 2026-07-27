from __future__ import annotations

import tensorflow as tf


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    name: str,
) -> tf.Tensor:
    value = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_conv1",
    )(inputs)
    value = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(value)
    value = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(value)
    value = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_conv2",
    )(value)
    value = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(value)
    return tf.keras.layers.Activation("relu", name=f"{name}_relu2")(value)


def encoder_block(
    inputs: tf.Tensor,
    filters: int,
    name: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    skip = conv_block(inputs, filters, name)
    pooled = tf.keras.layers.MaxPooling2D((2, 2), name=f"{name}_pool")(skip)
    return skip, pooled


def decoder_block(
    inputs: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    name: str,
) -> tf.Tensor:
    value = tf.keras.layers.Conv2DTranspose(
        filters, 2, strides=2, padding="same", name=f"{name}_up"
    )(inputs)
    value = tf.keras.layers.Concatenate(name=f"{name}_concat")([value, skip])
    return conv_block(value, filters, name)


def build_unet(
    input_height: int = 256,
    input_width: int = 256,
    input_channels: int = 3,
    base_channels: int = 16,
    output_channels: int = 1,
    dropout: float = 0.1,
) -> tf.keras.Model:
    if input_height % 16 or input_width % 16:
        raise ValueError("U-Net input height and width must be divisible by 16")
    inputs = tf.keras.layers.Input(
        (input_height, input_width, input_channels), name="image"
    )
    skip1, pool1 = encoder_block(inputs, base_channels, "encoder1")
    skip2, pool2 = encoder_block(pool1, base_channels * 2, "encoder2")
    skip3, pool3 = encoder_block(pool2, base_channels * 4, "encoder3")
    skip4, pool4 = encoder_block(pool3, base_channels * 8, "encoder4")
    bottleneck = conv_block(pool4, base_channels * 16, "bottleneck")
    bottleneck = tf.keras.layers.Dropout(dropout, name="bottleneck_dropout")(bottleneck)
    value = decoder_block(bottleneck, skip4, base_channels * 8, "decoder4")
    value = decoder_block(value, skip3, base_channels * 4, "decoder3")
    value = decoder_block(value, skip2, base_channels * 2, "decoder2")
    value = decoder_block(value, skip1, base_channels, "decoder1")
    outputs = tf.keras.layers.Conv2D(
        output_channels, 1, activation="sigmoid", name="segmentation_mask"
    )(value)
    return tf.keras.Model(inputs, outputs, name="lightweight_unet")
