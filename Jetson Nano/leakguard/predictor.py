from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from .config import ModelConfig, Rect
from .postprocess import postprocess_probability

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    probability: np.ndarray
    mask: np.ndarray
    inference_ms: float
    component_count: int


def configure_tensorflow_gpu() -> str:
    devices = tf.config.experimental.list_physical_devices("GPU")
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    return "GPU" if devices else "CPU"


class UNetPredictor:
    def __init__(self, config: ModelConfig) -> None:
        if not Path(config.path).is_file():
            raise FileNotFoundError(f"Model not found: {config.path}")
        self.config = config
        self.device = configure_tensorflow_gpu()
        LOG.info("Loading model from %s (%s)", config.path, self.device)
        self.model = tf.keras.models.load_model(str(config.path), compile=False)
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape
        if (
            len(input_shape) != 4
            or input_shape[-1] != 3
            or len(output_shape) != 4
            or output_shape[-1] != 1
        ):
            raise ValueError(
                f"Unexpected model shapes: input={input_shape}, output={output_shape}"
            )
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        warmup = tf.zeros(
            (1, self.input_height, self.input_width, 3), dtype=tf.float32
        )
        self.model(warmup, training=False)
        LOG.info(
            "Model ready: input=%s output=%s",
            input_shape,
            output_shape,
        )

    def _network_input(self, bgr: np.ndarray) -> tf.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
        tensor = tf.image.resize(
            tensor,
            [self.input_height, self.input_width],
            method=tf.image.ResizeMethod.BILINEAR,
        )
        return tensor[None, ...] / 255.0

    def predict(self, frame: np.ndarray, monitoring_roi: Rect) -> Prediction:
        frame_height, frame_width = frame.shape[:2]
        if self.config.crop_to_monitoring_roi:
            network_frame = frame[
                monitoring_roi.y:monitoring_roi.y2,
                monitoring_roi.x:monitoring_roi.x2,
            ]
            if network_frame.size == 0:
                raise ValueError("Monitoring ROI is empty after clipping")
        else:
            network_frame = frame

        started = time.perf_counter()
        output = self.model(self._network_input(network_frame), training=False)
        probability_small = output.numpy()[0, :, :, 0]
        inference_ms = (time.perf_counter() - started) * 1000.0

        probability = np.zeros((frame_height, frame_width), dtype=np.float32)
        if self.config.crop_to_monitoring_roi:
            restored = cv2.resize(
                probability_small,
                (monitoring_roi.width, monitoring_roi.height),
                interpolation=cv2.INTER_LINEAR,
            )
            probability[
                monitoring_roi.y:monitoring_roi.y2,
                monitoring_roi.x:monitoring_roi.x2,
            ] = restored
        else:
            probability = cv2.resize(
                probability_small,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,
            )

        roi_probability = np.zeros_like(probability)
        roi_probability[
            monitoring_roi.y:monitoring_roi.y2,
            monitoring_roi.x:monitoring_roi.x2,
        ] = probability[
            monitoring_roi.y:monitoring_roi.y2,
            monitoring_roi.x:monitoring_roi.x2,
        ]
        result = postprocess_probability(
            probability=roi_probability,
            threshold=self.config.threshold,
            min_component_pixels=self.config.min_component_pixels,
            kernel_size=self.config.morphology_kernel,
            open_iterations=self.config.open_iterations,
            close_iterations=self.config.close_iterations,
        )
        return Prediction(
            probability=probability,
            mask=result.mask,
            inference_ms=inference_ms,
            component_count=result.component_count,
        )
