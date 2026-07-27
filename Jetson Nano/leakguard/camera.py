from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import CameraConfig

LOG = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class FrameSource:
    def __init__(
        self,
        config: CameraConfig,
        camera_index: Optional[int] = None,
        media_path: Optional[str] = None,
        gstreamer_pipeline: Optional[str] = None,
    ) -> None:
        self.config = config
        self.camera_index = config.index if camera_index is None else camera_index
        self.media_path = Path(media_path).expanduser() if media_path else None
        self.gstreamer_pipeline = gstreamer_pipeline
        self.capture: Optional[cv2.VideoCapture] = None
        self._still_image: Optional[np.ndarray] = None
        self._still_delivered = False

    @property
    def description(self) -> str:
        if self._still_image is not None or self.media_path is not None:
            return str(self.media_path)
        if self.gstreamer_pipeline:
            return "GStreamer pipeline"
        return f"camera {self.camera_index}"

    def open(self) -> None:
        if self.media_path and self.media_path.suffix.lower() in IMAGE_SUFFIXES:
            self._still_image = cv2.imread(str(self.media_path), cv2.IMREAD_COLOR)
            if self._still_image is None:
                raise RuntimeError(f"Could not read image: {self.media_path}")
            return
        if self.media_path:
            self.capture = cv2.VideoCapture(str(self.media_path))
        elif self.gstreamer_pipeline:
            self.capture = cv2.VideoCapture(
                self.gstreamer_pipeline, cv2.CAP_GSTREAMER
            )
        else:
            backend = self._backend()
            self.capture = cv2.VideoCapture(self.camera_index, backend)
            if not self.capture.isOpened() and backend != cv2.CAP_ANY:
                LOG.warning("Requested backend failed; retrying with CAP_ANY")
                self.capture.release()
                self.capture = cv2.VideoCapture(self.camera_index)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            if len(self.config.fourcc) == 4:
                self.capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.config.fourcc),
                )
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"Could not open {self.description}")
        LOG.info(
            "Opened %s: %dx%d @ %.1f FPS",
            self.description,
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.capture.get(cv2.CAP_PROP_FPS),
        )

    def _backend(self) -> int:
        if self.config.backend == "v4l2":
            return cv2.CAP_V4L2
        if self.config.backend == "gstreamer":
            return cv2.CAP_GSTREAMER
        if self.config.backend == "dshow":
            return cv2.CAP_DSHOW
        if platform.system() == "Linux":
            return cv2.CAP_V4L2
        if platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._still_image is not None:
            if self._still_delivered:
                return False, None
            self._still_delivered = True
            return True, self._still_image.copy()
        if self.capture is None:
            return False, None
        return self.capture.read()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
