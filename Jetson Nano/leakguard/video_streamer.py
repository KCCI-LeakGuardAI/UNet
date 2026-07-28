from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np

from .config import VideoStreamConfig

LOG = logging.getLogger("leakguard.video")

VIDEO_MAGIC = b"LGIM"
VIDEO_VERSION = 1
VIDEO_FLAG_ANNOTATED = 1
VIDEO_HEADER = struct.Struct("!4sBBHIQI")
VIDEO_HEADER_SIZE = VIDEO_HEADER.size


def encode_video_packet(
    frame: np.ndarray,
    sequence: int,
    timestamp_ms: int,
    config: VideoStreamConfig,
) -> bytes:
    """Resize and JPEG-encode one annotated frame with a fixed 24-byte header."""
    if frame is None or frame.size == 0:
        raise ValueError("Video frame must not be empty")

    height, width = frame.shape[:2]
    if width != config.width or height != config.height:
        frame = cv2.resize(
            frame,
            (config.width, config.height),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode video frame as JPEG")
    jpeg = encoded.tobytes()
    if len(jpeg) > config.max_frame_bytes:
        raise ValueError(
            f"Encoded JPEG is too large: {len(jpeg)} > "
            f"{config.max_frame_bytes} bytes"
        )

    header = VIDEO_HEADER.pack(
        VIDEO_MAGIC,
        VIDEO_VERSION,
        VIDEO_FLAG_ANNOTATED,
        VIDEO_HEADER_SIZE,
        sequence & 0xFFFFFFFF,
        timestamp_ms,
        len(jpeg),
    )
    return header + jpeg


class LatestFrameTcpStreamer:
    """Send the latest annotated frame without blocking the inference loop."""

    def __init__(self, config: VideoStreamConfig) -> None:
        self.config = config
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp_ms = 0
        self._latest_update = 0.0
        self._generation = 0
        self._next_accept = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"leakguard-video-{self.config.port}",
            daemon=True,
        )
        self._thread.start()

    def update(self, frame: np.ndarray) -> bool:
        """Offer a frame; return True only when accepted at the configured FPS."""
        now = time.monotonic()
        with self._lock:
            if now < self._next_accept:
                return False
            self._next_accept = now + 1.0 / self.config.fps
            self._latest_frame = frame.copy()
            self._latest_timestamp_ms = int(time.time() * 1000.0)
            self._latest_update = now
            self._generation += 1
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._close_socket()
        if self._thread:
            self._thread.join(
                timeout=self.config.connect_timeout_seconds + 1.0
            )
        self._thread = None

    def _run(self) -> None:
        next_connect = 0.0
        last_sent_generation = -1
        sequence = 0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._socket is None:
                if now < next_connect:
                    self._stop_event.wait(min(next_connect - now, 0.1))
                    continue
                if not self._connect():
                    next_connect = (
                        time.monotonic()
                        + self.config.reconnect_interval_seconds
                    )
                    continue
                last_sent_generation = -1

            current = self._current_frame(last_sent_generation)
            if current is None:
                self._stop_event.wait(0.01)
                continue
            frame, timestamp_ms, generation = current
            try:
                packet = encode_video_packet(
                    frame,
                    sequence=sequence,
                    timestamp_ms=timestamp_ms,
                    config=self.config,
                )
                assert self._socket is not None
                self._socket.sendall(packet)
                last_sent_generation = generation
                sequence = (sequence + 1) & 0xFFFFFFFF
            except (OSError, RuntimeError, ValueError) as exc:
                LOG.warning("Video stream send failed (%s); reconnecting", exc)
                self._close_socket()
                next_connect = (
                    time.monotonic()
                    + self.config.reconnect_interval_seconds
                )

        self._close_socket()

    def _current_frame(
        self, last_sent_generation: int
    ) -> Optional[tuple[np.ndarray, int, int]]:
        with self._lock:
            if self._latest_frame is None:
                return None
            if self._generation == last_sent_generation:
                return None
            if (
                time.monotonic() - self._latest_update
                > self.config.frame_stale_seconds
            ):
                return None
            return (
                self._latest_frame,
                self._latest_timestamp_ms,
                self._generation,
            )

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.connect_timeout_seconds,
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self._socket = sock
            LOG.info(
                "Connected video stream to %s:%d",
                self.config.host,
                self.config.port,
            )
            return True
        except OSError as exc:
            LOG.warning(
                "Could not connect video stream to %s:%d (%s); retrying",
                self.config.host,
                self.config.port,
                exc,
            )
            return False

    def _close_socket(self) -> None:
        sock, self._socket = self._socket, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def with_video_overrides(
    config: VideoStreamConfig,
    host: Optional[str],
    port: Optional[int],
    disabled: bool,
) -> VideoStreamConfig:
    return replace(
        config,
        enabled=False if disabled else config.enabled,
        host=host or config.host,
        port=port if port is not None else config.port,
    )
