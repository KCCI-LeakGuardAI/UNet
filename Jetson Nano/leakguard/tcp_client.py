from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import replace
from typing import Dict, Optional

from .analyzer import LeakStatus
from .config import NetworkConfig

LOG = logging.getLogger("leakguard.tcp")

# Raspberry Pi/STM32 protocol state values.
LEVEL_TO_STATE: Dict[str, int] = {
    "NORMAL": 0,
    "SMALL_LEAK": 1,
    "WARNING": 2,
    "DANGER": 3,
}
MAX_AREA_VALUE = 65535


def encode_status_packet(status: LeakStatus) -> bytes:
    """Encode one Raspberry Pi status line: state area spread zone\n."""
    try:
        state = LEVEL_TO_STATE[status.level]
    except KeyError as exc:
        raise ValueError(f"Unsupported leak level: {status.level}") from exc

    # area is percent x 100 (for example, 5.00% -> 500). The Raspberry Pi
    # forwards this field as an unsigned 16-bit value to the STM32.
    area = max(0, min(MAX_AREA_VALUE, int(round(status.leak_ratio * 100.0))))
    spread = int(bool(status.spreading))
    zone = int(bool(status.danger_overlap))
    return f"{state} {area} {spread} {zone}\n".encode("ascii")


class LatestStatusTcpClient:
    """Send the latest inference status periodically without blocking inference."""

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self._latest_packet: Optional[bytes] = None
        self._latest_update = 0.0
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
            name="leakguard-tcp",
            daemon=True,
        )
        self._thread.start()

    def update(self, status: LeakStatus) -> None:
        packet = encode_status_packet(status)
        with self._lock:
            self._latest_packet = packet
            self._latest_update = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_socket()
        if self._thread:
            self._thread.join(timeout=self.config.connect_timeout_seconds + 1.0)
        self._thread = None

    def _run(self) -> None:
        interval = 1.0 / self.config.send_hz
        next_send = time.monotonic()
        next_connect = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._socket is None:
                if now < next_connect:
                    self._stop_event.wait(min(next_connect - now, 0.1))
                    continue
                if not self._connect():
                    next_connect = time.monotonic() + self.config.reconnect_interval_seconds
                    continue
                next_send = time.monotonic()

            wait_seconds = next_send - time.monotonic()
            if wait_seconds > 0.0:
                self._stop_event.wait(min(wait_seconds, 0.1))
                continue
            next_send += interval
            if next_send < time.monotonic() - interval:
                next_send = time.monotonic() + interval

            packet = self._current_packet()
            if packet is None:
                continue
            try:
                assert self._socket is not None
                self._socket.sendall(packet)
            except OSError as exc:
                LOG.warning("TCP send failed (%s); reconnecting", exc)
                self._close_socket()
                next_connect = time.monotonic() + self.config.reconnect_interval_seconds

        self._close_socket()

    def _current_packet(self) -> Optional[bytes]:
        with self._lock:
            if self._latest_packet is None:
                return None
            if time.monotonic() - self._latest_update > self.config.status_stale_seconds:
                return None
            return self._latest_packet

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
                "Connected to Raspberry Pi at %s:%d",
                self.config.host,
                self.config.port,
            )
            return True
        except OSError as exc:
            LOG.warning(
                "Could not connect to Raspberry Pi at %s:%d (%s); retrying",
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


def with_network_overrides(
    config: NetworkConfig,
    host: Optional[str],
    port: Optional[int],
    disabled: bool,
) -> NetworkConfig:
    return replace(
        config,
        enabled=False if disabled else config.enabled,
        host=host or config.host,
        port=port if port is not None else config.port,
    )
