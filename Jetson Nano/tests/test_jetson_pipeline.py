from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from leakguard.analyzer import LeakAnalyzer, LeakStatus
from leakguard.config import NetworkConfig, Rect, VideoStreamConfig, load_config
from leakguard.postprocess import postprocess_probability
from leakguard.tcp_client import (
    LatestStatusTcpClient,
    encode_status_packet,
    expand_network_targets,
)
from leakguard.video_streamer import (
    LatestFrameTcpStreamer,
    VIDEO_FLAG_ANNOTATED,
    VIDEO_HEADER,
    VIDEO_HEADER_SIZE,
    VIDEO_MAGIC,
    VIDEO_VERSION,
    encode_video_packet,
)


ROOT = Path(__file__).resolve().parents[1]


def test_config_and_roi_scaling() -> None:
    config = load_config(ROOT / "config.yaml")
    monitoring, danger = config.roi.for_frame(320, 240)
    assert monitoring == Rect(40, 30, 240, 180)
    assert danger == Rect(215, 130, 50, 50)
    assert [target.port for target in expand_network_targets(config.network)] == [
        5000,
        5001,
    ]
    assert config.video_stream.enabled
    assert config.video_stream.port == 5003
    assert config.video_stream.fps == 5.0


def test_postprocess_removes_small_components() -> None:
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[10:12, 10:12] = 1.0
    probability[30:50, 30:50] = 1.0
    result = postprocess_probability(
        probability,
        threshold=0.5,
        min_component_pixels=20,
        kernel_size=1,
        open_iterations=0,
        close_iterations=0,
    )
    assert result.component_count == 1
    assert int(result.mask.sum()) == 400


def test_analyzer_levels_and_danger_override() -> None:
    config = load_config(ROOT / "config.yaml")
    analyzer = LeakAnalyzer(config.analysis, danger_minimum_overlap_pixels=20)
    monitoring = Rect(0, 0, 100, 100)
    danger = Rect(80, 80, 20, 20)

    normal = np.zeros((100, 100), dtype=np.uint8)
    assert analyzer.analyze(normal, monitoring, danger).level == "NORMAL"

    small = normal.copy()
    small[10:20, 10:20] = 1
    analyzer.analyze(small, monitoring, danger)
    small_status = analyzer.analyze(small, monitoring, danger)
    assert small_status.detected
    assert small_status.level == "SMALL_LEAK"

    danger_mask = small.copy()
    danger_mask[80:85, 80:85] = 1
    danger_status = analyzer.analyze(danger_mask, monitoring, danger)
    assert danger_status.danger_overlap
    assert danger_status.level == "DANGER"


def test_raspberry_pi_status_packet() -> None:
    status = LeakStatus(
        detected=True,
        leak_ratio=5.0,
        instant_ratio=5.0,
        spreading=True,
        spreading_delta=0.5,
        danger_overlap=True,
        leak_pixels=500,
        monitoring_pixels=10000,
        danger_overlap_pixels=25,
        level="DANGER",
        timestamp=0.0,
    )
    assert encode_status_packet(status) == b"3 500 1 1\n"


def test_tcp_client_sends_latest_status_at_10_hz() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    host, port = listener.getsockname()
    received = bytearray()

    def receive_until_closed() -> None:
        connection, _ = listener.accept()
        with connection:
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                received.extend(chunk)

    receiver = threading.Thread(target=receive_until_closed)
    receiver.start()
    client = LatestStatusTcpClient(
        NetworkConfig(
            enabled=True,
            host=host,
            port=port,
            send_hz=10.0,
            connect_timeout_seconds=1.0,
            reconnect_interval_seconds=0.1,
            status_stale_seconds=2.0,
        )
    )
    status = LeakStatus(
        detected=True,
        leak_ratio=2.36,
        instant_ratio=2.4,
        spreading=True,
        spreading_delta=0.3,
        danger_overlap=False,
        leak_pixels=236,
        monitoring_pixels=10000,
        danger_overlap_pixels=0,
        level="WARNING",
        timestamp=0.0,
    )
    try:
        client.update(status)
        client.start()
        time.sleep(0.55)
    finally:
        client.stop()
        receiver.join(timeout=2.0)
        listener.close()

    lines = received.splitlines()
    assert 5 <= len(lines) <= 7
    assert all(line == b"2 236 1 0" for line in lines)


def test_video_packet_header_and_jpeg_payload() -> None:
    config = VideoStreamConfig(
        enabled=True,
        host="127.0.0.1",
        port=5003,
        fps=5.0,
        jpeg_quality=65,
        width=64,
        height=48,
        connect_timeout_seconds=1.0,
        reconnect_interval_seconds=0.1,
        frame_stale_seconds=1.0,
        max_frame_bytes=100000,
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 180
    packet = encode_video_packet(
        frame,
        sequence=123,
        timestamp_ms=1720000000123,
        config=config,
    )
    fields = VIDEO_HEADER.unpack(packet[:VIDEO_HEADER_SIZE])
    magic, version, flags, header_size, sequence, timestamp_ms, jpeg_size = fields

    assert magic == VIDEO_MAGIC
    assert version == VIDEO_VERSION
    assert flags == VIDEO_FLAG_ANNOTATED
    assert header_size == VIDEO_HEADER_SIZE == 24
    assert sequence == 123
    assert timestamp_ms == 1720000000123
    assert jpeg_size == len(packet) - VIDEO_HEADER_SIZE

    decoded = np.frombuffer(packet[VIDEO_HEADER_SIZE:], dtype=np.uint8)
    image = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[:2] == (48, 64)


def test_video_streamer_sends_one_complete_frame() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    host, port = listener.getsockname()
    complete = threading.Event()
    received = {}

    def receive_exact(connection: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def receive_frame() -> None:
        connection, _ = listener.accept()
        with connection:
            header = receive_exact(connection, VIDEO_HEADER_SIZE)
            if len(header) != VIDEO_HEADER_SIZE:
                return
            fields = VIDEO_HEADER.unpack(header)
            payload = receive_exact(connection, fields[-1])
            received["fields"] = fields
            received["payload"] = payload
            complete.set()

    receiver = threading.Thread(target=receive_frame)
    receiver.start()
    config = VideoStreamConfig(
        enabled=True,
        host=host,
        port=port,
        fps=5.0,
        jpeg_quality=65,
        width=64,
        height=48,
        connect_timeout_seconds=1.0,
        reconnect_interval_seconds=0.1,
        frame_stale_seconds=2.0,
        max_frame_bytes=100000,
    )
    streamer = LatestFrameTcpStreamer(config)
    frame = np.full((48, 64, 3), 127, dtype=np.uint8)
    try:
        streamer.update(frame)
        streamer.start()
        assert complete.wait(2.0)
    finally:
        streamer.stop()
        receiver.join(timeout=2.0)
        listener.close()

    fields = received["fields"]
    assert fields[0] == VIDEO_MAGIC
    assert fields[1] == VIDEO_VERSION
    assert len(received["payload"]) == fields[-1]
