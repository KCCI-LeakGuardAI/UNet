from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def clip(self, frame_width: int, frame_height: int) -> "Rect":
        x1 = max(0, min(self.x, frame_width))
        y1 = max(0, min(self.y, frame_height))
        x2 = max(x1, min(self.x2, frame_width))
        y2 = max(y1, min(self.y2, frame_height))
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def scale(
        self,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> "Rect":
        return Rect(
            x=int(round(self.x * target_width / source_width)),
            y=int(round(self.y * target_height / source_height)),
            width=int(round(self.width * target_width / source_width)),
            height=int(round(self.height * target_height / source_height)),
        ).clip(target_width, target_height)


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int
    backend: str
    fourcc: str
    read_fail_limit: int


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    threshold: float
    crop_to_monitoring_roi: bool
    min_component_pixels: int
    morphology_kernel: int
    open_iterations: int
    close_iterations: int


@dataclass(frozen=True)
class AnalysisConfig:
    leak_detect_ratio: float
    warning_ratio: float
    danger_ratio: float
    ema_alpha: float
    detection_window_frames: int
    detection_required_frames: int
    spreading_window_frames: int
    spreading_delta_ratio: float
    spreading_confirm_frames: int
    spreading_escalates: bool


@dataclass(frozen=True)
class RoiConfig:
    reference_width: int
    reference_height: int
    monitoring: Rect
    danger: Rect
    danger_minimum_overlap_pixels: int

    def for_frame(self, width: int, height: int) -> tuple[Rect, Rect]:
        return (
            self.monitoring.scale(
                self.reference_width, self.reference_height, width, height
            ),
            self.danger.scale(
                self.reference_width, self.reference_height, width, height
            ),
        )


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str
    overlay_alpha: float
    show_mask_preview: bool
    console_interval_seconds: float
    snapshot_directory: Path


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool
    csv_path: Path
    interval_seconds: float


@dataclass(frozen=True)
class NetworkConfig:
    enabled: bool
    host: str
    port: int
    send_hz: float
    connect_timeout_seconds: float
    reconnect_interval_seconds: float
    status_stale_seconds: float


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    model: ModelConfig
    analysis: AnalysisConfig
    roi: RoiConfig
    display: DisplayConfig
    logging: LoggingConfig
    network: NetworkConfig


def _rect(value: Dict[str, Any], name: str) -> Rect:
    try:
        rect = Rect(
            x=int(value["x"]),
            y=int(value["y"]),
            width=int(value["width"]),
            height=int(value["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rectangle config: {name}") from exc
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError(f"{name} width and height must be positive")
    return rect


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    base = config_path.parent
    try:
        camera = raw["camera"]
        model = raw["model"]
        analysis = raw["analysis"]
        roi = raw["roi"]
        display = raw["display"]
        log = raw["logging"]
    except KeyError as exc:
        raise ValueError(f"Missing top-level config section: {exc.args[0]}") from exc
    network = raw.get("network", {})

    app_config = AppConfig(
        camera=CameraConfig(
            index=int(camera["index"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["fps"]),
            backend=str(camera.get("backend", "auto")).lower(),
            fourcc=str(camera.get("fourcc", "MJPG")),
            read_fail_limit=int(camera.get("read_fail_limit", 30)),
        ),
        model=ModelConfig(
            path=_resolve(base, str(model["path"])),
            threshold=float(model["threshold"]),
            crop_to_monitoring_roi=bool(
                model.get("crop_to_monitoring_roi", False)
            ),
            min_component_pixels=int(model["min_component_pixels"]),
            morphology_kernel=int(model["morphology_kernel"]),
            open_iterations=int(model.get("open_iterations", 1)),
            close_iterations=int(model.get("close_iterations", 1)),
        ),
        analysis=AnalysisConfig(
            leak_detect_ratio=float(analysis["leak_detect_ratio"]),
            warning_ratio=float(analysis["warning_ratio"]),
            danger_ratio=float(analysis["danger_ratio"]),
            ema_alpha=float(analysis["ema_alpha"]),
            detection_window_frames=int(analysis["detection_window_frames"]),
            detection_required_frames=int(analysis["detection_required_frames"]),
            spreading_window_frames=int(analysis["spreading_window_frames"]),
            spreading_delta_ratio=float(analysis["spreading_delta_ratio"]),
            spreading_confirm_frames=int(analysis["spreading_confirm_frames"]),
            spreading_escalates=bool(
                analysis.get("spreading_escalates", True)
            ),
        ),
        roi=RoiConfig(
            reference_width=int(roi["reference_width"]),
            reference_height=int(roi["reference_height"]),
            monitoring=_rect(roi["monitoring"], "roi.monitoring"),
            danger=_rect(roi["danger"], "roi.danger"),
            danger_minimum_overlap_pixels=int(
                roi["danger"].get("minimum_overlap_pixels", 20)
            ),
        ),
        display=DisplayConfig(
            window_name=str(display.get("window_name", "LeakGuard AI")),
            overlay_alpha=float(display.get("overlay_alpha", 0.45)),
            show_mask_preview=bool(display.get("show_mask_preview", True)),
            console_interval_seconds=float(
                display.get("console_interval_seconds", 2.0)
            ),
            snapshot_directory=_resolve(
                base, str(display.get("snapshot_directory", "snapshots"))
            ),
        ),
        logging=LoggingConfig(
            enabled=bool(log.get("enabled", True)),
            csv_path=_resolve(base, str(log.get("csv_path", "logs/status.csv"))),
            interval_seconds=float(log.get("interval_seconds", 1.0)),
        ),
        network=NetworkConfig(
            enabled=bool(network.get("enabled", False)),
            host=str(network.get("host", "127.0.0.1")),
            port=int(network.get("port", 5000)),
            send_hz=float(network.get("send_hz", 10.0)),
            connect_timeout_seconds=float(
                network.get("connect_timeout_seconds", 2.0)
            ),
            reconnect_interval_seconds=float(
                network.get("reconnect_interval_seconds", 2.0)
            ),
            status_stale_seconds=float(
                network.get("status_stale_seconds", 0.5)
            ),
        ),
    )
    _validate(app_config)
    return app_config


def _validate(config: AppConfig) -> None:
    if not 0.0 < config.model.threshold < 1.0:
        raise ValueError("model.threshold must be between 0 and 1")
    if config.model.morphology_kernel < 1:
        raise ValueError("model.morphology_kernel must be positive")
    if config.model.morphology_kernel % 2 == 0:
        raise ValueError("model.morphology_kernel must be odd")
    ratios = (
        config.analysis.leak_detect_ratio,
        config.analysis.warning_ratio,
        config.analysis.danger_ratio,
    )
    if not 0.0 <= ratios[0] < ratios[1] < ratios[2]:
        raise ValueError(
            "Area thresholds must satisfy detect < warning < danger"
        )
    if not 0.0 < config.analysis.ema_alpha <= 1.0:
        raise ValueError("analysis.ema_alpha must be in (0, 1]")
    if not (
        1
        <= config.analysis.detection_required_frames
        <= config.analysis.detection_window_frames
    ):
        raise ValueError("Invalid consecutive-frame detection settings")
    if not config.network.host:
        raise ValueError("network.host must not be empty")
    if not 1 <= config.network.port <= 65535:
        raise ValueError("network.port must be in 1..65535")
    if config.network.send_hz <= 0.0:
        raise ValueError("network.send_hz must be positive")
    if config.network.connect_timeout_seconds <= 0.0:
        raise ValueError("network.connect_timeout_seconds must be positive")
    if config.network.reconnect_interval_seconds <= 0.0:
        raise ValueError("network.reconnect_interval_seconds must be positive")
    if config.network.status_stale_seconds <= 0.0:
        raise ValueError("network.status_stale_seconds must be positive")
