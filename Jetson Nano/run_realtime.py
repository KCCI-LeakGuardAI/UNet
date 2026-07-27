from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from leakguard.analyzer import LeakAnalyzer
from leakguard.camera import FrameSource
from leakguard.config import load_config
from leakguard.event_logger import StatusCsvLogger
from leakguard.predictor import UNetPredictor
from leakguard.visualization import Telemetry, render_dashboard

LOG = logging.getLogger("leakguard")
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Jetson Nano real-time U-Net liquid leak detection"
    )
    parser.add_argument(
        "--config", default=str(root / "config.yaml"), help="YAML config path"
    )
    parser.add_argument("--model", help="Override model .h5 path")
    parser.add_argument("--camera", type=int, help="Override USB camera index")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Offline video file for validation")
    source.add_argument("--image", help="Single image for validation")
    source.add_argument("--gstreamer", help="GStreamer camera pipeline")
    parser.add_argument(
        "--headless", action="store_true", help="Disable OpenCV window"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames; 0=unlimited"
    )
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    config = load_config(args.config)
    if args.model:
        config = replace(
            config,
            model=replace(
                config.model, path=Path(args.model).expanduser().resolve()
            ),
        )

    predictor = UNetPredictor(config.model)
    analyzer = LeakAnalyzer(
        config.analysis, config.roi.danger_minimum_overlap_pixels
    )
    source = FrameSource(
        config.camera,
        camera_index=args.camera,
        media_path=args.video or args.image,
        gstreamer_pipeline=args.gstreamer,
    )
    csv_logger: Optional[StatusCsvLogger] = None
    if config.logging.enabled and not args.no_log:
        csv_logger = StatusCsvLogger(
            config.logging.csv_path, config.logging.interval_seconds
        )

    frame_index = 0
    failed_reads = 0
    fps = 0.0
    previous_time = time.perf_counter()
    last_console = 0.0
    last_rendered = None
    try:
        source.open()
        LOG.info("Inference started from %s", source.description)
        while not STOP_REQUESTED:
            loop_started = time.perf_counter()
            ok, frame = source.read()
            if not ok or frame is None:
                if args.image or args.video:
                    LOG.info("Input media finished")
                    break
                failed_reads += 1
                if failed_reads >= config.camera.read_fail_limit:
                    raise RuntimeError(
                        f"Camera read failed {failed_reads} consecutive times"
                    )
                time.sleep(0.02)
                continue
            failed_reads = 0
            frame_index += 1
            frame_height, frame_width = frame.shape[:2]
            monitoring_roi, danger_roi = config.roi.for_frame(
                frame_width, frame_height
            )
            prediction = predictor.predict(frame, monitoring_roi)
            status = analyzer.analyze(
                prediction.mask, monitoring_roi, danger_roi
            )
            if args.image:
                for _ in range(config.analysis.detection_required_frames - 1):
                    status = analyzer.analyze(
                        prediction.mask, monitoring_roi, danger_roi
                    )
            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instantaneous_fps if fps == 0.0 else 0.15 * instantaneous_fps + 0.85 * fps
            previous_time = now
            telemetry = Telemetry(
                fps=fps,
                end_to_end_ms=(now - loop_started) * 1000.0,
                frame_index=frame_index,
                device=predictor.device,
            )
            last_rendered = render_dashboard(
                frame=frame,
                prediction=prediction,
                status=status,
                monitoring_roi=monitoring_roi,
                danger_roi=danger_roi,
                telemetry=telemetry,
                overlay_alpha=config.display.overlay_alpha,
                threshold=config.model.threshold,
                show_mask_preview=config.display.show_mask_preview,
            )
            if csv_logger:
                csv_logger.write(status, prediction, telemetry)
            if now - last_console >= config.display.console_interval_seconds:
                LOG.info(
                    "level=%s ratio=%.3f%% detected=%s spreading=%s "
                    "danger=%s inference=%.1fms fps=%.1f",
                    status.level,
                    status.leak_ratio,
                    status.detected,
                    status.spreading,
                    status.danger_overlap,
                    prediction.inference_ms,
                    fps,
                )
                last_console = now

            if not args.headless:
                cv2.imshow(config.display.window_name, last_rendered)
                key = cv2.waitKey(0 if args.image else 1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    save_snapshot(
                        last_rendered, config.display.snapshot_directory
                    )
                if key in (ord("r"), ord("R")):
                    analyzer.reset()
                    LOG.info("Leak analyzer state reset")
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

        if args.image and args.headless and last_rendered is not None:
            save_snapshot(last_rendered, config.display.snapshot_directory)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    finally:
        source.release()
        if csv_logger:
            csv_logger.close()
        cv2.destroyAllWindows()
        LOG.info("Resources released; processed frames=%d", frame_index)


def save_snapshot(image: object, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = directory / f"leakguard_{timestamp}.jpg"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save snapshot: {path}")
    LOG.info("Snapshot saved: %s", path)
    return path


if __name__ == "__main__":
    sys.exit(main())
