from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.utils.logger import configure_logging

LOG = logging.getLogger("test_camera_exposure")


@dataclass
class ExposureResult:
    requested_auto: float
    requested_exposure: float
    auto_set_returned: bool
    exposure_set_returned: bool
    reported_auto: float
    reported_exposure: float
    frames: int
    mean_brightness: float
    brightness_std: float


def open_camera(index: int, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    attempts = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default"),
    ]
    for backend, name in attempts:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            LOG.info("Camera %d opened with %s", index, name)
            return camera, name
        camera.release()
    raise RuntimeError(f"Camera {index} could not be opened with any Windows backend")


def collect_brightness(
    camera: cv2.VideoCapture,
    seconds: float,
    warmup_frames: int,
    show: bool,
    label: str,
) -> tuple[int, float, float, bool]:
    for _ in range(warmup_frames):
        camera.read()
    values: list[float] = []
    deadline = time.monotonic() + seconds
    cancelled = False
    while time.monotonic() < deadline:
        ok, frame = camera.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values.append(float(np.mean(gray)))
        if show:
            cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"mean brightness: {values[-1]:.1f}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Exposure control test (Q/Esc: quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cancelled = True
                break
    if not values:
        return 0, float("nan"), float("nan"), cancelled
    return len(values), float(np.mean(values)), float(np.std(values)), cancelled


def measure_setting(
    camera: cv2.VideoCapture,
    auto_value: float,
    exposure: float,
    seconds: float,
    warmup_frames: int,
    show: bool,
) -> tuple[ExposureResult, bool]:
    auto_ok = bool(camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value))
    time.sleep(0.2)
    exposure_ok = bool(camera.set(cv2.CAP_PROP_EXPOSURE, exposure))
    time.sleep(0.2)
    reported_auto = float(camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    reported_exposure = float(camera.get(cv2.CAP_PROP_EXPOSURE))
    label = (
        f"requested AE={auto_value:g}, exposure={exposure:g} | "
        f"reported AE={reported_auto:g}, exposure={reported_exposure:g}"
    )
    frames, mean, std, cancelled = collect_brightness(
        camera, seconds, warmup_frames, show, label
    )
    return ExposureResult(
        requested_auto=auto_value,
        requested_exposure=exposure,
        auto_set_returned=auto_ok,
        exposure_set_returned=exposure_ok,
        reported_auto=reported_auto,
        reported_exposure=reported_exposure,
        frames=frames,
        mean_brightness=mean,
        brightness_std=std,
    ), cancelled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test whether a Windows webcam accepts OpenCV auto/manual exposure controls."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="Measurement duration for each exposure value")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--exposures", type=float, nargs="+", default=[-4, -6, -8])
    parser.add_argument("--manual-auto-value", type=float, default=0.25,
                        help="DirectShow commonly uses 0.25 for manual mode")
    parser.add_argument("--show", action="store_true",
                        help="Show preview; place/remove a white object to inspect response")
    parser.add_argument("--open-settings", action="store_true",
                        help="Ask DirectShow to open the camera's native property dialog")
    parser.add_argument("--output", default="artifacts/camera_exposure_test.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.seconds <= 0 or args.warmup_frames < 0:
        parser.error("--seconds must be positive and --warmup-frames cannot be negative")

    try:
        camera, backend = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 2

    initial = {
        "backend": backend,
        "camera": args.camera,
        "requested_resolution": [args.width, args.height],
        "reported_resolution": [
            int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ],
        "initial_auto_exposure": float(camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
        "initial_exposure": float(camera.get(cv2.CAP_PROP_EXPOSURE)),
    }
    LOG.info("Initial properties: AE=%s exposure=%s resolution=%s",
             initial["initial_auto_exposure"], initial["initial_exposure"],
             initial["reported_resolution"])
    if args.open_settings:
        settings_opened = bool(camera.set(cv2.CAP_PROP_SETTINGS, 1))
        LOG.info("Native camera settings dialog request returned: %s", settings_opened)
        if not settings_opened:
            LOG.warning("This driver did not expose a DirectShow property dialog")
    results: list[ExposureResult] = []
    try:
        for exposure in args.exposures:
            result, cancelled = measure_setting(
                camera, args.manual_auto_value, exposure,
                args.seconds, args.warmup_frames, args.show,
            )
            results.append(result)
            LOG.info(
                "request AE=%g exposure=%g | set=%s/%s | reported=%g/%g | "
                "brightness=%.2f +/- %.2f (%d frames)",
                result.requested_auto, result.requested_exposure,
                result.auto_set_returned, result.exposure_set_returned,
                result.reported_auto, result.reported_exposure,
                result.mean_brightness, result.brightness_std, result.frames,
            )
            if cancelled:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    changed = len({round(r.reported_exposure, 6) for r in results}) > 1
    accepted_manual = any(
        r.auto_set_returned and r.exposure_set_returned and r.frames > 0
        for r in results
    )
    report = {
        **initial,
        "results": [asdict(result) for result in results],
        "summary": {
            "set_calls_accepted": accepted_manual,
            "reported_exposure_changed": changed,
            "interpretation": (
                "Manual exposure control appears available."
                if accepted_manual and changed
                else "Manual control was not conclusively confirmed; inspect the returned values and visual test."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("%s", report["summary"]["interpretation"])
    LOG.info("Report written to %s", output)
    return 0 if results and all(result.frames > 0 for result in results) else 3


if __name__ == "__main__":
    raise SystemExit(main())
