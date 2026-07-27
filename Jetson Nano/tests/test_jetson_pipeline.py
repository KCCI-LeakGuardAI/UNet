from __future__ import annotations

from pathlib import Path

import numpy as np

from leakguard.analyzer import LeakAnalyzer
from leakguard.config import Rect, load_config
from leakguard.postprocess import postprocess_probability


ROOT = Path(__file__).resolve().parents[1]


def test_config_and_roi_scaling() -> None:
    config = load_config(ROOT / "config.yaml")
    monitoring, danger = config.roi.for_frame(320, 240)
    assert monitoring == Rect(40, 30, 240, 180)
    assert danger == Rect(215, 130, 50, 50)


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
