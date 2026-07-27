from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _plot_metric(
    history: Mapping[str, Sequence[float]],
    metric: str,
    validation_metric: str,
    title: str,
    output: Path,
) -> None:
    if metric not in history:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(history[metric], label="Train")
    if validation_metric in history:
        plt.plot(history[validation_metric], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def save_training_plots(
    history: Mapping[str, Sequence[float]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(history, "loss", "val_loss", "Loss", output_dir / "loss_curve.png")
    _plot_metric(
        history, "dice_score", "val_dice_score",
        "Dice Score", output_dir / "dice_curve.png",
    )
    _plot_metric(
        history, "iou_score", "val_iou_score",
        "IoU", output_dir / "iou_curve.png",
    )
