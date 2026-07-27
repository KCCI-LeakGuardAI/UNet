from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.data.dataset_loader import build_dataset, discover_pairs
from src.data.preprocessing import (
    decode_and_preprocess_image,
    decode_and_preprocess_mask,
)
from src.models.losses import bce_dice_loss
from src.models.metrics import custom_objects, training_metrics
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("evaluate")
FIELDS = [
    "relative_image", "dice", "iou", "precision", "recall",
    "true_pixels", "predicted_pixels", "tp", "fp", "fn",
]


def binary_metrics(true_mask: np.ndarray, predicted_mask: np.ndarray) -> dict[str, float]:
    true_value = true_mask.astype(bool)
    predicted_value = predicted_mask.astype(bool)
    tp = int(np.count_nonzero(true_value & predicted_value))
    fp = int(np.count_nonzero(~true_value & predicted_value))
    fn = int(np.count_nonzero(true_value & ~predicted_value))
    true_pixels, predicted_pixels = int(true_value.sum()), int(predicted_value.sum())
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "true_pixels": true_pixels, "predicted_pixels": predicted_pixels,
        "tp": tp, "fp": fp, "fn": fn,
    }


def save_prediction_panel(
    image_path: Path,
    true_mask_path: Path,
    probability: np.ndarray,
    threshold: float,
    output: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    true_mask = cv2.imread(str(true_mask_path), cv2.IMREAD_GRAYSCALE)
    predicted = (probability >= threshold).astype(np.uint8) * 255
    predicted = cv2.resize(
        predicted, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    true_bgr = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)
    predicted_bgr = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    selected = predicted > 0
    color = np.zeros_like(image)
    color[:, :, 2] = predicted
    overlay[selected] = cv2.addWeighted(image, 0.55, color, 0.45, 0)[selected]
    panel = np.hstack((image, true_bgr, predicted_bgr, overlay))
    labels = ("ORIGINAL", "GROUND TRUTH", "PREDICTION", "OVERLAY")
    part_width = image.shape[1]
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (0, 0, 0), -1)
    for index, label in enumerate(labels):
        cv2.putText(
            panel, label, (index * part_width + 5, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), panel)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net on untouched Test data.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_yaml(args.config)
    model_config = require(config, "model")
    data_config = require(config, "data")
    train_config = require(config, "training")
    output_config = require(config, "output")
    output_root = Path(args.output or output_config["root"])
    model_path = Path(args.model or output_root / output_config["best_model"])
    prediction_root = output_root / output_config["predictions"]
    if not model_path.is_file():
        LOG.error("Model not found: %s", model_path)
        return 2
    test_files = discover_pairs(data_config["test_images"], data_config["test_masks"])
    height, width = int(model_config["input_height"]), int(model_config["input_width"])
    batch_size = int(train_config["batch_size"])
    test_dataset = build_dataset(test_files, height, width, batch_size, False)
    model = tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects(), compile=False
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(train_config["learning_rate"])),
        loss=bce_dice_loss,
        metrics=training_metrics(),
    )
    aggregate = model.evaluate(test_dataset, verbose=1, return_dict=True)
    rows: list[dict[str, object]] = []
    image_root = Path(data_config["test_images"])
    for image_path, mask_path in zip(test_files.image_paths, test_files.mask_paths):
        image = decode_and_preprocess_image(
            tf.constant(str(image_path)), height, width
        )
        probability = model.predict(image[None, ...], verbose=0)[0, :, :, 0]
        resized_true = decode_and_preprocess_mask(
            tf.constant(str(mask_path)), height, width
        ).numpy()[:, :, 0].astype(bool)
        predicted = probability >= args.threshold
        metrics = binary_metrics(resized_true, predicted)
        relative = image_path.relative_to(image_root)
        rows.append({"relative_image": relative.as_posix(), **metrics})
        save_prediction_panel(
            image_path, mask_path, probability, args.threshold,
            prediction_root / relative.with_suffix(".png"),
        )
    results_csv = output_root / "logs" / "test_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    totals = {
        field: int(sum(int(row[field]) for row in rows))
        for field in ("tp", "fp", "fn")
    }
    total_tp, total_fp, total_fn = totals["tp"], totals["fp"], totals["fn"]
    pixel_global = {
        "dice": (2 * total_tp + 1e-6)
        / (2 * total_tp + total_fp + total_fn + 1e-6),
        "iou": (total_tp + 1e-6)
        / (total_tp + total_fp + total_fn + 1e-6),
        "precision": (total_tp + 1e-6) / (total_tp + total_fp + 1e-6),
        "recall": (total_tp + 1e-6) / (total_tp + total_fn + 1e-6),
        **totals,
    }
    class_names = sorted({str(row["relative_image"]).split("/")[0] for row in rows})
    per_class = {}
    for class_name in class_names:
        class_rows = [
            row for row in rows
            if str(row["relative_image"]).split("/")[0] == class_name
        ]
        per_class[class_name] = {
            "count": len(class_rows),
            **{
                metric: float(np.mean([float(row[metric]) for row in class_rows]))
                for metric in ("dice", "iou", "precision", "recall")
            },
        }
    summary = {
        "model": str(model_path),
        "threshold": args.threshold,
        "test_pairs": len(rows),
        "keras_evaluation": {key: float(value) for key, value in aggregate.items()},
        "pixel_global": pixel_global,
        "per_image_mean": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in ("dice", "iou", "precision", "recall")
        },
        "per_image_median": {
            metric: float(np.median([float(row[metric]) for row in rows]))
            for metric in ("dice", "iou", "precision", "recall")
        },
        "per_class_mean": per_class,
    }
    summary_path = output_root / "logs" / "test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Test evaluation complete: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
