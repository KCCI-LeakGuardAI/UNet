from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.dataset_loader import build_dataset, discover_pairs
from src.models.losses import bce_dice_loss
from src.models.metrics import training_metrics
from src.models.unet import build_unet
from src.monitoring.training_plots import save_training_plots
from src.utils.config import load_yaml, require
from src.utils.logger import configure_logging

LOG = logging.getLogger("train")


def configure_gpu() -> list[str]:
    devices = tf.config.experimental.list_physical_devices("GPU")
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as exc:
            LOG.warning("Could not enable GPU memory growth: %s", exc)
    return [device.name for device in devices]


def resolve_output(root: Path, configured: str) -> Path:
    return root / configured


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the lightweight binary U-Net.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--epochs", type=int, help="Override configured epochs")
    parser.add_argument("--output", help="Override output root")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_yaml(args.config)
    model_config = require(config, "model")
    data_config = require(config, "data")
    train_config = require(config, "training")
    output_config = require(config, "output")
    seed = int(data_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpu_devices = configure_gpu()
    LOG.info("TensorFlow=%s GPUs=%s", tf.__version__, gpu_devices or "CPU only")

    train_files = discover_pairs(data_config["train_images"], data_config["train_masks"])
    val_files = discover_pairs(data_config["val_images"], data_config["val_masks"])
    height, width = int(model_config["input_height"]), int(model_config["input_width"])
    batch_size = int(train_config["batch_size"])
    train_dataset = build_dataset(
        train_files, height, width, batch_size, True, seed,
        int(data_config.get("shuffle_buffer", len(train_files))),
    )
    val_dataset = build_dataset(
        val_files, height, width, batch_size, False, seed
    )
    model = build_unet(
        input_height=height,
        input_width=width,
        input_channels=int(model_config["input_channels"]),
        base_channels=int(model_config["base_channels"]),
        output_channels=int(model_config["output_channels"]),
        dropout=float(model_config["dropout"]),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(train_config["learning_rate"])),
        loss=bce_dice_loss,
        metrics=training_metrics(),
    )
    model.summary(print_fn=lambda line: LOG.info(line))
    output_root = Path(args.output or output_config["root"])
    best_model = resolve_output(output_root, output_config["best_model"])
    final_model = resolve_output(output_root, output_config["final_model"])
    saved_model = resolve_output(output_root, output_config["saved_model"])
    training_log = resolve_output(output_root, output_config["training_log"])
    summary_path = resolve_output(output_root, output_config["summary"])
    plots_dir = resolve_output(output_root, output_config["plots"])
    for path in (best_model, final_model, training_log, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if saved_model.exists():
        shutil.rmtree(saved_model)
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    checkpoint_metric = str(train_config["checkpoint_metric"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model), monitor=checkpoint_metric, mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=checkpoint_metric, mode="max",
            patience=int(train_config["early_stopping_patience"]),
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5,
            patience=int(train_config["reduce_lr_patience"]),
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(training_log)),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(str(final_model), include_optimizer=True)
    model.save(str(saved_model), include_optimizer=False, save_format="tf")
    save_training_plots(history.history, plots_dir)
    best_epoch = int(np.argmax(history.history[checkpoint_metric])) + 1
    summary = {
        "tensorflow_version": tf.__version__,
        "gpu_devices": gpu_devices,
        "train_pairs": len(train_files),
        "validation_pairs": len(val_files),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_dice_score": float(max(history.history[checkpoint_metric])),
        "best_val_iou_score": float(max(history.history.get("val_iou_score", [0.0]))),
        "best_model": str(best_model),
        "final_model": str(final_model),
        "saved_model": str(saved_model),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Training complete: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
