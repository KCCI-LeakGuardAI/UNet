from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from src.data.preprocessing import load_pair

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetFiles:
    image_paths: list[Path]
    mask_paths: list[Path]

    def __len__(self) -> int:
        return len(self.image_paths)


def discover_pairs(image_root: str | Path, mask_root: str | Path) -> DatasetFiles:
    image_root, mask_root = Path(image_root), Path(mask_root)
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Image or mask root not found: {image_root}, {mask_root}")
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    missing: list[str] = []
    for image_path in images:
        relative = image_path.relative_to(image_root)
        mask_path = mask_root / relative.with_suffix(".png")
        if not mask_path.is_file():
            missing.append(relative.as_posix())
            continue
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} masks, including: {preview}")
    if not image_paths:
        raise ValueError(f"No image/mask pairs found under {image_root}")
    return DatasetFiles(image_paths, mask_paths)


def build_dataset(
    files: DatasetFiles,
    image_height: int,
    image_width: int,
    batch_size: int,
    training: bool,
    seed: int = 42,
    shuffle_buffer: int | None = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((
        [str(path) for path in files.image_paths],
        [str(path) for path in files.mask_paths],
    ))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(shuffle_buffer or len(files), len(files)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        lambda image_path, mask_path: load_pair(
            image_path, mask_path, image_height, image_width
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = not training
    return dataset.with_options(options)
