"""
Đọc manifest (data/splits/*.json) -> tf.data.Dataset cho train/val/test.
Ưu tiên đọc ảnh 224x224 trong data/processed (nhanh); nếu chưa có thì
đọc ảnh gốc trong data/raw (kết quả như nhau).
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.densenet121.config import (
    ROOT_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
    CLASS_NAMES_PATH,
    USE_MIX_AUG,
    MIX_PROB,
    MIXUP_ALPHA,
    CUTMIX_ALPHA,
)
from src.densenet121.image_utils import load_image_array, processed_path_for

AUTOTUNE = tf.data.AUTOTUNE


def load_json(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Khong tim thay {path}. Hay chay: python -m src.densenet121.prepare_data"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path):
    """Manifest lưu đường dẫn tương đối so với thư mục dự án."""
    path = Path(path)
    return str(path if path.is_absolute() else ROOT_DIR / path)


def resolve_image_path(path):
    raw_path = resolve_path(path)
    processed = processed_path_for(raw_path)
    if processed is not None and processed.exists():
        return str(processed)
    return raw_path


def read_images(records):
    """Đọc toàn bộ ảnh vào RAM (2376 ảnh x 224x224x3 ~ 360 MB)."""
    paths = [resolve_image_path(r["path"]) for r in records]
    missing = sum("processed" not in Path(p).parts for p in paths)
    if missing:
        print(
            f"  Canh bao: {missing}/{len(paths)} anh doc tu data/raw (cham hon). "
            "Chay 'python -m src.densenet121.prepare_data --processed' de tao data/processed."
        )
    images = np.stack([load_image_array(p, IMAGE_SIZE) for p in paths])
    labels = np.array([r["label"] for r in records], dtype=np.int32)
    return images, labels


# ---------------------------------------------------------------------------
# MixUp / CutMix (tuỳ chọn, chỉ cho tập train) - bật bằng USE_MIX_AUG
# ---------------------------------------------------------------------------

def sample_beta(alpha, shape):
    x = tf.random.gamma(shape, alpha)
    y = tf.random.gamma(shape, alpha)
    return x / (x + y)


def mixup(images, labels):
    n = tf.shape(images)[0]
    index = tf.random.shuffle(tf.range(n))
    lam = sample_beta(MIXUP_ALPHA, [n])
    lam = tf.maximum(lam, 1.0 - lam)
    images = (tf.reshape(lam, [-1, 1, 1, 1]) * images
              + tf.reshape(1.0 - lam, [-1, 1, 1, 1]) * tf.gather(images, index))
    labels = (tf.reshape(lam, [-1, 1]) * labels
              + tf.reshape(1.0 - lam, [-1, 1]) * tf.gather(labels, index))
    return images, labels


def cutmix(images, labels):
    n = tf.shape(images)[0]
    height, width = IMAGE_SIZE
    index = tf.random.shuffle(tf.range(n))

    lam = sample_beta(CUTMIX_ALPHA, [n])
    ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(ratio * height, tf.int32)
    cut_w = tf.cast(ratio * width, tf.int32)
    cy = tf.random.uniform([n], 0, height, dtype=tf.int32)
    cx = tf.random.uniform([n], 0, width, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)

    rows = tf.range(height)[tf.newaxis, :]
    cols = tf.range(width)[tf.newaxis, :]
    mask_y = (rows >= y1[:, None]) & (rows < y2[:, None])
    mask_x = (cols >= x1[:, None]) & (cols < x2[:, None])
    mask = tf.cast(mask_y[:, :, None] & mask_x[:, None, :], tf.float32)[..., None]

    images = images * (1.0 - mask) + tf.gather(images, index) * mask
    area = tf.reduce_mean(mask, axis=[1, 2, 3])[:, None]
    labels = labels * (1.0 - area) + tf.gather(labels, index) * area
    return images, labels


def mix_augment(images, labels):
    choice = tf.random.uniform([])
    return tf.case(
        [
            (choice < MIX_PROB / 2, lambda: mixup(images, labels)),
            (choice < MIX_PROB, lambda: cutmix(images, labels)),
        ],
        default=lambda: (images, labels),
    )


# ---------------------------------------------------------------------------

def create_dataset(records, num_classes, training=False):
    images, labels = read_images(records)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        dataset = dataset.shuffle(len(records), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE)
    # float32 [0, 255] + nhãn one-hot (để dùng label smoothing)
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32), tf.one_hot(y, num_classes)),
        num_parallel_calls=AUTOTUNE,
    )

    if training and USE_MIX_AUG:
        dataset = dataset.map(mix_augment, num_parallel_calls=AUTOTUNE)

    return dataset.prefetch(AUTOTUNE)


def load_data():
    """Trả về (train_ds, val_ds, test_ds, class_names)."""
    class_names = load_json(CLASS_NAMES_PATH)
    train_records = load_json(TRAIN_MANIFEST)
    val_records = load_json(VAL_MANIFEST)
    test_records = load_json(TEST_MANIFEST)
    test_paths = np.asarray(
    [
        resolve_image_path(record["path"])
        for record in test_records
    ],
    dtype=str
)

    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    print("Cac lop:", ", ".join(f"{i}={n}" for i, n in enumerate(class_names)))

    num_classes = len(class_names)
    return (
        create_dataset(train_records, num_classes, training=True),
        create_dataset(val_records, num_classes),
        create_dataset(test_records, num_classes),
        class_names,
    )


def compute_class_weights(class_names):
    """Trọng số lớp = tổng / (số lớp x số ảnh của lớp): lớp ít ảnh -> trọng số lớn."""
    labels = [r["label"] for r in load_json(TRAIN_MANIFEST)]
    counts = np.bincount(labels, minlength=len(class_names))
    total = counts.sum()
    return {i: float(total / (len(class_names) * c)) for i, c in enumerate(counts)}
