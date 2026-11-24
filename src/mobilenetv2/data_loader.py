import json
from pathlib import Path

import tensorflow as tf

from src.mobilenetv2.config import (
    ROOT_DIR,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)
from src.mobilenetv2.image_utils import load_image_array, processed_path_for

AUTOTUNE = tf.data.AUTOTUNE


def load_json(file_path):
    """Đọc dữ liệu từ file JSON."""
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Khong tim thay {file_path}. Hay chay: python -m src.mobilenetv2.prepare_data"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path):
    """Đường dẫn trong manifest (tuyệt đối hoặc tương đối so với thư mục dự án)."""
    path = Path(path)
    return str(path if path.is_absolute() else ROOT_DIR / path)


def resolve_image_path(path):
    """Dùng ảnh 224x224 trong data/processed nếu đã có (nhanh hơn), không thì ảnh gốc."""
    raw_path = resolve_path(path)
    processed = processed_path_for(raw_path)
    if processed is not None and processed.exists():
        return str(processed)
    return raw_path


def _read_image(path):
    return load_image_array(path.decode("utf-8"), IMAGE_SIZE)


def load_image(image_path, label):
    """
    Đọc ảnh bằng image_utils.load_image_array - GIỐNG HỆT predict.py / gradcam.py
    (xoay theo EXIF, RGB, resize có antialias). Đọc bằng PIL nên hỗ trợ cả .webp.
    """

    image = tf.numpy_function(_read_image, [image_path], tf.uint8)

    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    # Giữ pixel ở thang [0, 255]. model.py sẽ preprocess đúng một lần.
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)

    return image, label


def create_dataset(records, training=False):
    """Tạo TensorFlow Dataset từ danh sách ảnh."""

    image_paths = [
        resolve_image_path(record["path"]) for record in records
    ]

    labels = [
        record["label"] for record in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def load_data():
    """Đọc và tạo ba tập dữ liệu."""

    print("Dang doc cac file JSON...")

    train_records = load_json(TRAIN_MANIFEST)
    val_records = load_json(VAL_MANIFEST)
    test_records = load_json(TEST_MANIFEST)

    class_names = load_json(CLASS_NAMES_PATH)

    print("\nSo luong anh:")
    print("Tổng số ảnh sau khi xử lý :", len(train_records) + len(val_records) + len(test_records))
    print("Training:", len(train_records))
    print("Validation:", len(val_records))
    print("Testing:", len(test_records))

    print("\nDanh sach lop:")
    for index, name in enumerate(class_names):
        print(f"{index}: {name}")

    train_dataset = create_dataset(
        train_records,
        training=True
    )

    val_dataset = create_dataset(
        val_records,
        training=False
    )

    test_dataset = create_dataset(
        test_records,
        training=False
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names
    )


def check_dataset():
    """Kiểm tra kích thước ảnh và nhãn."""

    train_dataset, val_dataset, test_dataset, class_names = (
        load_data()
    )

    print("\nKIEM TRA DU LIEU DA TIEN XU LY")

    for images, labels in train_dataset.take(1):
        print("Kich thuoc batch anh:", images.shape)
        print("Kich thuoc batch nhan:", labels.shape)
        print("Gia tri pixel nho nhat:",
              float(tf.reduce_min(images)))
        print("Gia tri pixel lon nhat:",
              float(tf.reduce_max(images)))

    print("\nSo lop:", len(class_names))
    print("So batch Training:",
          tf.data.experimental.cardinality(train_dataset).numpy())
    print("So batch Validation:",
          tf.data.experimental.cardinality(val_dataset).numpy())
    print("So batch Testing:",
          tf.data.experimental.cardinality(test_dataset).numpy())

    print("\nHOAN THANH KIEM TRA DATA LOADER!")


if __name__ == "__main__":
    check_dataset()
