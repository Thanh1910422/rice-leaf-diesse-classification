import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

from config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    EPOCHS,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    MODEL_DIR,
    OUTPUT_DIR,
    MODEL_PATH,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
)


def load_datasets():
    """Đọc ảnh và chia train/validation có phân tầng theo lớp."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}"
        )

    class_names = sorted(
        folder.name
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_names) < 2:
        raise ValueError(
            "Cần ít nhất 2 thư mục lớp ảnh để huấn luyện."
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    file_paths = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_dir = DATA_DIR / class_name

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_extensions:
                file_paths.append(str(path))
                labels.append(label)

    if not file_paths:
        raise ValueError(f"Không tìm thấy ảnh trong: {DATA_DIR}")

    file_paths = np.array(file_paths)
    labels = np.array(labels)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=labels,
    )

       # Lưu tập validation để evaluate.py sử dụng lại
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_data = {
        "paths": val_paths.tolist(),
        "labels": val_labels.tolist(),
        "class_names": class_names,
    }

    validation_file = OUTPUT_DIR / "validation_data.json"
    with open(validation_file, "w", encoding="utf-8") as f:
        json.dump(
            validation_data,
            f,
            ensure_ascii=False,
            indent=4,
        )

    def make_dataset(paths, image_labels, training=False):
        def load_image(path, label):
            image = tf.io.read_file(path)
            image = tf.io.decode_image(
                image,
                channels=3,
                expand_animations=False,
            )
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32)
            return image, label

        ds = tf.data.Dataset.from_tensor_slices((paths, image_labels))

        if training:
            ds = ds.shuffle(
                len(paths),
                seed=SEED,
                reshuffle_each_iteration=True,
            )

        ds = ds.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    print("\nSố ảnh theo lớp:")
    for label, class_name in enumerate(class_names):
        train_count = int(np.sum(train_labels == label))
        val_count = int(np.sum(val_labels == label))
        print(
            f"{class_name}: train={train_count}, validation={val_count}"
        )

    return train_ds, val_ds, class_names



def build_model(num_classes):
    """Xây dựng CNN tự thiết kế để phân loại bệnh lá lúa."""

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

    # Tăng cường dữ liệu chỉ trong quá trình huấn luyện
    x = augmentation(inputs)

    # Chuẩn hóa pixel từ [0, 255] về [0, 1]
    x = layers.Rescaling(1.0 / 255)(x)

    # Khối CNN 1
    x = layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    # Khối CNN 2
    x = layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Khối CNN 3
    x = layers.Conv2D(
        128, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        128, 3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    # Khối CNN 4
    x = layers.Conv2D(
        256, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Phân loại
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="RiceLeaf_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_history(history):
    """Lưu biểu đồ accuracy và loss."""

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation",
    )
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation",
    )
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds, class_names = load_datasets()

    print("\nCác lớp được phát hiện:")
    for index, name in enumerate(class_names):
        print(f"{index}: {name}")

    # Tính trọng số để giảm ảnh hưởng khi các lớp mất cân bằng
    labels = np.concatenate(
        [labels.numpy() for _, labels in train_ds.unbatch().batch(4096)]
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels,
    )
    class_weights = {
        i: float(weight) for i, weight in enumerate(weights)
    }

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Lưu mô hình cuối cùng
    model.save(MODEL_PATH)

    # Lưu biểu đồ
    plot_history(history)

    # Lưu lịch sử huấn luyện dạng JSON
    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history_data, f, indent=4)

    print("\nHuấn luyện hoàn tất!")
    print(f"Mô hình cuối: {MODEL_PATH}")
    print(f"Mô hình tốt nhất: {BEST_MODEL_PATH}")
    print(f"Danh sách lớp: {CLASS_NAMES_PATH}")
    print(f"Biểu đồ: {OUTPUT_DIR / 'training_history.png'}")


if __name__ == "__main__":
    main()
