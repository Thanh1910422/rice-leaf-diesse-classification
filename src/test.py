
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
)


def load_validation_dataset():
    """Đọc đúng tập validation đã lưu khi chia dữ liệu huấn luyện."""

    validation_file = OUTPUT_DIR / "validation_data.json"

    if not validation_file.exists():
        raise FileNotFoundError(
            "Không tìm thấy validation_data.json. "
            "Hãy chạy lại train.py để tạo tập validation mới."
        )

    with open(validation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_names = data["class_names"]
    paths = data["paths"]
    labels = np.array(data["labels"], dtype=np.int32)

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

    val_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    val_ds = val_ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return val_ds, class_names


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có mô hình tốt nhất. Hãy chạy train.py trước."
        )

    val_ds, class_names = load_validation_dataset()

    model = keras.models.load_model(BEST_MODEL_PATH)

    loss, accuracy = model.evaluate(val_ds, verbose=1)

    y_true = np.concatenate(
        [labels.numpy() for _, labels in val_ds],
        axis=0,
    )

    probabilities = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4%}")

    print("\n===== SỐ ẢNH THỰC TẾ THEO LỚP =====")
    for i, name in enumerate(class_names):
        count = int(np.sum(y_true == i))
        print(f"{name}: {count} ảnh")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(
        figsize=(max(9, len(class_names) * 1.5), 8)
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    plt.title("Confusion Matrix - Rice Leaf Classification")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nĐã lưu báo cáo tại: {OUTPUT_DIR}")
    print(" - classification_report.json")
    print(" - confusion_matrix.png")


if __name__ == "__main__":
    main()
