
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from config import (
    IMAGE_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
)


def load_image(image_path):
    """Đọc và chuẩn hóa ảnh đầu vào."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh: {image_path}"
        )

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            Image.Resampling.BILINEAR,
        )
        image = np.asarray(img, dtype=np.float32)

    # Thêm chiều batch: (1, height, width, channels)
    return np.expand_dims(image, axis=0)


def predict_image(image_path, top_k=3):
    """Dự đoán lớp của ảnh lá lúa."""

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có mô hình. Hãy chạy python src/train.py trước."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Không tìm thấy class_names.json."
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = keras.models.load_model(BEST_MODEL_PATH)
    image = load_image(image_path)

    probabilities = model.predict(image, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Số đầu ra của mô hình không khớp số lớp."
        )

    top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    print("\n===== KẾT QUẢ DỰ ĐOÁN =====")
    print(f"Ảnh: {image_path}")

    for rank, idx in enumerate(top_indices, start=1):
        print(
            f"{rank}. {class_names[idx]}: "
            f"{probabilities[idx] * 100:.2f}%"
        )

    predicted_index = int(top_indices[0])
    confidence = float(probabilities[predicted_index])

    print("\nKết luận của mô hình:")
    print(f"Lớp dự đoán: {class_names[predicted_index]}")
    print(f"Độ tin cậy: {confidence * 100:.2f}%")

    if confidence < 0.60:
        print(
            "Cảnh báo: độ tin cậy thấp. "
            "Nên kiểm tra ảnh và đối chiếu thêm."
        )

    return class_names[predicted_index], confidence


def main():
    parser = argparse.ArgumentParser(
        description="Phân loại bệnh trên lá lúa."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Đường dẫn đến ảnh lá lúa cần dự đoán.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Số lớp có xác suất cao nhất cần hiển thị.",
    )

    args = parser.parse_args()
    predict_image(args.image, args.top_k)


if __name__ == "__main__":
    main()
