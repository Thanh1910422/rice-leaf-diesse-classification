
import sys
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from tensorflow import keras

from src.config import (
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
)


def load_class_names():
    """Doc ten 6 lop tu file JSON."""

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path):
    """Doc va tien xu ly anh dau vao."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay anh: {image_path}"
        )

    # Doc anh va chuyen sang RGB
    image = Image.open(image_path).convert("RGB")

    # Resize anh
    image = image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0])
    )

    # Chuyen thanh mang numpy
    image = np.asarray(image, dtype=np.float32)

    # Them chieu batch
    image = np.expand_dims(image, axis=0)

    # Tien xu ly dung MobileNetV2
    # Không preprocess tại đây: model.py đã tích hợp preprocess_input.
    return image


def predict_image(image_path):
    """Du doan benh la lua tu mot anh moi."""

    print("=" * 60)
    print("DU DOAN BENH LA LUA BANG CNN MOBILENETV2")
    print("=" * 60)

    # Kiem tra mo hinh
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {FINAL_MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay danh sach lop: {CLASS_NAMES_PATH}"
        )

    # Nap ten lop va mo hinh
    class_names = load_class_names()
    model = keras.models.load_model(FINAL_MODEL_PATH)

    print("\nDa nap mo hinh thanh cong!")
    print("Anh dau vao:", image_path)

    # Tien xu ly anh
    image = preprocess_image(image_path)

    # Du doan
    probabilities = model.predict(image, verbose=0)[0]

    # Sap xep xac suat giam dan
    sorted_indices = np.argsort(probabilities)[::-1]

    best_index = int(sorted_indices[0])
    predicted_class = class_names[best_index]
    confidence = float(probabilities[best_index])

    # Hien thi ket qua
    print("\n" + "=" * 60)
    print("KET QUA DU DOAN")
    print("=" * 60)

    print("Lop du doan:", predicted_class)
    print(f"Do tin cay: {confidence * 100:.2f}%")

    print("\nTOP 3 KET QUA:")

    for rank, index in enumerate(sorted_indices[:3], start=1):
        class_name = class_names[int(index)]
        probability = float(probabilities[index])

        print(
            f"{rank}. {class_name}: "
            f"{probability * 100:.2f}%"
        )

    print("\nXAC SUAT CUA TUNG LOP:")

    for index, class_name in enumerate(class_names):
        print(
            f"{class_name}: "
            f"{probabilities[index] * 100:.2f}%"
        )

    print("\nHOAN THANH DU DOAN!")

    return predicted_class, confidence


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input(
            "Nhap duong dan anh la lua can du doan: "
        ).strip().strip('"')

    try:
        predict_image(image_path)

    except Exception as error:
        print("\nLOI KHI DU DOAN:")
        print(error)
