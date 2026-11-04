
import csv
import json
import numpy as np

from pathlib import Path
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import (
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    OUTPUT_DIR,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = REPORT_DIR / "batch_predictions.csv"


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0])
    )

    image = np.asarray(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image


def predict_folder(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc: {folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Duong dan khong phai thu muc: {folder_path}"
        )

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {FINAL_MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay danh sach lop: {CLASS_NAMES_PATH}"
        )

    image_paths = [
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print("Khong tim thay anh trong thu muc!")
        return

    print("=" * 60)
    print("DU DOAN HANG LOAT BENH LA LUA")
    print("=" * 60)

    print(f"So luong anh tim thay: {len(image_paths)}")

    class_names = load_class_names()

    print("\nDang nap mo hinh...")
    model = keras.models.load_model(FINAL_MODEL_PATH)

    print("Da nap mo hinh thanh cong!\n")

    results = []
    success_count = 0
    error_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        try:
            image = preprocess_image(image_path)

            probabilities = model.predict(
                image,
                verbose=0
            )[0]

            best_index = int(np.argmax(probabilities))

            predicted_class = class_names[best_index]
            confidence = float(probabilities[best_index]) * 100

            results.append([
                str(image_path),
                predicted_class,
                f"{confidence:.2f}%"
            ])

            success_count += 1

            print(
                f"{index}/{len(image_paths)} | "
                f"{image_path.name} | "
                f"{predicted_class} | "
                f"{confidence:.2f}%"
            )

        except Exception as error:
            error_count += 1

            print(
                f"{index}/{len(image_paths)} | "
                f"LOI: {image_path.name} | {error}"
            )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Image Path",
            "Predicted Class",
            "Confidence"
        ])

        writer.writerows(results)

    print("\n" + "=" * 60)
    print("KET QUA TONG KET")
    print("=" * 60)

    print(f"Tong so anh: {len(image_paths)}")
    print(f"So anh du doan thanh cong: {success_count}")
    print(f"So anh bi loi: {error_count}")

    print(f"\nDa luu ket qua tai:")
    print(CSV_PATH)


if __name__ == "__main__":

    folder_path = input(
        "Nhap duong dan thu muc chua anh la lua: "
    ).strip().strip('"')

    try:
        predict_folder(folder_path)

    except Exception as error:
        print("\nLOI KHI DU DOAN:")
        print(error)
