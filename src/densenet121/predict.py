"""
Dự đoán bệnh cho một ảnh hoặc cả thư mục ảnh.

    python -m src.densenet121.predict duong/dan/anh.jpg
    python -m src.densenet121.predict duong/dan/thu_muc     (kết quả lưu outputs/reports/predictions.csv)
"""

import csv
import sys
from pathlib import Path

import keras
import numpy as np

from src.densenet121.config import FINAL_MODEL_PATH, CLASS_NAMES_PATH, REPORT_DIR, SUPPORTED_EXTENSIONS
from src.densenet121.data_loader import load_json
from src.densenet121.image_utils import load_image_array, to_model_input

# Tên hiển thị tiếng Việt. Khoá đã chuẩn hoá (chữ thường, bỏ dấu cách/_/-)
# nên khớp với "Leaf scald", "Leaf_scald", "leaf-scald"...
VIETNAMESE_NAMES = {
    "bacterialleafblight": "Bac la (bacterial leaf blight)",
    "brownspot": "Dom nau (brown spot)",
    "healthyriceleaf": "La khoe manh (healthy rice leaf)",
    "leafblast": "Dao on la (leaf blast)",
    "leafscald": "Chay bia la (leaf scald)",
    "sheathblight": "Kho van (sheath blight)",
}


def display_name(class_name):
    key = "".join(ch for ch in class_name.lower() if ch.isalnum())
    return VIETNAMESE_NAMES.get(key, class_name)


def load_trained_model():
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Chua co mo hinh {FINAL_MODEL_PATH}. Hay chay: python -m src.densenet121.train")
    return keras.models.load_model(FINAL_MODEL_PATH)


def predict_image(model, class_names, image_path):
    probs = model.predict(to_model_input(load_image_array(image_path)), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return [(class_names[i], float(probs[i])) for i in order]


def show(image_path, ranking):
    name, prob = ranking[0]
    print(f"\nAnh: {image_path}")
    print(f"  Ket qua : {display_name(name)}  ({prob * 100:.2f}%)")
    print("  Top 3   : " + " | ".join(f"{n} {p * 100:.1f}%" for n, p in ranking[:3]))
    if prob < 0.5:
        print("  Luu y: do tin cay thap, nen kiem tra lai bang mat.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = Path(sys.argv[1])
    model = load_trained_model()
    class_names = load_json(CLASS_NAMES_PATH)

    if target.is_file():
        show(target, predict_image(model, class_names, target))
        return

    if not target.is_dir():
        raise FileNotFoundError(f"Khong tim thay: {target}")

    images = sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["anh", "du_doan", "do_tin_cay_%", "top2", "top2_%"])
        for image_path in images:
            ranking = predict_image(model, class_names, image_path)
            show(image_path, ranking)
            writer.writerow([
                str(image_path), ranking[0][0], round(ranking[0][1] * 100, 2),
                ranking[1][0], round(ranking[1][1] * 100, 2),
            ])

    print(f"\nDa du doan {len(images)} anh. Ket qua: {csv_path}")


if __name__ == "__main__":
    main()
