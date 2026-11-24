"""
Xuất các ảnh bị đoán SAI để kiểm tra bằng mắt (phát hiện ảnh gán nhãn sai).

    python -m src.densenet121.export_misclassified          # tập test
    python -m src.densenet121.export_misclassified val      # tập validation

Kết quả: outputs/misclassified/<tap>/<lop_that>__thanh__<lop_du_doan>/<tin_cay>pct__<ten_anh>
         outputs/misclassified/<tap>/misclassified.csv
"""

import csv
import shutil
import sys
from pathlib import Path

import keras
import numpy as np

from src.densenet121.config import (
    FINAL_MODEL_PATH, TEST_MANIFEST, VAL_MANIFEST, CLASS_NAMES_PATH, MISCLASSIFIED_DIR,
)
from src.densenet121.data_loader import load_json, resolve_path, create_dataset


def main(split="test"):
    manifest = {"test": TEST_MANIFEST, "val": VAL_MANIFEST}[split]
    records = load_json(manifest)
    class_names = load_json(CLASS_NAMES_PATH)

    model = keras.models.load_model(FINAL_MODEL_PATH)
    dataset = create_dataset(records, len(class_names), training=False)  # không shuffle
    probs = model.predict(dataset, verbose=0)

    out_dir = MISCLASSIFIED_DIR / split
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rows = []
    for record, p in zip(records, probs):
        pred = int(np.argmax(p))
        if pred == record["label"]:
            continue
        src = Path(resolve_path(record["path"]))
        folder = out_dir / f"{class_names[record['label']]}__thanh__{class_names[pred]}"
        folder.mkdir(exist_ok=True)
        shutil.copy2(src, folder / f"{p[pred] * 100:05.1f}pct__{src.name}")
        rows.append({
            "anh": record["path"],
            "nhan_that": class_names[record["label"]],
            "du_doan": class_names[pred],
            "tin_cay_du_doan_%": round(float(p[pred]) * 100, 1),
            "xac_suat_nhan_that_%": round(float(p[record["label"]]) * 100, 1),
        })

    rows.sort(key=lambda r: -r["tin_cay_du_doan_%"])
    with open(out_dir / "misclassified.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "anh", "nhan_that", "du_doan", "tin_cay_du_doan_%", "xac_suat_nhan_that_%",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Tap {split}: {len(records)} anh, doan sai {len(rows)}")
    for r in rows:
        print(f"  {r['tin_cay_du_doan_%']:5.1f}%  {r['nhan_that']:>22} -> {r['du_doan']:<22} {r['anh']}")
    print("\nThu muc:", out_dir)
    print("Anh sai voi do tin cay > 80% rat dang nghi bi gan nhan sai.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
