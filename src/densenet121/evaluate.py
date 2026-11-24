"""
Đánh giá mô hình trên tập TEST (ảnh mô hình chưa từng thấy).

    python -m src.densenet121.evaluate
"""

import json
import time

import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from src.densenet121.config import FINAL_MODEL_PATH, TEST_MANIFEST, CLASS_NAMES_PATH, REPORT_DIR, PLOT_DIR
from src.densenet121.data_loader import load_json, create_dataset


def plot_confusion_matrix(cm, class_names, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=40, ha="right")
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("Du doan")
    ax.set_ylabel("Thuc te")
    ax.set_title("Confusion matrix - tap test (so anh / % theo hang)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{cm[i, j]}\n{cm_norm[i, j] * 100:.0f}%",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, class_names, path):
    """ROC một-với-phần-còn-lại cho từng lớp."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == index, y_prob[:, index])
        auc = roc_auc_score(y_true == index, y_prob[:, index])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - tap test")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# [THEM DE SO SANH] đo thời gian dự đoán 1 ảnh - không ảnh hưởng mô hình
def measure_latency(model, image_shape, repeats=30):
    x = np.random.uniform(0, 255, size=(1, *image_shape)).astype("float32")
    for _ in range(5):
        model(x, training=False)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x, training=False)
    return (time.perf_counter() - start) / repeats * 1000


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DANH GIA DENSENET121 TREN TAP TEST")
    print("=" * 60)

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Chua co mo hinh {FINAL_MODEL_PATH}. Hay chay: python -m src.densenet121.train")

    class_names = load_json(CLASS_NAMES_PATH)
    records = load_json(TEST_MANIFEST)
    test_ds = create_dataset(records, len(class_names), training=False)
    print("So anh test:", len(records))

    model = keras.models.load_model(FINAL_MODEL_PATH)
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    y_true = np.array([r["label"] for r in records])

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    macro_f1 = report_dict["macro avg"]["f1-score"]

    print("\n" + "=" * 60)
    print(f"Test Accuracy: {test_acc * 100:.2f}% ({int((y_pred == y_true).sum())}/{len(y_true)} anh)")
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"MCC          : {mcc:.4f}")
    print(f"ROC-AUC (macro, one-vs-rest): {roc_auc:.4f}")
    print("=" * 60)
    print("\nCLASSIFICATION REPORT:\n" + report_text)
    print("CONFUSION MATRIX (hang = thuc te, cot = du doan):")
    print(cm)

    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\nTest Loss: {test_loss:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\nMCC: {mcc:.4f}\nROC-AUC (macro, OvR): {roc_auc:.4f}\n\n")
        f.write(report_text)
        f.write("\nConfusion matrix (hang = thuc te, cot = du doan):\n")
        f.write(np.array2string(cm))
    with open(REPORT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        report_dict["mcc"] = mcc
        report_dict["roc_auc_macro_ovr"] = roc_auc
        # [THEM DE SO SANH]
        report_dict["test_loss"] = test_loss
        report_dict["latency_ms"] = measure_latency(model, model.input_shape[1:])
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    plot_confusion_matrix(cm, class_names, PLOT_DIR / "confusion_matrix.png")
    plot_roc_curves(y_true, y_prob, class_names, PLOT_DIR / "roc_curves.png")

    # [THEM DE SO SANH] xác suất từng ảnh test -> src/compare_models.py (McNemar)
    np.savez(
        REPORT_DIR / "test_predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
        paths=np.array([r["path"] for r in records]),
    )

    print("\nDa luu:")
    print(" ", REPORT_DIR / "classification_report.txt")
    print(" ", PLOT_DIR / "confusion_matrix.png")
    print(" ", PLOT_DIR / "roc_curves.png")


if __name__ == "__main__":
    main()
