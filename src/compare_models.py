"""
So sánh MobileNetV2 và DenseNet121 trên CÙNG bộ dữ liệu, CÙNG tập test.

Chạy SAU KHI đã chuẩn bị dữ liệu, train và evaluate cả 2 mô hình:
    python -m src.mobilenetv2.prepare_data
    python -m src.densenet121.prepare_data
    python -m src.mobilenetv2.train
    python -m src.mobilenetv2.evaluate
    python -m src.densenet121.train
    python -m src.densenet121.evaluate
    python -m src.compare_models

Hai mô hình có thư mục, cấu hình và xử lý dữ liệu RIÊNG (src/mobilenetv2/,
src/densenet121/). File này chỉ ĐỌC kết quả của chúng, kiểm tra 2 mô hình có
cùng danh sách lớp và CÙNG tập test, rồi so sánh.

Kết quả trong outputs/comparison/:
    comparison_report.md     báo cáo tiếng Việt (bảng + kiểm định), dán vào báo cáo
    comparison_table.csv     bảng chỉ số (mở bằng Excel)
    per_class_f1.png         F1 từng lớp của 2 mô hình
    val_accuracy_curves.png  val accuracy theo epoch
    confusion_matrices.png   2 confusion matrix cạnh nhau
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mobilenetv2 import config as mobilenetv2_config
from src.densenet121 import config as densenet121_config

CONFIGS = {"mobilenetv2": mobilenetv2_config, "densenet121": densenet121_config}
MODEL_NAMES = tuple(CONFIGS)
NAMES = {"mobilenetv2": "MobileNetV2", "densenet121": "DenseNet121"}
DATASET = mobilenetv2_config.DATASET
COMPARISON_DIR = mobilenetv2_config.ROOT_DIR / "outputs" / "comparison"

# Màu theo bảng màu đã kiểm tra (an toàn cho người mù màu); màu gắn với
# MÔ HÌNH, không đổi theo thứ hạng.
COLORS = {"mobilenetv2": "#2a78d6", "densenet121": "#eb6834"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------------------
# Đọc kết quả
# ---------------------------------------------------------------------------

def load_json_if_exists(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_results(name):
    cfg = CONFIGS[name]
    report = load_json_if_exists(cfg.REPORT_DIR / "classification_report.json")
    predictions = cfg.REPORT_DIR / "test_predictions.npz"
    if report is None or not predictions.exists():
        return None
    data = np.load(predictions, allow_pickle=False)
    history = {}
    for stage in ("stage1", "stage2"):
        h = load_json_if_exists(cfg.LOG_DIR / f"{stage}_history.json")
        if h:
            history[stage] = h
    return {
        "report": report,
        "summary": load_json_if_exists(cfg.OUTPUT_DIR / "training_summary.json") or {},
        "class_names": load_json_if_exists(cfg.CLASS_NAMES_PATH),
        "y_true": data["y_true"],
        "y_prob": data["y_prob"],
        "paths": data["paths"],
        "history": history,
    }


def normalize_path(path):
    """C:/.../data/raw/x.jpg hoặc data/raw/x.jpg -> data/raw/x.jpg (để so khớp 2 mô hình)."""
    path = Path(str(path).replace("\\", "/"))
    root = Path(str(mobilenetv2_config.ROOT_DIR).replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return path.as_posix()


def restrict_to_common_test(results, class_names):
    """
    2 mô hình có prepare_data riêng nên tập test có thể khác nhau. Chỉ giữ các ảnh
    nằm trong tập test của CẢ 2 mô hình (không mô hình nào thấy khi train) và
    tính lại mọi chỉ số trên tập chung đó.
    """
    from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score

    index = {}
    for name, r in results.items():
        index[name] = {normalize_path(p): i for i, p in enumerate(r["paths"])}
    first = next(iter(results))
    common = [p for p in index[first] if all(p in index[n] for n in results)]
    if not common:
        raise SystemExit("2 mo hinh KHONG co anh test chung nao -> khong the so sanh.")

    labels = list(range(len(class_names)))
    for name, r in results.items():
        rows = np.array([index[name][p] for p in common])
        r["full_test_size"] = len(r["y_true"])
        r["y_true"] = r["y_true"][rows]
        r["y_prob"] = r["y_prob"][rows]
        r["paths"] = np.array(common)
        y_pred = r["y_prob"].argmax(axis=1)
        report = classification_report(
            r["y_true"], y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        report["mcc"] = float(matthews_corrcoef(r["y_true"], y_pred))
        try:
            report["roc_auc_macro_ovr"] = float(roc_auc_score(
                r["y_true"], r["y_prob"], multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            report["roc_auc_macro_ovr"] = float("nan")
        prob_true = np.clip(r["y_prob"][np.arange(len(rows)), r["y_true"]], 1e-7, 1.0)
        report["test_loss"] = float(-np.log(prob_true).mean())
        report["latency_ms"] = r["report"].get("latency_ms")
        r["report"] = report

    first_true = results[first]["y_true"]
    for name, r in results.items():
        if not np.array_equal(r["y_true"], first_true):
            raise SystemExit(f"Nhan cua anh test chung khac nhau giua {first} va {name}.")
    return len(common)


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------

def wilson_interval(correct, total, z=1.96):
    """Khoảng tin cậy 95% của accuracy (Wilson)."""
    p = correct / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return centre - half, centre + half


def mcnemar_exact(correct_a, correct_b):
    """
    Kiểm định McNemar (chính xác, nhị thức) trên CÙNG tập test.
    b = A đúng, B sai; c = A sai, B đúng. p < 0.05 -> khác biệt có ý nghĩa thống kê.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Biểu đồ
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_per_class_f1(results, class_names, path):
    backbones = list(results)
    x = np.arange(len(class_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    style_axes(ax)
    for i, b in enumerate(backbones):
        f1 = [results[b]["report"][c]["f1-score"] for c in class_names]
        offset = (i - (len(backbones) - 1) / 2) * (width + 0.02)
        ax.bar(x + offset, f1, width, color=COLORS[b], label=NAMES[b],
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [c.replace("_", " ") for c in class_names], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score", color=INK_2)
    ax.set_title("F1-score từng lớp trên tập test", color=INK, loc="left", fontsize=12)
    # Chú thích đặt phía trên vùng vẽ để không đè lên cột
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(backbones))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_val_curves(results, path):
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    style_axes(ax)
    for b, r in results.items():
        h1 = r["history"].get("stage1", {}).get("val_accuracy", [])
        h2 = r["history"].get("stage2", {}).get("val_accuracy", [])
        values = [v * 100 for v in h1 + h2]
        if not values:
            continue
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, color=COLORS[b], linewidth=2, label=NAMES[b])
        if h1:
            ax.axvline(len(h1) + 0.5, color=COLORS[b], linestyle=":", linewidth=1)
        ax.annotate(f"{NAMES[b]} {values[-1]:.1f}%", (epochs[-1], values[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xlabel("Epoch (đường chấm: bắt đầu fine-tuning)", color=INK_2)
    ax.set_ylabel("Validation accuracy (%)", color=INK_2)
    ax.set_title("Validation accuracy theo epoch", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right")
    longest = max(
        len(r["history"].get("stage1", {}).get("val_accuracy", []))
        + len(r["history"].get("stage2", {}).get("val_accuracy", []))
        for r in results.values()
    )
    ax.set_xlim(0.5, longest * 1.15 + 1)   # chừa chỗ cho nhãn cuối đường
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrices(results, class_names, path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    labels = [c.replace("_", " ") for c in class_names]
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (b, r) in zip(axes, results.items()):
        y_pred = r["y_prob"].argmax(axis=1)
        k = len(class_names)
        cm = np.zeros((k, k), dtype=int)
        for t, p in zip(r["y_true"], y_pred):
            cm[t, p] += 1
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        ticks = np.arange(k)
        ax.set_xticks(ticks, labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(ticks, labels, fontsize=8)
        ax.set_xlabel("Dự đoán", color=INK_2)
        ax.set_ylabel("Thực tế", color=INK_2)
        ax.set_title(NAMES[b], color=INK, loc="left")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center",
                        fontsize=7, color="#ffffff" if norm[i, j] > 0.55 else INK)
    fig.suptitle("Confusion matrix trên tập test (số ảnh / % theo hàng)", color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------

def fmt(value, pattern, missing="—"):
    return missing if value is None else pattern.format(value)


def main():
    # Terminal Windows mặc định cp1252 -> lỗi khi in tiếng Việt
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for b in MODEL_NAMES:
        r = load_results(b)
        if r is None:
            print(f"Chua co ket qua cua {NAMES[b]}. Hay chay:\n"
                  f"  python -m src.{b}.train\n  python -m src.{b}.evaluate")
        else:
            results[b] = r
    if len(results) < 2:
        raise SystemExit("Can ket qua evaluate cua CA 2 mo hinh de so sanh.")

    a, b = list(results)
    class_names = results[a]["class_names"]
    if class_names != results[b]["class_names"]:
        raise SystemExit(
            f"2 mo hinh co danh sach lop khac nhau:\n  {a}: {class_names}\n"
            f"  {b}: {results[b]['class_names']}"
        )
    common_size = restrict_to_common_test(results, class_names)
    full_sizes = {x: results[x]["full_test_size"] for x in results}
    if any(n != common_size for n in full_sizes.values()):
        print(f"Canh bao: 2 mo hinh co tap test khac nhau ({full_sizes}). "
              f"Chi so sanh tren {common_size} anh test CHUNG.")

    # ---------------- Bảng chỉ số ----------------
    rows = []
    for name, r in results.items():
        rep, s = r["report"], r["summary"]
        y_pred = r["y_prob"].argmax(axis=1)
        correct = int((y_pred == r["y_true"]).sum())
        total = len(r["y_true"])
        low, high = wilson_interval(correct, total)
        rows.append({
            "Mô hình": NAMES[name],
            "Accuracy (%)": round(rep["accuracy"] * 100, 2),
            "KTC 95% accuracy (%)": f"{low * 100:.1f}–{high * 100:.1f}",
            "Precision macro": round(rep["macro avg"]["precision"], 4),
            "Recall macro": round(rep["macro avg"]["recall"], 4),
            "F1 macro": round(rep["macro avg"]["f1-score"], 4),
            "MCC": round(rep["mcc"], 4),
            "ROC-AUC macro": round(rep["roc_auc_macro_ovr"], 4),
            "Test loss": round(rep.get("test_loss", float("nan")), 4),
            "Số tham số (triệu)": fmt(s.get("total_params") and s["total_params"] / 1e6, "{:.2f}"),
            "Dung lượng mô hình (MB)": fmt(s.get("model_size_mb"), "{:.1f}"),
            "Thời gian train (phút)": fmt(s.get("training_minutes"), "{:.1f}"),
            "Thời gian dự đoán 1 ảnh (ms, CPU)": fmt(rep.get("latency_ms"), "{:.1f}"),
        })

    with open(COMPARISON_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- McNemar ----------------
    correct_a = results[a]["y_prob"].argmax(1) == results[a]["y_true"]
    correct_b = results[b]["y_prob"].argmax(1) == results[b]["y_true"]
    only_a, only_b, p_value = mcnemar_exact(correct_a, correct_b)

    # ---------------- Biểu đồ ----------------
    plot_per_class_f1(results, class_names, COMPARISON_DIR / "per_class_f1.png")
    plot_val_curves(results, COMPARISON_DIR / "val_accuracy_curves.png")
    plot_confusion_matrices(results, class_names, COMPARISON_DIR / "confusion_matrices.png")

    # ---------------- Báo cáo Markdown ----------------
    lines = [f"# So sánh {NAMES[a]} và {NAMES[b]} - bộ dữ liệu {DATASET}", ""]
    if all(n == common_size for n in full_sizes.values()):
        lines.append(f"Tập test: {common_size} ảnh, giống nhau cho cả 2 mô hình.")
    else:
        lines.append(
            f"Tập test: {common_size} ảnh CHUNG của 2 mô hình (tập test riêng: "
            + ", ".join(f"{NAMES[x]} {n} ảnh" for x, n in full_sizes.items())
            + "). Mọi chỉ số được tính lại trên tập chung này."
        )
    lines.append("")
    keys = list(rows[0])
    lines.append("| Chỉ số | " + " | ".join(r["Mô hình"] for r in rows) + " |")
    lines.append("|---|" + "---|" * len(rows))
    for key in keys[1:]:
        lines.append(f"| {key} | " + " | ".join(str(r[key]) for r in rows) + " |")

    lines += ["", "## F1-score và recall từng lớp", ""]
    lines.append("| Lớp | " + " | ".join(f"F1 {NAMES[x]}" for x in results)
                 + " | " + " | ".join(f"Recall {NAMES[x]}" for x in results) + " |")
    lines.append("|---|" + "---|" * (2 * len(results)))
    for c in class_names:
        f1s = [f"{results[x]['report'][c]['f1-score']:.4f}" for x in results]
        recs = [f"{results[x]['report'][c]['recall']:.4f}" for x in results]
        lines.append(f"| {c} | " + " | ".join(f1s) + " | " + " | ".join(recs) + " |")

    verdict = (
        "khác biệt **có ý nghĩa thống kê** (p < 0.05)"
        if p_value < 0.05 else
        "khác biệt **không có ý nghĩa thống kê** (p ≥ 0.05): trên tập test này "
        "chưa đủ bằng chứng để kết luận mô hình nào chính xác hơn"
    )
    lines += [
        "", "## Kiểm định McNemar (cùng tập test)", "",
        f"- Ảnh chỉ {NAMES[a]} đúng ({NAMES[b]} sai): **{only_a}**",
        f"- Ảnh chỉ {NAMES[b]} đúng ({NAMES[a]} sai): **{only_b}**",
        f"- p-value (McNemar chính xác): **{p_value:.4f}** → {verdict}.",
        "",
        "## Biểu đồ", "",
        "- `per_class_f1.png`: F1-score từng lớp",
        "- `val_accuracy_curves.png`: validation accuracy theo epoch",
        "- `confusion_matrices.png`: confusion matrix của 2 mô hình",
    ]
    if any(not results[x]["summary"] for x in results):
        lines += ["", "*Ghi chú: mô hình thiếu `training_summary.json` (train bằng code cũ) "
                      "nên không có số tham số / thời gian train. Train lại để có đủ số liệu.*"]

    report_path = COMPARISON_DIR / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ---------------- In ra màn hình ----------------
    print("=" * 70)
    print(f"SO SANH {NAMES[a].upper()} VA {NAMES[b].upper()} - BO DU LIEU {DATASET}")
    print("=" * 70)
    width = max(len(k) for k in keys)
    print(f"{'':<{width}}  " + "  ".join(f"{r['Mô hình']:>14}" for r in rows))
    for key in keys[1:]:
        print(f"{key:<{width}}  " + "  ".join(f"{str(r[key]):>14}" for r in rows))
    print("-" * 70)
    print(f"McNemar: chi {NAMES[a]} dung = {only_a}, chi {NAMES[b]} dung = {only_b}, "
          f"p = {p_value:.4f} -> {'CO' if p_value < 0.05 else 'KHONG'} khac biet y nghia thong ke")
    print("\nDa luu vao:", COMPARISON_DIR)


if __name__ == "__main__":
    main()
