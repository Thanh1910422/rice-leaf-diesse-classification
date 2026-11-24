"""
Grad-CAM: tô màu vùng ảnh mà MobileNetV2 dựa vào để ra quyết định.
Màu đỏ = vùng ảnh hưởng mạnh. Mô hình tốt phải "nhìn" vào vết bệnh,
không phải nền ruộng / bầu trời / tay người cầm lá.

    python -m src.mobilenetv2.gradcam                  # 3 ảnh test mỗi lớp -> outputs/gradcam/test_samples.png
    python -m src.mobilenetv2.gradcam duong/dan/anh.jpg
    python -m src.mobilenetv2.gradcam duong/dan/thu_muc
"""

import random
import sys
from pathlib import Path

import keras
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.mobilenetv2.config import (
    FINAL_MODEL_PATH, TEST_MANIFEST, CLASS_NAMES_PATH, GRADCAM_DIR,
    SUPPORTED_EXTENSIONS, SEED,
)
from src.mobilenetv2.data_loader import load_json, resolve_path
from src.mobilenetv2.image_utils import load_image_array
from src.mobilenetv2.model import find_backbone


def head_layers(model, backbone):
    """Các lớp nằm SAU MobileNetV2: GAP -> BatchNorm -> Dropout -> Dense(256) -> Dropout -> softmax."""
    index = model.layers.index(backbone)
    return model.layers[index + 1:]


def compute_gradcam(model, image):
    """
    image: uint8 (224, 224, 3). Trả về (heatmap 224x224 trong [0,1], xác suất).
    Tính trực tiếp qua các lớp của mô hình (bỏ qua augmentation vì khi dự đoán
    nó không làm gì; Dropout/BatchNorm chạy ở chế độ suy luận).
    """
    backbone = find_backbone(model)

    # Giống bước preprocess_input bên trong model.py: [0, 255] -> [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(image[None].astype("float32"))
    conv = backbone(tf.convert_to_tensor(x), training=False)   # (1, 7, 7, 1280)

    with tf.GradientTape() as tape:
        tape.watch(conv)
        probs = conv
        for layer in head_layers(model, backbone):
            probs = layer(probs, training=False)
        class_index = int(tf.argmax(probs[0]))
        score = probs[:, class_index]

    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)

    heatmap = np.asarray(
        Image.fromarray(np.uint8(cam * 255)).resize(image.shape[1::-1], Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return heatmap, np.asarray(probs[0])


def overlay(image, heatmap, alpha=0.4):
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3] * 255
    return np.uint8((1 - alpha) * image + alpha * colored)


def draw_rows(rows, class_names, path):
    """rows: list (image, heatmap, probs, nhan_that hoặc None, tieu_de)."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)), squeeze=False)
    for (image, heatmap, probs, true_label, title), (ax1, ax2) in zip(rows, axes):
        pred = int(np.argmax(probs))
        ax1.imshow(image)
        ax1.set_title(title, fontsize=8)
        ax2.imshow(overlay(image, heatmap))
        text = f"Du doan: {class_names[pred]} ({probs[pred] * 100:.1f}%)"
        if true_label is not None:
            text += f"\nThat: {class_names[true_label]}" + (" - DUNG" if pred == true_label else " - SAI")
        ax2.set_title(text, fontsize=8, color="black" if true_label in (None, pred) else "red")
        ax1.axis("off")
        ax2.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("Da luu:", path)


def main():
    model = keras.models.load_model(FINAL_MODEL_PATH)
    class_names = load_json(CLASS_NAMES_PATH)

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        files = [target] if target.is_file() else sorted(
            p for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for file in files:
            image = load_image_array(file)
            heatmap, probs = compute_gradcam(model, image)
            draw_rows([(image, heatmap, probs, None, file.name)], class_names,
                      GRADCAM_DIR / f"{file.stem}_gradcam.png")
        return

    # Mặc định: 3 ảnh test ngẫu nhiên mỗi lớp
    records = load_json(TEST_MANIFEST)
    rng = random.Random(SEED)
    rows = []
    for label in range(len(class_names)):
        chosen = [r for r in records if r["label"] == label]
        for record in rng.sample(chosen, min(3, len(chosen))):
            image = load_image_array(resolve_path(record["path"]))
            heatmap, probs = compute_gradcam(model, image)
            rows.append((image, heatmap, probs, label, Path(record["path"]).name))
    draw_rows(rows, class_names, GRADCAM_DIR / "test_samples.png")


if __name__ == "__main__":
    main()
