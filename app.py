"""
Rice leaf disease app: show MobileNetV2 and DenseNet121 predictions side by side.
Run: streamlit run app.py
"""
import importlib
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

MODEL_LABELS = {"mobilenetv2": "MobileNetV2", "densenet121": "DenseNet121"}
VIETNAMESE_NAMES = {
    "bacterialleafblight": "Bạc lá (Bacterial Leaf Blight)",
    "brownspot": "Đốm nâu (Brown Spot)",
    "healthyriceleaf": "Lá khỏe mạnh (Healthy)",
    "leafblast": "Đạo ôn lá (Leaf Blast)",
    "leafscald": "Cháy bìa lá (Leaf Scald)",
    "sheathblight": "Khô vằn (Sheath Blight)",
}


def display_name(class_name):
    key = "".join(ch for ch in class_name.lower() if ch.isalnum())
    return VIETNAMESE_NAMES.get(key, class_name)


st.set_page_config(page_title="Rice Leaf Disease Classification",
                   page_icon="🌾", layout="wide")
st.markdown(
    '<div class="main-title">🌾 HỆ THỐNG DỰ ĐOÁN BỆNH TRÊN LÁ LÚA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'Tải lên một ảnh lá lúa để xem kết quả dự đoán của '
    '<b>MobileNetV2 và DenseNet121 song song</b>.'
    '</div>',
    unsafe_allow_html=True
)
st.divider()

AVAILABLE_MODELS = {}
MODEL_ERRORS = {}
for model_name in MODEL_LABELS:
    try:
        config = importlib.import_module(f"src.{model_name}.config")
        if config.FINAL_MODEL_PATH.exists():
            AVAILABLE_MODELS[model_name] = config
        else:
            MODEL_ERRORS[model_name] = f"Không tìm thấy mô hình: {config.FINAL_MODEL_PATH}"
    except Exception as error:
        MODEL_ERRORS[model_name] = str(error)


@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path)


@st.cache_data
def load_class_names(class_names_path):
    path = Path(class_names_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy danh sách lớp: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_for_model(model_name, config, image):
    image_utils = importlib.import_module(f"src.{model_name}.image_utils")
    model = load_model(str(config.FINAL_MODEL_PATH))
    class_names = load_class_names(str(config.CLASS_NAMES_PATH))

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Mô hình có {model.output_shape[-1]} lớp nhưng "
            f"class_names.json có {len(class_names)} lớp."
        )

    prepared = image_utils.prepare_pil_image(image)
    model_input = image_utils.to_model_input(prepared)
    probabilities = model.predict(model_input, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError("Số xác suất dự đoán không khớp số lớp.")

    order = np.argsort(probabilities)[::-1]
    best = int(order[0])
    return {
        "class_names": class_names,
        "probabilities": probabilities,
        "order": order,
        "prediction": class_names[best],
        "confidence": float(probabilities[best]) * 100,
        "input_shape": model_input.shape,
    }


def show_model_result(model_name, config, image):
    st.subheader(MODEL_LABELS[model_name])
    if config is None:
        st.warning("Mô hình chưa sẵn sàng hoặc chưa được train.")
        if model_name in MODEL_ERRORS:
            st.caption(MODEL_ERRORS[model_name])
        return

    try:
        with st.spinner(f"{MODEL_LABELS[model_name]} đang phân tích..."):
            result = predict_for_model(model_name, config, image)

        confidence = result["confidence"]
        result_text = (
            f"### {display_name(result['prediction'])}\n"
            f"Độ tin cậy: **{confidence:.2f}%**"
        )
        if confidence >= 80:
            st.success(result_text)
        elif confidence >= 50:
            st.warning(result_text + "\n\nĐộ tin cậy trung bình, nên kiểm tra lại.")
        else:
            st.error(result_text + "\n\nĐộ tin cậy thấp, kết quả chưa chắc chắn.")

        st.markdown("**Xác suất các lớp**")
        for rank, index in enumerate(result["order"], start=1):
            prob = float(result["probabilities"][index])
            label = display_name(result["class_names"][index])
            st.progress(prob, text=f"{rank}. {label}: {prob * 100:.2f}%")

        with st.expander("Thông tin tiền xử lý"):
            st.write("Shape đầu vào:", result["input_shape"])
            st.write("Kích thước ảnh gốc:", image.size)
    except Exception as error:
        st.error(f"Lỗi khi dự đoán bằng {MODEL_LABELS[model_name]}: {error}")


if not AVAILABLE_MODELS:
    st.error("Chưa tìm thấy mô hình đã train. Kiểm tra FINAL_MODEL_PATH trong config.py.")
    for name, message in MODEL_ERRORS.items():
        st.caption(f"{MODEL_LABELS[name]}: {message}")
    st.stop()

# Giao diện 3 cột theo sơ đồ: ảnh đầu vào | MobileNetV2 | DenseNet121
st.markdown(
    """
    <style>
    /* Cỡ chữ chung */
    html, body, [class*="st-"], .stMarkdown, .stText, label,
    [data-testid="stCaptionContainer"] {
        font-size: 1.25rem;
    }

    /* Tiêu đề các cột */
    .panel-heading {
        font-size: 1.5rem !important;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    /* Tiêu đề chính */
    .main-title {
        font-size: 2rem !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Dòng mô tả bên dưới */
    .main-description {
        font-size: 1.25rem !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Tiêu đề phụ */
    h2 {
        font-size: 2rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
    }

    /* Chữ trong khu vực tải ảnh */
    [data-testid="stFileUploader"] {
        font-size: 1.25rem;
    }

    [data-testid="stFileUploader"] section {
        font-size: 1.25rem;
    }

    /* Chữ trong nút bấm */
    .stButton button {
        font-size: 1.25rem !important;
        font-weight: 600;
    }

    /* Chữ trong thông báo kết quả */
    [data-testid="stAlert"] {
        font-size: 1.25rem !important;
    }

    /* Chữ trong thanh tiến trình */
    [data-testid="stProgress"] p {
        font-size: 1.5rem;
    }

    /* Giữ khoảng cách khu vực tải ảnh gọn */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stFileUploader"]) {
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

image_col, mobilenet_col, densenet_col = st.columns([1.0, 1.15, 1.15], gap="large")

with image_col:
    st.markdown('<div class="panel-heading">Ảnh tải lên</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Chọn ảnh lá lúa",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    image = None
    if uploaded_file is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            preview = image.copy()
            preview.thumbnail((520, 520))
            st.image(
                preview,
                caption=f"{uploaded_file.name}\n{image.width} × {image.height} px",
                use_container_width=True,
            )
            st.caption("Ảnh đầu vào được dùng chung cho cả hai mô hình.")
        except Exception as error:
            st.error(f"Không thể đọc ảnh tải lên: {error}")
            image = None
    else:
        st.info("Tải ảnh lá lúa lên để bắt đầu dự đoán.")

with mobilenet_col:
    st.markdown('<div class="panel-heading">MobileNetV2</div>', unsafe_allow_html=True)
    if image is not None:
        show_model_result("mobilenetv2", AVAILABLE_MODELS.get("mobilenetv2"), image)
    else:
        st.info("Kết quả MobileNetV2 sẽ hiển thị tại đây.")

with densenet_col:
    st.markdown('<div class="panel-heading">DenseNet121</div>', unsafe_allow_html=True)
    if image is not None:
        show_model_result("densenet121", AVAILABLE_MODELS.get("densenet121"), image)
    else:
        st.info("Kết quả DenseNet121 sẽ hiển thị tại đây.")

if image is not None:
    st.divider()
    st.subheader("Thông tin liên quan đến mô hình dự đoán")
    info_mobilenet, info_densenet = st.columns(2, gap="large")
    with info_mobilenet:
        st.markdown("**MobileNetV2**")
        cfg = AVAILABLE_MODELS.get("mobilenetv2")
        if cfg is not None:
            st.write("Đường dẫn mô hình:", str(cfg.FINAL_MODEL_PATH))
            st.write("Danh sách lớp:", str(cfg.CLASS_NAMES_PATH))
        else:
            st.caption(MODEL_ERRORS.get("mobilenetv2", "Mô hình chưa sẵn sàng."))
    with info_densenet:
        st.markdown("**DenseNet121**")
        cfg = AVAILABLE_MODELS.get("densenet121")
        if cfg is not None:
            st.write("Đường dẫn mô hình:", str(cfg.FINAL_MODEL_PATH))
            st.write("Danh sách lớp:", str(cfg.CLASS_NAMES_PATH))
        else:
            st.caption(MODEL_ERRORS.get("densenet121", "Mô hình chưa sẵn sàng."))

    st.caption(
        "Độ tin cậy là xác suất mô hình ước lượng, không phải xác nhận chắc chắn bệnh. "
        "Mỗi mô hình sử dụng tiền xử lý ảnh riêng theo lúc huấn luyện."
    )

st.divider()
st.caption("Rice Leaf Disease Classification | MobileNetV2 + DenseNet121 Transfer Learning")
