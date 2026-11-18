import json
import numpy as np
import streamlit as st

from PIL import Image, ImageOps
from tensorflow import keras
from src.config import (
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
)
# Hai hàm này có trong cả code gốc lẫn code mới -> app chạy với mọi phiên bản
from src.image_utils import prepare_pil_image, to_model_input

# Tên hiển thị tiếng Việt. Khoá đã chuẩn hoá (chữ thường, bỏ dấu cách/_/-)
# nên khớp với "Leaf scald", "Leaf_scald", "Narrow_Brown_Spot"...
VIETNAMESE_NAMES = {
    # Bộ Rice_Leaf_AUG
    "bacterialleafblight": "Bạc lá (Bacterial Leaf Blight)",
    "brownspot": "Đốm nâu (Brown Spot)",
    "healthyriceleaf": "Lá khỏe mạnh (Healthy)",
    "leafblast": "Đạo ôn lá (Leaf Blast)",
    "leafscald": "Cháy bìa lá (Leaf Scald)",
    "sheathblight": "Khô vằn (Sheath Blight)",
    # Bộ RiceLeafDisease-BD5
    "blast": "Đạo ôn (Blast)",
    "narrowbrownspot": "Đốm nâu hẹp (Narrow Brown Spot)",
    "normalleaf": "Lá khỏe mạnh (Normal Leaf)",
    "tungro": "Vàng lùn Tungro (Tungro)",
}


def display_name(class_name):
    key = "".join(ch for ch in class_name.lower() if ch.isalnum())
    return VIETNAMESE_NAMES.get(key, class_name)


# 1. CAU HINH GIAO DIEN

st.set_page_config(
    page_title="Rice Leaf Disease Classification",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 HỆ THỐNG PHÂN LOẠI BỆNH LÁ LÚA")
st.markdown(
    "Ứng dụng sử dụng mô hình **MobileNetV2 Transfer Learning** "
    "để nhận diện bệnh trên lá lúa."
)

st.divider()

# 2. TAI MO HINH

@st.cache_resource
def load_model():
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {FINAL_MODEL_PATH}"
        )

    return keras.models.load_model(FINAL_MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay danh sach lop: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 3. TIEN XU LY ANH

def preprocess_image(image):
    # Dùng chung hàm tiền xử lý với lúc huấn luyện:
    # xoay theo EXIF, chuyển RGB, resize 224x224, thêm chiều batch.
    return to_model_input(prepare_pil_image(image))
# 4. GIAO DIEN TAI ANH

uploaded_file = st.file_uploader(
    "Tải ảnh lá lúa cần nhận diện",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    original_image = ImageOps.exif_transpose(Image.open(uploaded_file))

    # Ảnh bên trái - kết quả bên phải, cùng một màn hình.
    # Ảnh điện thoại thường dọc và rất cao -> thu nhỏ BẢN XEM TRƯỚC (cao tối đa
    # PREVIEW_MAX_HEIGHT px) để ảnh và kết quả nằm gọn trong một màn hình.
    # Mô hình vẫn dự đoán trên ảnh gốc đầy đủ.
    PREVIEW_MAX_HEIGHT = 480
    PREVIEW_MAX_WIDTH = 640

    preview = original_image.copy()
    preview.thumbnail((PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT))

    image_col, result_col = st.columns([1, 1], gap="large")

    with image_col:
        st.image(
            preview,
            caption=(
                f"{uploaded_file.name} - "
                f"{original_image.size[0]}x{original_image.size[1]}"
            ),
            width=preview.size[0]
        )

    with result_col:
        try:
            # 5. DU DOAN (tự động ngay khi tải ảnh)
            with st.spinner("Mô hình đang phân tích ảnh..."):
                model = load_model()
                class_names = load_class_names()

                processed_image = preprocess_image(original_image)

                probabilities = model.predict(
                    processed_image,
                    verbose=0
                )[0]

            sorted_indices = np.argsort(probabilities)[::-1]
            best_index = int(sorted_indices[0])
            predicted_class = class_names[best_index]
            confidence = float(probabilities[best_index]) * 100

            # 6. KET QUA CHINH - khung nổi bật, hiện đủ tên bệnh dài
            st.subheader("Kết quả nhận diện")

            result_text = (
                f"### {display_name(predicted_class)}\n"
                f"Độ tin cậy: **{confidence:.2f}%**"
            )
            if confidence >= 80:
                st.success(result_text)
            elif confidence >= 50:
                st.warning(result_text + "  \nĐộ tin cậy trung bình, nên kiểm tra lại.")
            else:
                st.error(result_text + "  \nĐộ tin cậy thấp, kết quả không chắc chắn.")

            # 7. XAC SUAT CAC LOP (sắp xếp giảm dần, top 3 in đậm)
            st.markdown("**Xác suất các lớp**")

            for rank, index in enumerate(sorted_indices, start=1):
                probability = float(probabilities[index])
                label = f"{display_name(class_names[index])}: {probability * 100:.2f}%"

                st.progress(
                    probability,
                    text=f"**{rank}. {label}**" if rank <= 3 else label
                )

            st.caption(
                "Lưu ý: Độ tin cậy là xác suất mô hình ước lượng, "
                "không phải xác nhận chắc chắn bệnh thực tế."
            )

            with st.expander("Thông tin ảnh sau tiền xử lý"):
                st.write("Shape:", processed_image.shape)
                st.write("Data type:", processed_image.dtype)
                st.write("Pixel nhỏ nhất:", processed_image.min())
                st.write("Pixel lớn nhất:", processed_image.max())

        except Exception as error:
            st.error(f"Lỗi khi dự đoán: {error}")


else:
    st.info(
        "Vui lòng tải lên một ảnh lá lúa để bắt đầu nhận diện."
    )


st.divider()

st.caption(
    "Rice Leaf Disease Classification | CNN - MobileNetV2"
)