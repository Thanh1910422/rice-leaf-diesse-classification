
import json
import numpy as np
import streamlit as st

from PIL import Image
from tensorflow import keras
from src.config import (
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
)


# ==============================
# 1. CAU HINH GIAO DIEN
# ==============================

st.set_page_config(
    page_title="Rice Leaf Disease Classification",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 HỆ THỐNG PHÂN LOẠI BỆNH LÁ LÚA")
st.markdown(
    "Ứng dụng sử dụng mô hình **MobileNetV2 Transfer Learning** "
    "để nhận diện 6 loại bệnh trên lá lúa."
)

st.divider()


# ==============================
# 2. TAI MO HINH
# ==============================

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


# ==============================
# 3. TIEN XU LY ANH
# ==============================

def preprocess_image(image):
    # Chuyển ảnh về RGB
    image = image.convert("RGB")

    # Resize ảnh về 224 x 224
    image = image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0])
    )

    # Chuyển ảnh thành mảng NumPy float32
    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Thêm chiều Batch
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==============================
# 4. GIAO DIEN TAI ANH
# ==============================

st.subheader("1. Tải ảnh lá lúa lên")

uploaded_file = st.file_uploader(
    "Chọn ảnh cần nhận diện",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original_image,
            caption="Ảnh lá lúa đầu vào",
            use_container_width=True
        )

    with col2:
        st.subheader("Thông tin ảnh")
        st.write("Tên ảnh:", uploaded_file.name)
        st.write("Kích thước:", original_image.size)

    st.divider()

    # ==============================
    # 5. DU DOAN
    # ==============================

    if st.button(
        "🔍 DỰ ĐOÁN BỆNH LÁ LÚA",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner("Mô hình đang phân tích ảnh..."):

                model = load_model()
                class_names = load_class_names()

                processed_image = preprocess_image(
                    original_image
                )
                st.subheader("Thông tin ảnh sau tiền xử lý")

                st.write("Shape:", processed_image.shape)
                st.write("Data type:", processed_image.dtype)
                st.write("Pixel nhỏ nhất:", processed_image.min())
                st.write("Pixel lớn nhất:", processed_image.max())
                probabilities = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                best_index = int(np.argmax(probabilities))

                predicted_class = class_names[best_index]

                confidence = float(
                    probabilities[best_index]
                ) * 100

            # ==============================
            # 6. HIEN THI KET QUA
            # ==============================

            st.success("Đã hoàn thành dự đoán!")

            st.subheader("2. Kết quả nhận diện")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Bệnh được dự đoán",
                    predicted_class
                )

            with result_col2:
                st.metric(
                    "Độ tin cậy",
                    f"{confidence:.2f}%"
                )

            st.divider()

            # ==============================
            # 7. XAC SUAT CAC LOP
            # ==============================

            st.subheader("3. Xác suất dự đoán của 6 lớp")

            for index, class_name in enumerate(class_names):

                probability = float(
                    probabilities[index]
                ) * 100

                st.write(
                    f"**{class_name}: {probability:.2f}%**"
                )

                st.progress(
                    float(probabilities[index])
                )

            # ==============================
            # 8. TOP 3
            # ==============================

            st.divider()
            st.subheader("4. Top 3 kết quả")

            sorted_indices = np.argsort(
                probabilities
            )[::-1]

            for rank, index in enumerate(
                sorted_indices[:3],
                start=1
            ):

                st.write(
                    f"{rank}. **{class_names[index]}** "
                    f"- {probabilities[index] * 100:.2f}%"
                )

            st.caption(
                "Lưu ý: Độ tin cậy là xác suất mô hình ước lượng, "
                "không phải xác nhận chắc chắn bệnh thực tế."
            )

        except Exception as error:
            st.error(f"Lỗi khi dự đoán: {error}")


else:
    st.info(
        "Vui lòng tải lên một ảnh lá lúa để bắt đầu nhận diện."
    )


st.divider()

st.caption(
    "Rice Leaf Disease Classification | "
    "CNN - MobileNetV2"
)
