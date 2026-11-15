"""
Tiền xử lý ảnh DÙNG CHUNG cho huấn luyện, đánh giá, predict.py và app.py.

Dùng một hàm duy nhất để tránh lệch tiền xử lý giữa train và suy luận:
- Xoay ảnh theo EXIF Orientation (đa số ảnh điện thoại trong bộ dữ liệu có
  Orientation = 6, tf.io.decode_image bỏ qua thẻ này).
- Resize có khử răng cưa (PIL BILINEAR), thay vì tf.image.resize không
  antialias khi thu nhỏ ảnh 1600x1200 -> 224x224.
"""

import numpy as np
from PIL import Image, ImageOps

from src.config import IMAGE_SIZE


def prepare_pil_image(image, image_size=IMAGE_SIZE):
    """PIL Image -> mảng uint8 (H, W, 3), pixel thang [0, 255]."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(
        (image_size[1], image_size[0]),
        Image.BILINEAR
    )
    return np.asarray(image, dtype=np.uint8)


def load_image_array(image_path, image_size=IMAGE_SIZE):
    """Đọc ảnh từ đường dẫn -> mảng uint8 (H, W, 3)."""
    with Image.open(image_path) as image:
        return prepare_pil_image(image, image_size)


def to_model_input(image_array):
    """Mảng uint8 (H, W, 3) -> batch float32 (1, H, W, 3).

    Không gọi preprocess_input ở đây vì model.py đã tích hợp sẵn.
    """
    return np.expand_dims(image_array.astype(np.float32), axis=0)
