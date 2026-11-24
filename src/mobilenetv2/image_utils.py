"""
Hàm xử lý ảnh dùng chung cho prepare_data, data_loader, predict.
Không import TensorFlow -> prepare_data chạy được mà không cần TF.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.mobilenetv2.config import IMAGE_SIZE, SOURCE_DIR, PROCESSED_IMAGE_DIR


def prepare_pil_image(image, image_size=IMAGE_SIZE):
    """
    Ảnh PIL (vd. ảnh người dùng tải lên app.py) -> xoay đúng chiều theo EXIF
    -> RGB -> resize (có antialias). Trả về mảng uint8 (H, W, 3).
    Đây là bước tiền xử lý DUY NHẤT: train, evaluate, predict và app.py
    đều đi qua hàm này nên ảnh luôn được xử lý giống hệt nhau.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image_size[1], image_size[0]
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR, reducing_gap=None)
    return np.asarray(image, dtype=np.uint8)


def load_image_array(image_path, image_size=IMAGE_SIZE):
    """Đọc ảnh từ file rồi xử lý bằng prepare_pil_image."""
    with Image.open(image_path) as image:
        return prepare_pil_image(image, image_size)


def to_model_input(image_array):
    """(H, W, 3) uint8 -> (1, H, W, 3) float32, pixel [0, 255].
    Chuẩn hoá về [-1, 1] được làm BÊN TRONG mô hình, KHÔNG chia 255 ở đây."""
    return np.expand_dims(np.asarray(image_array).astype(np.float32), axis=0)


def processed_path_for(raw_path):
    """
    data/raw/RiceLeafDisease-BD5/Blast/IMG_001.jpg
      -> data/processed/224x224/Blast/IMG_001.png
    Trả về None nếu ảnh nằm ngoài SOURCE_DIR (vd: ảnh khi predict).
    """
    try:
        relative = Path(raw_path).absolute().relative_to(SOURCE_DIR.absolute())
    except ValueError:
        return None
    return PROCESSED_IMAGE_DIR / relative.with_suffix(".png")


def save_processed_image(raw_path, overwrite=False):
    """
    Lưu ảnh đã xử lý dạng PNG (không nén mất dữ liệu): đọc lại cho ra mảng
    GIỐNG HỆT load_image_array(raw_path). Trả về True nếu vừa tạo file mới.
    """
    target = processed_path_for(raw_path)
    if target is None or (target.exists() and not overwrite):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(load_image_array(raw_path)).save(target)
    return True
