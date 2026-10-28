
from pathlib import Path

# Thư mục gốc của dự án
ROOT_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn bộ dữ liệu theo cấu trúc trong ảnh

DATA_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
    / "archive (1)"
    / "Rice_Leaf_AUG"
)

# Thư mục đầu ra
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Thông số xử lý ảnh
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Thông số huấn luyện
EPOCHS = 10
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Tên các file đầu ra
MODEL_PATH = MODEL_DIR / "rice_leaf_model.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_rice_leaf_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"