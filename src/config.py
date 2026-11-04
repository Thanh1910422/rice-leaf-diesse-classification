
from pathlib import Path

# Thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
SOURCE_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
    / "Rice_Leaf_AUG"
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLIT_DIR = ROOT_DIR / "data" / "splits"

# Thư mục lưu mô hình
MODEL_DIR = ROOT_DIR / "models"

# Thư mục lưu kết quả
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Cấu hình ảnh
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

# Chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Huấn luyện
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 3

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

FINE_TUNE_LAST_LAYERS = 30

# Định dạng ảnh
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# File dữ liệu
TRAIN_MANIFEST = SPLIT_DIR / "train.json"
VAL_MANIFEST = SPLIT_DIR / "val.json"
TEST_MANIFEST = SPLIT_DIR / "test.json"

CLASS_NAMES_PATH = SPLIT_DIR / "class_names.json"

# File mô hình
BEST_MODEL_PATH = MODEL_DIR / "best_rice_leaf_model.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final_rice_leaf_model.keras"
