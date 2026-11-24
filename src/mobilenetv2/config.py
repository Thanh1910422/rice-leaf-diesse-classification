from pathlib import Path

# -------------------------------------------------
# Đường dẫn
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "mobilenetv2"
DATASET = "Rice_Leaf_AUG"

# Dữ liệu ảnh gốc dùng chung
SOURCE_DIR = ROOT_DIR / "data" / "raw" / DATASET

# Dữ liệu đã xử lý và chia tập riêng cho MobileNetV2
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / MODEL_NAME
SPLIT_DIR = ROOT_DIR / "data" / "splits" / MODEL_NAME

TRAIN_MANIFEST = SPLIT_DIR / "train.json"
VAL_MANIFEST = SPLIT_DIR / "val.json"
TEST_MANIFEST = SPLIT_DIR / "test.json"
CLASS_NAMES_PATH = SPLIT_DIR / "class_names.json"

DUPLICATE_REPORT_PATH = SPLIT_DIR / "duplicate_report.csv"
DUPLICATE_PREVIEW_PATH = (
    ROOT_DIR / "outputs" / MODEL_NAME
    / "reports" / "near_duplicate_preview.jpg"
)

# Mô hình
MODEL_DIR = ROOT_DIR / "models" / MODEL_NAME

BEST_STAGE1_MODEL_PATH = MODEL_DIR / "mobilenetv2_stage1_best.keras"
BEST_STAGE2_MODEL_PATH = MODEL_DIR / "mobilenetv2_stage2_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "mobilenetv2_rice_leaf_final.keras"
# Mô hình tốt nhất giai đoạn 1 (trước đây trỏ nhầm vào FINAL_MODEL_PATH
# -> shutil.copy2 chép file vào chính nó và báo lỗi SameFileError)
BEST_MODEL_PATH = BEST_STAGE1_MODEL_PATH

# Kết quả huấn luyện và đánh giá
OUTPUT_DIR = ROOT_DIR / "outputs" / MODEL_NAME

LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

# Cấu hình ảnh
IMAGE_SIZE = (224, 224)
PROCESSED_IMAGE_DIR = PROCESSED_DIR / f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
BATCH_SIZE = 8
SEED = 42

# Chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Huấn luyện
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 40

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

FINE_TUNE_LAST_LAYERS = 30

# Định dạng ảnh
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

