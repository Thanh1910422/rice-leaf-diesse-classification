from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "densenet121"
DATASET = "Rice_Leaf_AUG"
SOURCE_DIR = ROOT_DIR / "data" / "raw" / "Rice_Leaf_AUG"

# Thư mục con KHÔNG phải lớp bệnh (vd. thư mục chứa file mô tả) -> bỏ qua
IGNORED_FOLDERS = {"__MACOSX"}

PROCESSED_DIR = ROOT_DIR / "data" / "processed" / MODEL_NAME
SPLIT_DIR = ROOT_DIR / "data" / "splits" / MODEL_NAME

TRAIN_MANIFEST = SPLIT_DIR / "train.json"
VAL_MANIFEST = SPLIT_DIR / "val.json"
TEST_MANIFEST = SPLIT_DIR / "test.json"
CLASS_NAMES_PATH = SPLIT_DIR / "class_names.json"

MODEL_DIR = ROOT_DIR / "models" / MODEL_NAME
BEST_MODEL_PATH = MODEL_DIR / "densenet121_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "densenet121_rice_leaf_final.keras"

OUTPUT_DIR = ROOT_DIR / "outputs" / MODEL_NAME
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 224x224 là kích thước gốc DenseNet121 được huấn luyện trên ImageNet (giống MobileNetV2).
IMAGE_SIZE = (224, 224)

# Ảnh đã xoay EXIF + resize sẵn: data/processed/224x224/<lop>/<ten>.png
PROCESSED_IMAGE_DIR = PROCESSED_DIR / f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"

# Chia 70/15/15 phân tầng theo lớp, sau khi loại ảnh trùng mã dHash 256.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# ---------------------------------------------------------------------------
# Huấn luyện
# ---------------------------------------------------------------------------
BATCH_SIZE = 16  
EPOCHS = 30
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 8

# MixUp / CutMix: bật nếu thấy train acc ~100% mà val thấp hơn nhiều
USE_MIX_AUG = False
MIX_PROB = 0.5
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
