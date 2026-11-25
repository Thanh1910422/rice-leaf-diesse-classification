import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from src.densenet121.config import (
    ROOT_DIR,
    SOURCE_DIR,
    SPLIT_DIR,
    PROCESSED_IMAGE_DIR,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
    CLASS_NAMES_PATH,
    IGNORED_FOLDERS,
    SUPPORTED_EXTENSIONS,
    TRAIN_RATIO,
    VAL_RATIO,
    SEED,
)
from src.densenet121.image_utils import save_processed_image

TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO


def calculate_dhash(file_path, hash_size=16):
    """Tính mã dHash 256 bit (16x16) để nhận diện ảnh trùng."""
    with Image.open(file_path) as image:
        image.draft("L", ((hash_size + 1) * 4, hash_size * 4))  # đọc JPEG nhanh hơn
        pixels = np.asarray(
            image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS),
            dtype=np.int16,
        )
    bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


def collect_images():
    """Thu thập ảnh và nhóm theo mã dHash."""

    hash_groups = defaultdict(list)
    bad_images = []

    for image_path in sorted(SOURCE_DIR.rglob("*")):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue

        # Bỏ file ẩn / file rác của macOS (__MACOSX/, ._anh.jpg)
        parts = image_path.relative_to(SOURCE_DIR).parts
        if image_path.name.startswith(".") or IGNORED_FOLDERS.intersection(parts):
            continue

        if image_path.parent == SOURCE_DIR:
            raise ValueError(
                f"Anh nam truc tiep trong {SOURCE_DIR}: {image_path.name}. "
                "Anh phai nam trong thu muc mang ten lop."
            )

        try:
            image_hash = calculate_dhash(image_path)
        except (UnidentifiedImageError, OSError, ValueError):
            bad_images.append(image_path)
            continue

        hash_groups[image_hash].append({
            # Đường dẫn tương đối so với ROOT_DIR (dùng cho data/processed)
            "path": image_path.absolute().relative_to(ROOT_DIR.absolute()).as_posix(),
            "class_name": image_path.parent.name,
            "source_folder": image_path.parent.parent.name,
        })

    class_names = sorted({
        item["class_name"]
        for group in hash_groups.values()
        for item in group
    })

    # Gán nhãn sau khi biết đủ danh sách lớp
    for group in hash_groups.values():
        for item in group:
            item["label"] = class_names.index(item["class_name"])

    if bad_images:
        print("So anh loi bi bo qua:", len(bad_images))
        for path in bad_images:
            print("  ", path)

    return hash_groups, class_names


def remove_duplicates(hash_groups):
    """Loại ảnh trùng cùng nhãn; bỏ qua nhóm trùng khác nhãn."""

    records = []
    duplicate_removed = 0
    cross_class_groups = []

    for image_hash, group in hash_groups.items():

        labels = {item["label"] for item in group}

        # Ảnh giống nhau nhưng thuộc các lớp khác nhau:
        # Không tự ý chọn nhãn, đưa vào danh sách kiểm tra.
        if len(labels) > 1:
            cross_class_groups.append({"dhash": image_hash, "images": group})
            continue

        # Giữ lại một ảnh đại diện trong mỗi nhóm.
        records.append(group[0])
        duplicate_removed += len(group) - 1

    print("\nKET QUA XU LY ANH TRUNG (dHash 256)")
    print("So anh trung cung nhan da loai:", duplicate_removed)
    print("So nhom trung khac nhan can kiem tra:", len(cross_class_groups))

    if cross_class_groups:
        report_path = SPLIT_DIR / "cross_class_duplicates.json"
        save_json(report_path, cross_class_groups)
        print("Danh sach anh can kiem tra:", report_path)

    return records


def save_json(file_path, data):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def split_dataset(records, class_names):
    """Chia dữ liệu theo tỷ lệ 70/15/15."""

    labels = [item["label"] for item in records]

    train_records, temp_records = train_test_split(
        records,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED,
        stratify=labels
    )

    temp_labels = [item["label"] for item in temp_records]

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_records, test_records = train_test_split(
        temp_records,
        test_size=relative_test_ratio,
        random_state=SEED,
        stratify=temp_labels
    )

    save_json(TRAIN_MANIFEST, train_records)
    save_json(VAL_MANIFEST, val_records)
    save_json(TEST_MANIFEST, test_records)
    save_json(CLASS_NAMES_PATH, class_names)

    print("\nKET QUA CHIA DU LIEU")
    print("Training:", len(train_records))
    print("Validation:", len(val_records))
    print("Testing:", len(test_records))
    print("Tong:", len(records))

    for split_name, split_records in [
        ("Training", train_records),
        ("Validation", val_records),
        ("Testing", test_records),
    ]:
        print(f"\n{split_name} theo tung lop:")

        for label, class_name in enumerate(class_names):
            count = sum(
                item["label"] == label
                for item in split_records
            )
            print(f"{class_name}: {count}")


def build_processed_images(overwrite=False):
    """Tạo ảnh 224x224 cho DenseNet121 từ 3 manifest (ảnh đã có -> bỏ qua)."""

    records = []
    for manifest in (TRAIN_MANIFEST, VAL_MANIFEST, TEST_MANIFEST):
        if not manifest.exists():
            raise FileNotFoundError(
                f"Chua co {manifest}. Hay chay: python -m src.densenet121.prepare_data"
            )
        with open(manifest, "r", encoding="utf-8") as f:
            records.extend(json.load(f))

    print("\nTAO ANH 224x224:", PROCESSED_IMAGE_DIR)
    created = 0
    for index, record in enumerate(records, start=1):
        if save_processed_image(ROOT_DIR / record["path"], overwrite):
            created += 1
        if index % 500 == 0 or index == len(records):
            print(f"  {index}/{len(records)} anh")
    print(f"Anh moi tao: {created} | Da co san: {len(records) - created}")


def main():
    print("=" * 60)
    print("CHUAN BI VA CHIA BO DU LIEU BENH LA LUA - DenseNet121")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc: {SOURCE_DIR}"
        )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    hash_groups, class_names = collect_images()

    print("\nSo lop:", len(class_names))
    print("Tong so nhom anh theo dHash:", len(hash_groups))

    records = remove_duplicates(hash_groups)

    if not records:
        raise ValueError("Khong co anh hop le de chia du lieu.")

    split_dataset(records, class_names)
    build_processed_images()

    print("\nHOAN THANH CHUAN BI DU LIEU!")


if __name__ == "__main__":
    if "--processed" in sys.argv:
        build_processed_images()
    else:
        main()