
import json
import hashlib
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    SOURCE_DIR,
    SPLIT_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SUPPORTED_EXTENSIONS,
)


def calculate_hash(file_path):
    """Tính mã SHA256 để nhận diện ảnh trùng."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def collect_images():
    """Thu thập ảnh và nhóm theo mã SHA256."""

    hash_groups = defaultdict(list)
    class_names = sorted(
        folder.name
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    for label, class_name in enumerate(class_names):
        class_dir = SOURCE_DIR / class_name

        for image_path in class_dir.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                image_hash = calculate_hash(image_path)

                hash_groups[image_hash].append({
                    "path": str(image_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                })

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
            cross_class_groups.append(group)
            continue

        # Giữ lại một ảnh đại diện trong mỗi nhóm.
        records.append(group[0])
        duplicate_removed += len(group) - 1

    print("\nKET QUA XU LY ANH TRUNG")
    print("So anh trung cung nhan da loai:", duplicate_removed)
    print("So nhom trung khac nhan can kiem tra:",
          len(cross_class_groups))

    if cross_class_groups:
        report_path = SPLIT_DIR / "cross_class_duplicates.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                cross_class_groups,
                f,
                ensure_ascii=False,
                indent=4
            )

        print("Danh sach anh can kiem tra:", report_path)

    return records


def save_json(file_path, data):
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

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    save_json(SPLIT_DIR / "train.json", train_records)
    save_json(SPLIT_DIR / "val.json", val_records)
    save_json(SPLIT_DIR / "test.json", test_records)
    save_json(SPLIT_DIR / "class_names.json", class_names)

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


def main():
    print("=" * 60)
    print("CHUAN BI VA CHIA BO DU LIEU BENH LA LUA")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc: {SOURCE_DIR}"
        )

    hash_groups, class_names = collect_images()

    print("\nSo lop:", len(class_names))
    print("Tong so nhom anh theo hash:", len(hash_groups))

    records = remove_duplicates(hash_groups)

    if not records:
        raise ValueError("Khong co anh hop le de chia du lieu.")

    split_dataset(records, class_names)

    print("\nHOAN THANH GIAI DOAN 2!")


if __name__ == "__main__":
    main()
