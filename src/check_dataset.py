
import hashlib
from collections import defaultdict
from PIL import Image, UnidentifiedImageError

from src.config import SOURCE_DIR, SUPPORTED_EXTENSIONS


def calculate_hash(image_path):
    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    print("=" * 65)
    print("KIEM TRA BO DU LIEU BENH LA LUA")
    print("=" * 65)

    if not SOURCE_DIR.exists():
        print(f"KHONG TIM THAY DUONG DAN: {SOURCE_DIR}")
        return

    class_folders = sorted(
        folder for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("Khong tim thay thu muc lop!")
        return

    print(f"\nDuong dan: {SOURCE_DIR}")
    print(f"So lop tim thay: {len(class_folders)}")

    total_images = 0
    valid_images = 0
    invalid_images = []
    hashes = defaultdict(list)
    class_counts = {}

    for class_folder in class_folders:
        images = [
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        class_counts[class_folder.name] = len(images)
        total_images += len(images)

        print(f"\nLop: {class_folder.name}")
        print(f"So anh: {len(images)}")

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()

                valid_images += 1

                image_hash = calculate_hash(image_path)
                hashes[image_hash].append(str(image_path))

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):
                invalid_images.append(str(image_path))

    duplicate_groups = [
        paths for paths in hashes.values()
        if len(paths) > 1
    ]

    duplicate_count = sum(
        len(paths) - 1 for paths in duplicate_groups
    )

    print("\n" + "=" * 65)
    print("KET QUA KIEM TRA")
    print("=" * 65)

    print(f"Tong so anh: {total_images}")
    print(f"Anh doc duoc: {valid_images}")
    print(f"Anh loi: {len(invalid_images)}")
    print(f"So nhom anh trung: {len(duplicate_groups)}")
    print(f"So anh trung du thua: {duplicate_count}")

    print("\nTHONG KE TUNG LOP:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    if invalid_images:
        print("\nMOT SO ANH LOI:")
        for path in invalid_images[:10]:
            print(path)

    if duplicate_groups:
        print("\nMOT SO NHOM ANH TRUNG:")
        for group in duplicate_groups[:5]:
            print("\n".join(group))
            print("-" * 30)

    print("\nHOAN THANH KIEM TRA!")


if __name__ == "__main__":
    main()
