
import json
import shutil
import tensorflow as tf

from tensorflow import keras

from src.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
)

from src.data_loader import load_data

from src.model import (
    build_model,
    compile_stage1,
    unfreeze_model,
)


def create_callbacks(checkpoint_path, log_path):
    """
    Tao callbacks de luu mo hinh va theo doi qua trinh huan luyen.
    """

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        keras.callbacks.CSVLogger(
            str(log_path),
            append=False
        )
    ]


def save_history(history, file_path):
    """
    Luu lich su huan luyen thanh JSON.
    """

    data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def train_stage1(model, train_data, val_data):
    """
    Giai doan 1: Feature Extraction.
    Dong bang MobileNetV2 va huan luyen lop phan loai.
    """

    print("\n" + "=" * 60)
    print("GIAI DOAN 1: FEATURE EXTRACTION")
    print("=" * 60)

    # Compile voi learning rate giai doan 1
    model = compile_stage1(model)

    callbacks = create_callbacks(
        BEST_MODEL_PATH,
        OUTPUT_DIR / "stage1_training.csv"
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks,
        verbose=1
    )

    save_history(
        history,
        OUTPUT_DIR / "stage1_history.json"
    )

    print("\nHoan thanh giai doan 1!")

    return history


def train_stage2(model, train_data, val_data):
    """
    Giai doan 2: Fine-tuning.
    Mo bang mot so lop cuoi cua MobileNetV2.
    """

    print("\n" + "=" * 60)
    print("GIAI DOAN 2: FINE-TUNING")
    print("=" * 60)

    # Mo bang cac lop cuoi va compile lai
    model = unfreeze_model(model)

    trainable_params = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    print(
        "\nSo tham so co the huan luyen:",
        trainable_params
    )

    callbacks = create_callbacks(
        FINAL_MODEL_PATH,
        OUTPUT_DIR / "stage2_training.csv"
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks,
        verbose=1
    )

    save_history(
        history,
        OUTPUT_DIR / "stage2_history.json"
    )

    print("\nHoan thanh giai doan 2!")

    return history


def main():

    # Tao thu muc neu chua ton tai
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("HUAN LUYEN MOBILENETV2 PHAN LOAI BENH LA LUA")
    print("=" * 60)

    # 1. DOC DU LIEU

    print("\n[1] DANG DOC DU LIEU...")

    train_data, val_data, test_data, class_names = load_data()

    num_classes = len(class_names)

    print("\nSo lop:", num_classes)

    print("\nDanh sach cac lop:")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    # 2. KHOI TAO MO HINH

    print("\n[2] DANG KHOI TAO MO HINH...")

    model, base_model = build_model(
        num_classes=num_classes
    )

    print("\nKhoi tao mo hinh thanh cong!")

    print("\nTong so tham so:", model.count_params())
    # 3. GIAI DOAN 1

    print("\n[3] BAT DAU GIAI DOAN 1...")

    train_stage1(
        model,
        train_data,
        val_data
    )
    # 4. NAP MO HINH TOT NHAT GIAI DOAN 1

    print("\n[4] NAP MO HINH TOT NHAT GIAI DOAN 1...")

    model = keras.models.load_model(BEST_MODEL_PATH)

    # Dùng mô hình tốt nhất Stage 1 làm phương án an toàn ban đầu.
    # Nếu Stage 2 không cải thiện val_loss, file FINAL vẫn hợp lệ.
    shutil.copy2(BEST_MODEL_PATH, FINAL_MODEL_PATH)

    print("\nNap mo hinh giai doan 1 thanh cong!")
    # 5. GIAI DOAN 2

    print("\n[5] BAT DAU GIAI DOAN 2...")

    train_stage2(
        model,
        train_data,
        val_data
    )
    # 6. KET THUC

    print("\n" + "=" * 60)
    print("HOAN THANH HUAN LUYEN!")
    print("=" * 60)

    print("\nMo hinh tot nhat giai doan 1:")
    print(BEST_MODEL_PATH)

    print("\nMo hinh tot nhat giai doan 2:")
    print(FINAL_MODEL_PATH)

    print("\nTap Test chua duoc su dung trong huan luyen.")

    print("\nHay su dung evaluate.py de danh gia tren tap Test.")


if __name__ == "__main__":
    main()
