
import json
import tensorflow as tf

from pathlib import Path
from tensorflow import keras

from src.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
    LEARNING_RATE_STAGE1,
    LEARNING_RATE_STAGE2,
)

from src.data_loader import load_data
from src.model import build_model, unfreeze_model


def create_callbacks(checkpoint_path, log_path):
    """Tạo callbacks để lưu mô hình và theo dõi huấn luyện."""

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
            patience=3,
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
    """Lưu lịch sử huấn luyện thành JSON."""

    data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def train_stage1(model, train_data, val_data):
    """Giai đoạn 1: Feature Extraction."""

    print("\n" + "=" * 60)
    print("GIAI DOAN 1: FEATURE EXTRACTION")
    print("=" * 60)

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
    """Giai đoạn 2: Fine-tuning."""

    print("\n" + "=" * 60)
    print("GIAI DOAN 2: FINE-TUNING")
    print("=" * 60)

    model = unfreeze_model(model)

    print("\nSo tham so co the huan luyen:",
          sum(
              tf.size(variable).numpy()
              for variable in model.trainable_variables
          )
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HUAN LUYEN MO HINH CNN PHAN LOAI BENH LA LUA")
    print("=" * 60)

    # 1. Đọc dữ liệu
    train_data, val_data, test_data, class_names = load_data()

    print("\nSo lop:", len(class_names))

    # 2. Khởi tạo mô hình
    model = build_model(num_classes=len(class_names))

    print("\nKhoi tao mo hinh thanh cong!")

    # 3. Giai đoạn 1
    train_stage1(
        model,
        train_data,
        val_data
    )

    # 4. Nạp lại mô hình tốt nhất của giai đoạn 1
    print("\nNap lai mo hinh tot nhat giai doan 1...")

    model = keras.models.load_model(BEST_MODEL_PATH)

    # 5. Giai đoạn 2
    train_stage2(
        model,
        train_data,
        val_data
    )

    print("\n" + "=" * 60)
    print("HOAN THANH HUAN LUYEN!")
    print("=" * 60)

    print("Mo hinh giai doan 1:", BEST_MODEL_PATH)
    print("Mo hinh giai doan 2:", FINAL_MODEL_PATH)

    print("\nLuu y: Tap Testing chua duoc su dung trong huan luyen.")


if __name__ == "__main__":
    main()
