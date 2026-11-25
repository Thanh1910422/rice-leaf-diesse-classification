import json
import time

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.densenet121.config import (
    MODEL_NAME,
    OUTPUT_DIR,
    MODEL_DIR,
    LOG_DIR,
    PLOT_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    SEED,
    USE_CLASS_WEIGHTS,
)
from src.densenet121.data_loader import load_data, compute_class_weights
from src.densenet121.model import build_model, compile_model, print_trainable_summary


def make_callbacks(checkpoint_path, log_path, patience):
    return [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(log_path),
    ]


def save_history(history, path):
    data = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def plot_history(history, path):
    """Vẽ accuracy và loss theo epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, key, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "loss", "Loss"),
    ]:
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    keras.utils.set_random_seed(SEED)
    start_time = time.time()  # [THEM DE SO SANH]
    for folder in (MODEL_DIR, LOG_DIR, PLOT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HUAN LUYEN DENSENET121 (1 PHA) - PHAN LOAI BENH LA LUA")
    print("=" * 60)

    print("\n[1] DOC DU LIEU")
    train_ds, val_ds, _, class_names = load_data()
    class_weight = compute_class_weights(class_names) if USE_CLASS_WEIGHTS else None
    if class_weight:
        print("Trong so lop:", {class_names[i]: round(w, 3) for i, w in class_weight.items()})

    print("\n[2] KHOI TAO MO HINH")
    model = build_model(len(class_names))
    compile_model(model)
    print_trainable_summary(model)

    print(f"\n[3] HUAN LUYEN TOAN BO DENSENET121 (lr = {LEARNING_RATE}, toi da {EPOCHS} epoch)")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=make_callbacks(
            BEST_MODEL_PATH, LOG_DIR / "training.csv",
            patience=EARLY_STOPPING_PATIENCE,
        ),
    )
    h = save_history(history, LOG_DIR / "history.json")

    # EarlyStopping đã khôi phục trọng số tốt nhất (val_loss thấp nhất)
    model.save(FINAL_MODEL_PATH)
    plot_history(h, PLOT_DIR / "training_curves.png")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)

    # [THEM DE SO SANH] tóm tắt cho src/compare_models.py - không ảnh hưởng mô hình
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "training_phases": 1,
            "total_params": int(model.count_params()),
            "training_minutes": round((time.time() - start_time) / 60, 1),
            "epochs": len(h["loss"]),
            "val_accuracy": float(val_acc),
            "model_size_mb": round(FINAL_MODEL_PATH.stat().st_size / 2**20, 1),
        }, f, indent=2)
    print("\n" + "=" * 60)
    print(f"Val accuracy (mo hinh cuoi): {val_acc * 100:.2f}% | Val loss: {val_loss:.4f}")
    print("Mo hinh:", FINAL_MODEL_PATH)
    print("Bieu do:", PLOT_DIR / "training_curves.png")
    print("Tiep theo: python -m src.densenet121.evaluate")


if __name__ == "__main__":
    main()
