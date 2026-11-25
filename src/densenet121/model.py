import math

import keras
from keras import layers

from src.densenet121.config import (
    IMAGE_SIZE,
    DROPOUT_RATE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
)

def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15, fill_mode="reflect"),
            layers.RandomZoom((-0.15, 0.1), fill_mode="reflect"),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(num_classes):
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    base_model = keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # 1 pha: mở TOÀN BỘ DenseNet121 ngay từ đầu, trừ BatchNorm (giữ thống kê ImageNet)
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = build_augmentation()(inputs)
    # [KHAC MOBILENETV2] chuẩn hoá kiểu "torch" mà DenseNet121 dùng khi học ImageNet
    x = keras.Sequential(
        [
            layers.Rescaling(1.0 / 255.0),
            layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
            ),
        ],
        name="preprocess_input",
    )(x)
    # training=False: BatchNorm của DenseNet121 luôn ở chế độ suy luận
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="DenseNet121_RiceLeaf")


def make_loss():
    return keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=make_loss(),
        metrics=["accuracy"],
    )
    return model


def find_backbone(model):
    """DenseNet121 là mô hình con (keras.Model) duy nhất ngoài augmentation / preprocess."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name not in ("augmentation", "preprocess_input"):
            return layer
    raise ValueError("Khong tim thay DenseNet121 trong mo hinh")


def print_trainable_summary(model):
    base_model = find_backbone(model)
    trainable = sum(layer.trainable for layer in base_model.layers)
    params = sum(math.prod(w.shape) for w in model.trainable_weights)
    print(f"Huan luyen {trainable}/{len(base_model.layers)} lop cua DenseNet121 "
          f"(BatchNorm co dinh) | {params:,} tham so duoc cap nhat")


if __name__ == "__main__":
    model = build_model(num_classes=6)
    compile_model(model)
    print_trainable_summary(model)
    model.summary()
