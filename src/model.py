
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    IMAGE_SIZE,
    LEARNING_RATE_STAGE1,
    LEARNING_RATE_STAGE2,
    FINE_TUNE_LAST_LAYERS,
)


def build_model(num_classes, image_size=IMAGE_SIZE):
    """
    Xay dung mo hinh MobileNetV2 Transfer Learning.
    """

    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="input_image"
    )

    # 1. Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomContrast(0.10)(x)

    # 2. MobileNetV2 preprocessing
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # 3. Pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Dong bang backbone o giai doan 1
    base_model.trainable = False

    x = base_model(x, training=False)

    # 4. Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Rice_Leaf_MobileNetV2"
    )

    return model, base_model


def compile_stage1(model):
    """
    Compile mo hinh cho giai doan 1:
    Chi huan luyen cac lop phan loai.
    """

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE_STAGE1
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def unfreeze_model(model):
    """
    Mo bang mot so lop cuoi cua MobileNetV2
    de fine-tuning o giai doan 2.
    """

    base_model = model.get_layer("mobilenetv2_1.00_224")

    # Cho phep fine-tuning backbone
    base_model.trainable = True

    # Dong bang tat ca cac lop truoc
    for layer in base_model.layers:
        layer.trainable = False

    # Mo bang cac lop cuoi
    for layer in base_model.layers[-FINE_TUNE_LAST_LAYERS:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # Compile lai voi learning rate nho
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE_STAGE2
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("XAY DUNG MO HINH MOBILENETV2")
    print("=" * 60)

    # Khoi tao mo hinh
    model, base_model = build_model(num_classes=6)

    # Compile giai doan 1
    compile_stage1(model)

    # Hien thi cau truc mo hinh
    model.summary()

    # Thong ke tham so
    total_params = model.count_params()

    trainable_params = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    non_trainable_params = total_params - trainable_params

    print("\nTONG SO THAM SO:", total_params)
    print("THAM SO CO THE HUAN LUYEN:", trainable_params)
    print("THAM SO KHONG HUAN LUYEN:", non_trainable_params)

    print("\nKHOI TAO MO HINH THANH CONG!")
