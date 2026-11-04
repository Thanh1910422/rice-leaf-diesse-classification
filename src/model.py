
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    IMAGE_SIZE,
    LEARNING_RATE_STAGE1,
    LEARNING_RATE_STAGE2,
    FINE_TUNE_LAST_LAYERS,
)


def build_model(num_classes=6):
    """
    Xay dung mo hinh CNN MobileNetV2
    Giai doan dau: dong bang backbone.
    """

    inputs = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="input_image"
    )

    # 1. Data augmentation
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation"
    )

    x = augmentation(inputs)

    # 2. MobileNetV2 backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Dong bang backbone trong giai doan 1
    base_model.trainable = False

    x = base_model(x, training=False)

    # 3. Classification head
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="dense_features"
    )(x)

    x = layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Rice_Leaf_MobileNetV2"
    )

    # 4. Compile model cho giai doan 1
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

    base_model.trainable = True

    # Mac dinh dong bang tat ca cac lop
    for layer in base_model.layers:
        layer.trainable = False

    # Mo bang cac lop cuoi, tru BatchNormalization
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
    print("XAY DUNG MO HINH CNN MOBILENETV2")
    print("=" * 60)

    model = build_model(num_classes=6)

    model.summary()

    total_params = model.count_params()

    trainable_params = sum(
        tf.size(variable).numpy()
        for variable in model.trainable_variables
    )

    print("\nTONG SO THAM SO:", total_params)
    print("THAM SO CO THE HUAN LUYEN:", trainable_params)

    print("\nKHOI TAO MO HINH THANH CONG!")
