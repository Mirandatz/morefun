import functools
import os
import pathlib

import cv2
import pandas as pd
import tensorflow as tf

DATASET_DIR = pathlib.Path("/dev/null")

IMAGE_DIR = DATASET_DIR / "validation_images"
IMAGE_CHANNELS = 3
IMAGE_EXTENSION = ".jpg"

MASK_DIR = DATASET_DIR / "validation_masks"
MASK_EXTENSIOn = ".bin"

SHUFFLING_BUFFER_SIZE = 64


def configure_tensorflow(
    unmute_tensorflow: bool, use_xla: bool, use_mixed_precision: bool
) -> None:
    if unmute_tensorflow and "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]

    if use_xla:
        tf.config.optimizer.set_jit("autoclustering")
    else:
        os.environ.pop("TF_XLA_FLAGS", None)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


@functools.cache
def get_gitignored_dir() -> pathlib.Path:
    raise NotImplementedError()


def get_output_dir() -> pathlib.Path:
    return get_gitignored_dir() / "semantic_segmentation"


def get_image_path(img_id: str) -> str:
    path = (IMAGE_DIR / img_id).with_suffix(".jpg")
    assert path.is_file()
    return str(path)


def load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)

    # must set channels to 3 because the dataset contains grayscale images
    return tf.io.decode_jpeg(raw, channels=IMAGE_CHANNELS)


def get_mask_path(img_id: str) -> str:
    path = (MASK_DIR / img_id).with_suffix(".bin")
    assert path.is_file()
    return str(path)


def load_mask(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    return tf.io.decode_raw(raw, out_type=tf.uint8)


def unflatten_mask(img: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Masks are stored flattened.
    This function "unflatten" them to match the image shape.
    """

    # ensure we are not operating on batches
    img_shape = tf.shape(img)
    assert len(img_shape) == 3  # height, width, depth

    mask_shape = img_shape[:2]
    reshaped_mask = tf.reshape(mask, mask_shape)
    return img, reshaped_mask


def load_classes_ids() -> dict[str, int]:
    classes_ids = (
        pd.read_csv(DATASET_DIR / "categories.csv")
        .set_index("name")["new_id"]
        .to_dict()
    )
    assert isinstance(classes_ids, dict)
    return classes_ids


def pad_image_and_mask(
    img: tf.Tensor,
    mask: tf.Tensor,
    size: int = 256,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Ensures that img and mask have shapes with `height` and `width` divisible by two.
    We do this because otherwise upsampling becomes painful.
    """

    new_img = tf.image.resize_with_pad(img, size, size)
    new_mask = tf.image.resize_with_pad(mask, size, size)
    return new_img, new_mask


def load_dataset(class_count: int) -> tf.data.Dataset:
    configure_tensorflow(
        unmute_tensorflow=False, use_xla=True, use_mixed_precision=True
    )

    image_filenames = sorted(path.stem for path in IMAGE_DIR.iterdir())

    image_paths = [get_image_path(id) for id in image_filenames]
    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    mask_paths = [get_mask_path(id) for id in image_filenames]
    masks_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(mask_paths).map(
        load_mask,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_dataset: tf.data.Dataset = (
        tf.data.Dataset.zip((image_ds, masks_ds))
        .map(
            unflatten_mask,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            lambda img, mask: (img, tf.one_hot(mask, class_count, dtype=tf.uint8)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            pad_image_and_mask,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )

    return validation_dataset


def create_model(class_count: int = 81) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, None, IMAGE_CHANNELS))

    rescaled = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, padding="same", activation="relu"
    )(rescaled)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

    x = tf.keras.layers.UpSampling2D(2)(x)

    outputs = tf.keras.layers.Conv2D(
        filters=class_count,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="softmax",
        name="pred",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=tf.keras.metrics.MeanIoU(class_count),
    )
    return model


def vgg16(class_count: int, l2: float = 0, dropout: float = 0) -> tf.keras.Model:
    """Convolutionized VGG16 network.
    Args:
      l2 (float): L2 regularization strength
      dropout (float): Dropout rate
    Returns:
      (keras Model)
    """
    # Input
    input_layer = tf.keras.Input(shape=(None, None, 3), batch_size=1, name="input")
    # Preprocessing
    # x = tf.keras.layers.Lambda(
    #     tf.keras.applications.vgg16.preprocess_input, name="preprocessing"
    # )(input_layer)
    x = input_layer
    # Block 1
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block1_conv1",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block1_conv2",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="block1_pool"
    )(x)
    #  Block 2
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block2_conv1",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block2_conv2",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="block2_pool"
    )(x)
    # Block 3
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block3_conv1",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block3_conv2",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block3_conv3",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="block3_pool"
    )(x)
    # Block 4
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block4_conv1",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block4_conv2",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block4_conv3",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="block4_pool"
    )(x)
    # Block 5
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block5_conv1",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block5_conv2",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="block5_conv3",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="block5_pool"
    )(x)
    # Convolutionized fully-connected layers
    x = tf.keras.layers.Conv2D(
        filters=4096,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="conv6",
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout, name="drop6")(x)
    x = tf.keras.layers.Conv2D(
        filters=4096,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
        name="conv7",
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout, name="drop7")(x)
    # Inference layer
    x = tf.keras.layers.Conv2D(
        filters=class_count,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="softmax",
        name="pred",
    )(x)

    model = tf.keras.Model(input_layer, x)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
    )
    return model


def main() -> None:
    configure_tensorflow(
        unmute_tensorflow=False,
        use_xla=True,
        use_mixed_precision=True,
    )

    classes_ids = load_classes_ids()
    class_count = len(classes_ids)

    dataset = load_dataset(class_count)
    dataset = (
        # dataset.cache()
        dataset.shuffle(buffer_size=SHUFFLING_BUFFER_SIZE)
        # .batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )

    for img, mask in dataset:
        img_np = img.numpy()
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        print(img_cv.shape)
        # cv2.imwrite()
        raise NotImplementedError()

    # # model = vgg16(class_count)
    # model = create_model(class_count)
    # model.fit(dataset)


if __name__ == "__main__":
    main()
