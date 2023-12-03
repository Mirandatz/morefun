import os
import pathlib

import tensorflow as tf


def config_env(
    unmute_tensorflow: bool,
    use_xla: bool,
    use_mixed_precision: bool,
) -> None:
    if unmute_tensorflow and "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]

    if use_xla:
        tf.config.optimizer.set_jit("autoclustering")
    else:
        os.environ.pop("TF_XLA_FLAGS", None)
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def load_train_partition(
    input_shape: tuple[int, int, int],
    batch_size: int,
    directory: pathlib.Path,
    rng_seed: int,
) -> tf.data.Dataset:
    train: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        batch_size=None,
        image_size=(input_shape[0], input_shape[1]),
        label_mode="categorical",
        shuffle=False,
        color_mode="rgb",
    )

    preprocessed = train.map(
        lambda d, t: (tf.keras.applications.vgg16.preprocess_input(d), t)
    )

    return (
        preprocessed.cache()
        .shuffle(
            buffer_size=preprocessed.cardinality().numpy(),
            seed=rng_seed,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_non_train_partition(
    input_shape: tuple[int, int, int],
    batch_size: int,
    directory: pathlib.Path,
) -> tf.data.Dataset:
    test: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        batch_size=None,
        image_size=(input_shape[0], input_shape[1]),
        label_mode="categorical",
        shuffle=False,
        color_mode="rgb",
    )

    preprocessed = test.map(
        lambda d, t: (tf.keras.applications.vgg16.preprocess_input(d), t)
    )

    return (
        preprocessed.cache()
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_model(
    input_shape: tuple[int, int, int],
    batch_size: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size)

    data_aug = tf.keras.layers.RandomFlip("horizontal")(inputs)
    data_aug = tf.keras.layers.RandomRotation(factor=15 / 360)(data_aug)
    data_aug = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(
        data_aug
    )

    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="max",
        classes=10,
    )

    outputs = base_model(data_aug)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    return model


def main() -> None:
    config_env(unmute_tensorflow=False, use_xla=True, use_mixed_precision=True)

    input_shape = (32, 32, 3)
    batch_size = 512
    rng_seed = 42
    dataset_dir = pathlib.Path("/app/datasets/cifar10")

    model = get_model(input_shape, batch_size)

    train = load_train_partition(
        input_shape=input_shape,
        batch_size=batch_size,
        directory=dataset_dir / "train",
        rng_seed=rng_seed,
    )

    test = load_non_train_partition(
        input_shape=input_shape,
        batch_size=batch_size,
        directory=dataset_dir / "test",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=6, monitor="loss", restore_best_weights=True
        )
    ]
    model.fit(train, epochs=999, callbacks=callbacks, verbose=1)

    model.evaluate(test)


if __name__ == "__main__":
    main()
