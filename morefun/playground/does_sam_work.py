import pathlib

import tensorflow as tf

import morefun.neural_networks.sharpness_aware_minimization as sam


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
        lambda d, t: (tf.keras.applications.resnet_v2.preprocess_input(d), t)
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
        lambda d, t: (tf.keras.applications.resnet_v2.preprocess_input(d), t)
    )

    return (
        preprocessed.cache()
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_model(
    input_shape: tuple[int, int, int], class_count: int, batch_size: int
) -> tf.keras.Model:
    base_model = tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="max",
        classes=class_count,
        classifier_activation="softmax",
    )

    model_input = tf.keras.Input(shape=input_shape, batch_size=batch_size)

    x = model_input
    # x = tf.keras.layers.Resizing(height=224, width=224)(x)
    x = tf.keras.layers.RandomFlip(mode="horizontal")(x)
    x = tf.keras.layers.RandomRotation(factor=15.0 / 360)(x)
    x = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)

    model_output = base_model(x)

    model = tf.keras.Model(inputs=model_input, outputs=model_output)

    return model


def main() -> None:
    tf.config.optimizer.set_jit("autoclustering")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    RNG_SEED = 42

    batch_size = 256
    input_shape = (224, 224, 3)
    class_count = 10

    dataset_dir = pathlib.Path("/datasets/cifar10_train_test")

    model = get_model(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=batch_size,
    )

    model = sam.SharpnessAwareMinimization(model)  # type: ignore

    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    train = load_train_partition(
        input_shape=input_shape,
        batch_size=batch_size,
        directory=dataset_dir / "train",
        rng_seed=RNG_SEED,
    )

    test = load_non_train_partition(
        input_shape=input_shape,
        batch_size=batch_size,
        directory=dataset_dir / "test",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=6,
            monitor="loss",
            restore_best_weights=True,
        )
    ]

    model.fit(train, epochs=999, callbacks=callbacks)
    loss, acc = model.evaluate(test)
    print("loss", loss)
    print("acc", acc)


if __name__ == "__main__":
    main()
