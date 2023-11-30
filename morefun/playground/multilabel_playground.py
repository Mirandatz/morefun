import os
from typing import Literal

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io.arff import loadarff

import morefun.randomness as rand


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


def load_yeast(
    partition: Literal["all", "train", "test"],
    batch_size: int,
) -> tf.data.Dataset:
    # https://mulan.sourceforge.net/datasets-mlc.html
    match partition:
        case "all":
            path = "morefun/playground/input/yeast.arff"

        case "train":
            path = "morefun/playground/input/yeast-train.arff"

        case "test":
            path = "morefun/playground/input/yeast-test.arff"
        case _:
            raise ValueError(f"unknown partition={partition}")

    dataset, metadata = loadarff(path)
    df = pd.DataFrame(dataset)

    column_names = df.columns
    features_names = [c for c in column_names if "Att" in c]
    targets_names = [c for c in column_names if "Class" in c]
    assert set(features_names).isdisjoint(targets_names)
    assert set(features_names) | set(targets_names) == set(column_names)

    features = df[features_names].astype(np.float32)
    targets = df[targets_names].replace({b"0": 0, b"1": 1})

    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    return (
        ds.cache()
        .shuffle(
            ds.cardinality(),
            seed=rand.get_fixed_seed(),
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def main() -> None:
    config_env(
        unmute_tensorflow=False,
        use_xla=True,
        use_mixed_precision=True,
    )

    dataset = load_yeast("train", batch_size=128)

    features_spec, target_spec = dataset.element_spec
    n_inputs: int = features_spec.shape[1]
    n_outputs: int = target_spec.shape[1]

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_dim=n_inputs, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(80, input_dim=n_inputs, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(n_outputs, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[keras.metrics.Precision(), keras.metrics.Recall()],
    )

    model.fit(
        dataset,
        epochs=999,
        callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=6)],
    )
    metrics_names = model.metrics_names
    metrics_results = model.evaluate(dataset)
    print(dict(zip(metrics_names, metrics_results)))


if __name__ == "__main__":
    main()
