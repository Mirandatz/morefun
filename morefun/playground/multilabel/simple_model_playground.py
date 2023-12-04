import os

import keras
import tensorflow as tf

from morefun.datasets.load_and_preprocess import cache_shuffle_batch_prefetch
from morefun.datasets.yeast import load_test, load_train


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


def main() -> None:
    config_env(
        unmute_tensorflow=False,
        use_xla=True,
        use_mixed_precision=True,
    )

    batch_size = 256

    train = load_train()
    features_spec, target_spec = train.element_spec
    n_inputs: int = features_spec.shape[0]
    n_outputs: int = target_spec.shape[0]

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
        cache_shuffle_batch_prefetch(train, batch_size),
        epochs=999,
        callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=6)],
    )
    metrics_names = model.metrics_names
    metrics_results = model.evaluate(
        cache_shuffle_batch_prefetch(load_test(), batch_size)
    )
    print(dict(zip(metrics_names, metrics_results)))


if __name__ == "__main__":
    main()
