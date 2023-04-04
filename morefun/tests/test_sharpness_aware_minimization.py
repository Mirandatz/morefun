import pathlib
import tempfile

import tensorflow as tf

import morefun.neural_networks.sharpness_aware_minimization as sam


def test_serialization() -> None:
    layers = [tf.keras.layers.Dense(units=1, kernel_initializer="zeros")]
    model = sam.SharpnessAwareMinimization(tf.keras.Sequential(layers))  # type: ignore

    model.compile(optimizer="adam", loss="mean_absolute_error")

    features = tf.ones(shape=(1, 3))
    targets = tf.ones(shape=(1, 1))

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _tmp:
        model_path = pathlib.Path(_tmp) / "test_sam_serialization"

        p1 = model(features)

        tf.keras.models.save_model(model, model_path)

        model.fit(features, targets, epochs=1)

        p2 = model(features)

        tf.keras.models.load_model(model_path)

        p3 = model(features)

        assert tf.reduce_all(tf.equal(p1, p3))
        assert not tf.reduce_all(tf.equal(p1, p2))
