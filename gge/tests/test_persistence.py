import pathlib
import tempfile

import numpy as np
import tensorflow as tf

import gge.persistence as gp

# auto-import fixture
from gge.tests.fixtures import hide_gpu_from_tensorflow  # noqa


def test_save_and_load_tf_model() -> None:
    """Can save and load a model."""

    # dummy model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")

    # dummy data
    X = np.zeros(shape=(3, 5))
    y = np.zeros(shape=(3, 1))

    # update models weights
    model.fit(X, y, epochs=1, verbose=0)

    original_prediction = model.predict(X)

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _dir:
        path = pathlib.Path(_dir) / "wahetv"
        gp.save_and_zip_tf_model(model, path)
        loaded_model = gp.load_zipped_tf_model(path)

    new_prediction = loaded_model.predict(X)

    assert np.array_equal(new_prediction, original_prediction)
