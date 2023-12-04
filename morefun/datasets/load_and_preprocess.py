from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io.arff import loadarff

from morefun.randomness import get_fixed_seed


def check_integrity(path: Path, expected_md5_hex_digest: str) -> bytes:
    content = path.read_bytes()
    actual_md5 = md5(content).hexdigest()
    if actual_md5 != expected_md5_hex_digest:
        raise ValueError(
            f"md5 mismatch. expected={expected_md5_hex_digest}, actual={actual_md5}"
        )


def load_arff_into_tf_dataset(path: Path) -> tf.data.Dataset:
    data, metadata = loadarff(path)

    df = pd.DataFrame(data)

    # fix targets
    fixed_targets = df.replace({b"0": 0, b"1": 1})

    column_names = list(fixed_targets.columns)
    features_names = [c for c in column_names if "Att" in c]
    targets_names = [c for c in column_names if "Class" in c]
    assert set(features_names).isdisjoint(targets_names)
    assert set(features_names) | set(targets_names) == set(column_names)

    features = df[features_names].astype(np.float32)
    targets = df[targets_names].replace({b"0": 0, b"1": 1}).astype(np.bool_)

    return tf.data.Dataset.from_tensor_slices((features, targets))


def cache_shuffle_batch_prefetch(
    ds: tf.data.Dataset,
    batch_size: int,
    rng_seed: int = get_fixed_seed(),
) -> tf.data.Dataset:
    return (
        ds.shuffle(
            ds.cardinality(),
            seed=rng_seed,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
