from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io.arff import loadarff


def check_integrity(path: Path, expected_md5_hex_digest: str) -> bytes:
    content = path.read_bytes()
    actual_md5 = md5(content).hexdigest()
    if actual_md5 != expected_md5_hex_digest:
        raise ValueError(
            f"md5 mismatch. expected={expected_md5_hex_digest}, actual={actual_md5}"
        )


def _load_yeast_arff(path: Path) -> tf.data.Dataset:
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


def load_train() -> tf.data.Dataset:
    arff_path = Path("/app/datasets/multi-label/yeast/decompressed/yeast-train.arff")
    expected_md5 = "a1c8eeb7cbaf8ae140c5106ff84db332"
    check_integrity(arff_path, expected_md5)
    return _load_yeast_arff(arff_path)


def load_test() -> tf.data.Dataset:
    arff_path = Path("/app/datasets/multi-label/yeast/decompressed/yeast-test.arff")
    expected_md5 = "c85f3dfd6508efa5402762849e1f9dc4"
    check_integrity(arff_path, expected_md5)
    return _load_yeast_arff(arff_path)
