from pathlib import Path

import tensorflow as tf

from morefun.datasets.load_and_preprocess import (
    check_integrity,
    load_arff_into_tf_dataset,
)

NUM_TRAIN_INSTANCES = 1500
NUM_TEST_INSTANCES = 917

FEATURE_SHAPE = 103
TARGET_SHAPE = 14


def load_train() -> tf.data.Dataset:
    arff_path = Path("/app/datasets/multi-label/yeast/decompressed/yeast-train.arff")
    expected_md5 = "a1c8eeb7cbaf8ae140c5106ff84db332"
    check_integrity(arff_path, expected_md5)
    ds = load_arff_into_tf_dataset(arff_path)
    assert ds.cardinality().numpy() == NUM_TRAIN_INSTANCES
    features_spec, target_spec = ds.element_spec
    assert features_spec.shape == FEATURE_SHAPE
    assert target_spec.shape == TARGET_SHAPE
    return ds


def load_test() -> tf.data.Dataset:
    arff_path = Path("/app/datasets/multi-label/yeast/decompressed/yeast-test.arff")
    expected_md5 = "c85f3dfd6508efa5402762849e1f9dc4"
    check_integrity(arff_path, expected_md5)
    ds = load_arff_into_tf_dataset(arff_path)
    assert ds.cardinality().numpy() == NUM_TEST_INSTANCES
    features_spec, target_spec = ds.element_spec
    assert features_spec.shape == FEATURE_SHAPE
    assert target_spec.shape == TARGET_SHAPE
    return ds
