import pathlib
import re

import diskcache
import tensorflow as tf
import tensorflow_addons as tfa

# import gge.experiments.settings as gset
import gge.fitnesses as gf
import gge.layers as gl
import gge.persistence
import gge.phenotypes
import gge.randomness
import gge.redirection

BATCH_SIZE = 64
MAX_EPOCHS = 2048


GIT_IGNORED_DIRECTORY = pathlib.Path(__file__).parent / "gitignored"

CACHE_DIR = GIT_IGNORED_DIRECTORY / "cache"
CACHE_ENTRY_EXPIRATION_TIME_SEC = 2**31
CACHE = diskcache.Cache(CACHE_DIR)
memoize = CACHE.memoize(expire=CACHE_ENTRY_EXPIRATION_TIME_SEC)

DATASETS_DIR = pathlib.Path("/gge/datasets")
INPUT_SHAPE = gl.Shape(height=32, width=32, depth=3)

CIFAR10_RESULTS = GIT_IGNORED_DIRECTORY / "cifar10_results"
LAST_DITCH_CIFAR_10_RESULTS = GIT_IGNORED_DIRECTORY / "last_ditch_cifar10"

CIFAR100_RESULTS = GIT_IGNORED_DIRECTORY / "cifar100_results"
LAST_DITCH_CIFAR_100_RESULTS = GIT_IGNORED_DIRECTORY / "last_ditch_cifar100"


def get_train_dataset(
    train_directory: pathlib.Path,
    validation_directory: pathlib.Path,
    input_shape: gl.Shape,
    batch_size: int,
) -> tf.data.Dataset:
    just_train = gf.load_dataset_and_rescale(
        directory=train_directory,
        input_shape=input_shape,
    )

    just_validation = gf.load_dataset_and_rescale(
        directory=validation_directory,
        input_shape=input_shape,
    )

    concatenated = just_train.concatenate(just_validation)

    cached = concatenated.cache()

    shuffled = cached.shuffle(
        buffer_size=cached.cardinality().numpy(),
        seed=gge.randomness.get_fixed_seed(),
        reshuffle_each_iteration=True,
    )

    batched = shuffled.batch(batch_size, drop_remainder=True)

    prefetched = batched.prefetch(tf.data.AUTOTUNE)

    return prefetched


def get_test_dataset(
    test_directory: pathlib.Path,
    input_shape: gl.Shape,
    batch_size: int,
) -> tf.data.Dataset:
    return gf.load_non_train_partition(
        input_shape=input_shape,
        batch_size=batch_size,
        directory=test_directory,
    )


def load_model_from_json_path(json_path: pathlib.Path) -> tf.keras.Model:
    model = tf.keras.models.model_from_json(json_path.read_text())
    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    model.compile(
        optimizer=ranger,
        loss="categorical_crossentropy",
        metrics="accuracy",
    )
    return model


@memoize  # type: ignore
def calc_model_test_accuracy(
    model_json_path: pathlib.Path,
    input_shape: gl.Shape,
    batch_size: int,
    max_epochs: int,
    train_dataset_path: pathlib.Path,
    validation_dataset_path: pathlib.Path,
    test_dataset_path: pathlib.Path,
) -> float:

    model = load_model_from_json_path(model_json_path)

    with gge.redirection.discard_stderr_and_stdout():
        train_dataset = get_train_dataset(
            train_directory=train_dataset_path,
            validation_directory=validation_dataset_path,
            input_shape=input_shape,
            batch_size=batch_size,
        )

    model.fit(
        train_dataset,
        epochs=max_epochs,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=40,
                restore_best_weights=True,
            )
        ],
    )

    with gge.redirection.discard_stderr_and_stdout():
        test_dataset = get_test_dataset(
            test_directory=test_dataset_path,
            input_shape=input_shape,
            batch_size=batch_size,
        )

    loss, accuracy = model.evaluate(test_dataset)

    assert isinstance(accuracy, float)
    return accuracy


def _get_dataset_name(path: pathlib.Path) -> str:
    if "cifar100" in str(path):
        return "cifar100"

    elif "cifar10" in str(path):
        return "cifar10"

    else:
        raise ValueError(f"wat: {path}")


def _get_run_number(path: pathlib.Path) -> int:
    seed_nr = re.findall(
        pattern=r"seed_(\d)",
        string=str(path),
    )
    assert len(seed_nr) == 1
    return int(seed_nr[0])


def _is_last_ditch(path: pathlib.Path) -> bool:
    return "last_ditch" in str(path)


def _get_genotype_uuid(path: pathlib.Path) -> str:
    return path.stem


def _get_dataset_paths(
    dataset_name: str,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:

    match dataset_name:
        case "cifar10":
            dataset_dir = DATASETS_DIR / "cifar10"

        case "cifar100":
            dataset_dir = DATASETS_DIR / "cifar100"

        case _:
            raise ValueError(f"wat: {dataset_name}")

    train = dataset_dir / "train"
    validation = dataset_dir / "validation"
    test = dataset_dir / "test"

    return train, validation, test


def _get_test_accuracy(model_json_path: pathlib.Path) -> float:
    dataset_name = _get_dataset_name(model_json_path)
    train, validation, test = _get_dataset_paths(dataset_name)

    acc = calc_model_test_accuracy(
        model_json_path=model_json_path,
        input_shape=INPUT_SHAPE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        train_dataset_path=train,
        validation_dataset_path=validation,
        test_dataset_path=test,
    )

    assert isinstance(acc, float)

    return acc


def _get_number_of_params(model_json_path: pathlib.Path) -> int:
    model = load_model_from_json_path(model_json_path)
    param_count = model.count_params()
    assert isinstance(param_count, int)
    return param_count


def main() -> None:
    for json_path in GIT_IGNORED_DIRECTORY.rglob("*.json"):
        print(
            _get_dataset_name(json_path),
            _is_last_ditch(json_path),
            _get_run_number(json_path),
            _get_genotype_uuid(json_path),
            _get_test_accuracy(json_path),
            _get_number_of_params(json_path),
            sep=",",
        )


if __name__ == "__main__":
    main()
