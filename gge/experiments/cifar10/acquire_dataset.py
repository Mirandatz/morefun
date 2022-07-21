import pathlib
import typing

import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger

DATASET_NAME = "cifar10"

PARTITIONS_DIR = pathlib.Path(__file__).parent / "partitions"

TRAIN_DIR = PARTITIONS_DIR / "train"
VALIDATION_DIR = PARTITIONS_DIR / "validation"
TEST_DIR = PARTITIONS_DIR / "test"

VALIDATION_RATIO = 0.25


def save_instance(instance: tf.Tensor, path: pathlib.Path) -> None:
    instance_as_np = instance.numpy()
    colors_fixed = cv2.cvtColor(
        src=instance_as_np,
        code=cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(
        filename=str(path),
        img=colors_fixed,
    )


def save_partition(
    partition_name: typing.Literal["train", "test"],
    output_dir: pathlib.Path,
) -> None:
    ds = tfds.load(
        DATASET_NAME,
        split=partition_name,
        as_supervised=True,
        data_dir=pathlib.Path("/dev") / "shm" / "tensorflow_datasets" / DATASET_NAME,
    ).cache()
    for index, (instance_data, instance_label) in enumerate(ds):
        class_dir = output_dir / str(instance_label.numpy())
        class_dir.mkdir(parents=True, exist_ok=True)
        instance_path = class_dir / f"{index}.png"
        save_instance(instance_data, instance_path)
        logger.info(instance_path)


def move_files(
    source: pathlib.Path, destination: pathlib.Path, ratio_to_move: float
) -> None:
    assert source.is_dir()
    assert destination.is_dir()
    assert 0 < ratio_to_move < 1

    src_files = list(source.iterdir())
    assert len(src_files) > 0
    assert all(f.is_file() for f in src_files)

    num_to_move = len(src_files) * ratio_to_move
    files_to_move = src_files[-int(num_to_move) :]
    assert len(files_to_move) > 0

    # ensure consistent splits between runs
    sorted_files = sorted(files_to_move, key=lambda path: path.with_suffix("").name)

    for f in sorted_files:
        f.rename(destination / f.name)


def split_partition(
    source: pathlib.Path, destination: pathlib.Path, ratio_to_move: float
) -> None:
    assert source.is_dir()
    assert destination.is_dir()
    assert 0 < ratio_to_move < 1

    for src_class_dir in source.iterdir():
        assert src_class_dir.is_dir()

        dst_class_dir = destination / src_class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        move_files(
            source=src_class_dir,
            destination=dst_class_dir,
            ratio_to_move=ratio_to_move,
        )


def main() -> None:
    assert not TRAIN_DIR.exists()
    assert not VALIDATION_DIR.exists()
    assert not TEST_DIR.exists()

    TRAIN_DIR.mkdir(parents=True, exist_ok=False)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    TEST_DIR.mkdir(parents=True, exist_ok=False)

    save_partition(partition_name="train", output_dir=TRAIN_DIR)
    split_partition(
        source=TRAIN_DIR,
        destination=VALIDATION_DIR,
        ratio_to_move=VALIDATION_RATIO,
    )

    save_partition(partition_name="test", output_dir=TEST_DIR)


if __name__ == "__main__":
    main()
