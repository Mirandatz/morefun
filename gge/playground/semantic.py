import functools
import os
import pathlib

import numpy as np
import tensorflow as tf

DATASET_DIR = pathlib.Path("/gge/datasets/coco")

IMAGES_DIR = DATASET_DIR / "validation_images"
MASKS_DIR = DATASET_DIR / "validation_masks"


def configure_tensorflow(
    unmute_tensorflow: bool, use_xla: bool, use_mixed_precision: bool
) -> None:

    if unmute_tensorflow and "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]

    if use_xla:
        tf.config.optimizer.set_jit("autoclustering")
    else:
        os.environ.pop("TF_XLA_FLAGS", None)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


@functools.cache
def get_project_root() -> pathlib.Path:
    this_file = pathlib.Path(__file__)
    filesystem_root = pathlib.Path(this_file.root)

    current_dir = this_file.parent

    while current_dir != filesystem_root:
        if (current_dir / ".gge_root").exists():
            return current_dir

        current_dir = current_dir.parent

    raise ValueError("unable to find `project root directory`")


@functools.cache
def get_gitignored_dir() -> pathlib.Path:
    gitignored = get_project_root() / "gge" / "playground" / "gitignored"

    if gitignored.exists():
        return gitignored

    raise ValueError("unable to find `gitignored directory`")


@functools.cache
def get_output_dir() -> pathlib.Path:
    output_dir = get_gitignored_dir() / "semantic_segmentation"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_image_path(img_id: str) -> str:
    path = (IMAGES_DIR / img_id).with_suffix(".jpg")
    assert path.is_file()
    return str(path)


def load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    return tf.io.decode_jpeg(raw)


def get_mask_path(img_id: str) -> str:
    path = (MASKS_DIR / img_id).with_suffix(".bin")
    assert path.is_file()
    return str(path)


def load_mask(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    return tf.io.decode_raw(raw, out_type=tf.uint8)


def reshape_mask(img: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # shape = height, width, num_of_channels
    mask_shape = tf.shape(img)[:2]
    reshaped_mask = tf.reshape(mask, mask_shape)
    return img, reshaped_mask


def heh() -> tf.data.Dataset:
    configure_tensorflow(
        unmute_tensorflow=False, use_xla=True, use_mixed_precision=True
    )

    image_ids = sorted(path.stem for path in IMAGES_DIR.iterdir())

    image_paths = [get_image_path(id) for id in image_ids]
    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    mask_paths = [get_mask_path(id) for id in image_ids]
    masks_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(mask_paths).map(
        load_mask,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_dataset: tf.data.Dataset = tf.data.Dataset.zip((image_ds, masks_ds)).map(
        reshape_mask,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_dataset = (
        validation_dataset.cache()
        .shuffle(buffer_size=validation_dataset.cardinality().numpy())
        .prefetch(tf.data.AUTOTUNE)
    )

    for img, mask in validation_dataset:
        print(img.shape, mask.shape)
        exit(0)

    # mask_ds = make_masks_dataset(image_ids)
    # for mask in mask_ds:
    #     print(mask.shape)

    # for img, mask in tf.data.Dataset.zip((image_ds, mask_ds)):
    #     print(img.shape)
    #     print(mask.shape)
    #     break


def main() -> None:
    heh()

    num_columns = 999
    num_rows = 123

    rng = np.random.default_rng(seed=42)
    original_arr = (
        rng.integers(
            low=-10,
            high=10,
            size=num_columns * num_rows,
        )
        .astype(np.int32)
        .reshape(num_columns, num_rows)
    )

    import tempfile

    with tempfile.NamedTemporaryFile(dir="/dev/shm") as file:
        original_arr.tofile(file.name)

        raw_tensor = tf.io.read_file(file.name)
        decoded_tensor = tf.io.decode_raw(raw_tensor, out_type=tf.int32)
        reshaped_tensor = tf.reshape(decoded_tensor, shape=(num_columns, num_rows))
        tensor_as_np = reshaped_tensor.numpy()

        assert np.array_equal(tensor_as_np, original_arr)

    if original_arr is not None:
        return


if __name__ == "__main__":
    main()
