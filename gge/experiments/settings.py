import pathlib
import sys
import typing

import tomli
from loguru import logger

Settings = dict[str, typing.Any]

WORKDIR = pathlib.Path("/gge")
LOGGING_DIR = WORKDIR / "logging"


def read_settings_file(
    path: pathlib.Path = pathlib.Path("/gge/settings.toml"),
) -> dict[str, typing.Any]:
    with path.open("rb") as file:
        return tomli.load(file)


def configure_logger(settings: Settings) -> None:
    log_settings = settings["logging"]
    log_level = log_settings["log_level"]

    if log_level not in ["DEBUG", "INFO", "WARNING"]:
        raise ValueError(f"unknown log_level=<{log_level}>")

    LOGGING_DIR.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(sink=sys.stderr, level=log_level)
    logger.add(sink=LOGGING_DIR / "log.txt")

    # checking if we can write to both sinks
    logger.info("started logger")


# def _configure_logger(settings: dict[str, typing.Any]) -> None:
#     log_level = settings["log_level"]
#     if log_level not in ["DEBUG", "INFO", "WARNING"]:
#         raise ValueError(f"unknown log_level=<{log_level}>")

#     log_file =


#     if not log_dir.is_dir():
#         raise ValueError(f"log_dir is not a directory, path=<{log_dir}>")

#     logger.remove()
#     logger.add(sink=sys.stderr, level=log_level)
#     logger.add(sink=log_dir / "log.txt")

#     # checking if we can write to both sinks
#     logger.info("started logger")


# def _validate_dataset_dir(
#     path: pathlib.Path,
#     img_height: int,
#     img_width: int,
#     expected_num_instances: int,
#     expected_class_count: int,
# ) -> None:
#     import tensorflow as tf

#     with gge.redirection.discard_stderr_and_stdout():
#         with tf.device("cpu"):
#             ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
#                 directory=path,
#                 batch_size=None,
#                 image_size=(img_height, img_width),
#                 label_mode="categorical",
#                 shuffle=False,
#                 color_mode="rgb",
#             )

#             num_instances = ds.cardinality().numpy()
#             if num_instances != expected_num_instances:
#                 raise ValueError(
#                     f"unexpected number of instances. found=<{num_instances}>, expected=<{expected_num_instances}>"
#                 )

#             num_classes = len(ds.class_names)
#             if num_classes != expected_class_count:
#                 raise ValueError(
#                     f"unexpected number of classes. found=<{num_classes}>, expected=<{expected_class_count}>"
#                 )
