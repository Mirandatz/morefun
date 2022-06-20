import typing

import pytest
import tensorflow as tf
from loguru import logger


@pytest.fixture(autouse=True)
def remove_logger_sinks() -> None:
    logger.remove()


@pytest.fixture(autouse=True)
def hide_gpu_from_tensorflow() -> typing.Generator[None, None, None]:
    with tf.device("/device:CPU:0"):
        yield None
