import contextlib
import time
import typing

from loguru import logger


@contextlib.contextmanager
def millisecond_stopwatch(description: str) -> typing.Iterator[None]:
    start = time.perf_counter()
    yield
    diff = time.perf_counter() - start
    diff_ms = diff * 1000
    logger.info(f"{description}: {diff_ms:.2f} milliseconds")
