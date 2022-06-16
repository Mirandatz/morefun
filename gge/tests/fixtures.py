import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def remove_logger_sinks() -> None:
    logger.remove()
