import collections

from loguru import logger


class NameGenerator:
    def __init__(self) -> None:
        self._instance_counter: collections.Counter[str] = collections.Counter()

    def _create_name(self, prefix: str) -> str:
        assert prefix

        count = self._instance_counter[prefix]
        name = f"{prefix}_{count}"
        self._instance_counter[prefix] += 1

        logger.debug(f"Generated name=<{name}> for prefix=<{prefix}>")
        return name

    def gen_name(self, prefix_or_type: str | type) -> str:
        if isinstance(prefix_or_type, str):
            return self._create_name(prefix_or_type)
        else:
            return self._create_name(prefix_or_type.__name__)
