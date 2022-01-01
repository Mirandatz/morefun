import collections


class NameGenerator:
    def __init__(self) -> None:
        self._instance_counter: collections.Counter[str] = collections.Counter()

    def _create_name(self, prefix: str) -> str:
        assert prefix

        count = self._instance_counter[prefix]
        name = f"{prefix}_{count}"
        self._instance_counter[prefix] += 1
        return name

    def create_name(self, prefix_or_type: str | type) -> str:
        if isinstance(prefix_or_type, str):
            return self._create_name(prefix_or_type)

        elif isinstance(prefix_or_type, type):
            return self._create_name(prefix_or_type.__name__)

        else:
            raise ValueError("prefix_or_type must be a str or a type")
