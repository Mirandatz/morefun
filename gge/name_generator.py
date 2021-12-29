import collections


class NameGenerator:
    def __init__(self) -> None:
        self._instance_counter: collections.Counter[str] = collections.Counter()

    def create_name(self, prefix: str) -> str:
        assert prefix

        count = self._instance_counter[prefix]
        name = f"{prefix}_{count}"
        self._instance_counter[prefix] += 1
        return name
