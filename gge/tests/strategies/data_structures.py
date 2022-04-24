import typing

import attrs

T = typing.TypeVar("T")


@attrs.frozen
class ParsingTestData(typing.Generic[T]):
    tokenstream: str
    parsed: T
