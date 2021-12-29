import dataclasses
import enum

import typeguard


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Fork:
    name: str

    def __post_init__(self) -> None:
        assert self.name


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Merge:
    name: str

    def __post_init__(self) -> None:
        assert self.name


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Conv2D:
    name: str
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.name
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


class PoolType(enum.Enum):
    MAX_POOLING = enum.auto()
    AVG_POOLING = enum.auto()


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Pool:
    name: str
    pooling_type: PoolType
    stride: int

    def _post_init__(self) -> None:
        assert self.name
        assert self.stride > 0


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class BatchNorm:
    name: str

    def __post_init__(self) -> None:
        assert self.name


Layer = Conv2D | Pool | BatchNorm | Fork | Merge
