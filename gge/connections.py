import enum
import typing

import numpy as np
import numpy.typing as npt
import typeguard

from gge.randomness import RNG

MIN_NODE_COUNT = 2
"""
We could allow backbones with a single node,
but forcing the minimum to 2 simplifies the code a lot.
"""


AdjacencyMatrix = typing.NewType("AdjacencyMatrix", npt.NDArray[np.bool8])


@enum.unique
class MergeMethod(enum.Enum):

    # effectively disable "multiconnections", keeping only the backbone
    NO_MERGE = enum.auto()

    ADD = enum.auto()
    MULTIPLY = enum.auto()
    CONCAT = enum.auto()


@enum.unique
class ReshapeMethod(enum.Enum):
    DOWNSAMPLE = enum.auto()
    UPSAMPLE = enum.auto()


@typeguard.typechecked
class ConnectionsBlueprint:
    def __init__(self, node_count: int, rng: RNG) -> None:
        if node_count < MIN_NODE_COUNT:
            raise ValueError(
                f"node_count must be >= {MIN_NODE_COUNT}. value was {node_count}"
            )

        self._matrix = create_adjacency_matrix(node_count, rng)
        self._merge_methods = create_merge_methods(node_count, rng)
        self._reshape_methods = create_reshape_methods(node_count, rng)

        self._hash = hash(
            (
                tuple(self._matrix.flatten()),
                self._merge_methods,
                self._reshape_methods,
            ),
        )

    @property
    def matrix(self) -> AdjacencyMatrix:
        copy = self._matrix.copy()
        return AdjacencyMatrix(copy)

    @property
    def merge_methods(self) -> tuple[MergeMethod, ...]:
        return self._merge_methods

    @property
    def reshape_methods(self) -> tuple[ReshapeMethod, ...]:
        return self._reshape_methods

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if id(self) == id(other):
            return True

        if not isinstance(other, ConnectionsBlueprint):
            return NotImplemented

        if self._reshape_methods != other._reshape_methods:
            return False

        if self._merge_methods != other._merge_methods:
            return False

        return np.array_equal(self._matrix, other._matrix)


def create_merge_methods(node_count: int, rng: RNG) -> tuple[MergeMethod, ...]:
    if node_count < MIN_NODE_COUNT:
        raise ValueError(
            f"node_count must be >= {MIN_NODE_COUNT}. value was {node_count}"
        )

    valid_values = np.asarray(MergeMethod)
    chosen = rng.choice(
        a=valid_values,
        size=node_count,
        replace=True,
    )

    return tuple(chosen)


def create_reshape_methods(node_count: int, rng: RNG) -> tuple[ReshapeMethod, ...]:
    if node_count < MIN_NODE_COUNT:
        raise ValueError(
            f"node_count must be >= {MIN_NODE_COUNT}. value was {node_count}"
        )

    valid_values = np.asarray(ReshapeMethod)
    chosen = rng.choice(
        a=valid_values,
        size=node_count,
        replace=True,
    )

    return tuple(chosen)


def create_adjacency_matrix(size: int, rng: RNG) -> AdjacencyMatrix:
    if size < 2:
        raise ValueError(f"size must be >= 2. value={size}")

    backbone_connections = np.eye(N=size, k=1)

    random_connections = rng.integers(
        low=0,
        high=1,
        endpoint=True,
        size=(size, size),
    )

    # k=2 removes self-loops (k=1) and preserves the backbone connections
    ribcage_connections = np.triu(random_connections, k=2)  # type: ignore

    matrix = np.logical_or(backbone_connections, ribcage_connections)
    matrix.flags.writeable = False

    return AdjacencyMatrix(matrix)
