import dataclasses
import enum
import itertools

import gge.backbones as bb
import gge.layers as gl
import gge.randomness as rand


@enum.unique
class MergeStrategy(enum.Enum):

    # effectively disable "multiconnections", keeping only the backbone
    NO_MERGE = enum.auto()
    ADD = enum.auto()
    MULTIPLY = enum.auto()
    CONCAT = enum.auto()


@enum.unique
class ReshapeStrategy(enum.Enum):
    DOWNSAMPLE = enum.auto()
    UPSAMPLE = enum.auto()


@dataclasses.dataclass(frozen=True)
class MergeLayer:
    forks_mask: tuple[bool, ...]
    merge_strategy: MergeStrategy
    reshape_strategy: ReshapeStrategy

    def __post_init__(self) -> None:
        assert isinstance(self.forks_mask, tuple)
        for value in self.forks_mask:
            assert isinstance(value, bool)

        assert isinstance(self.merge_strategy, MergeStrategy)
        assert isinstance(self.reshape_strategy, ReshapeStrategy)


@dataclasses.dataclass(frozen=True)
class ConnectionsSchema:
    merge_layers: tuple[MergeLayer, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.merge_layers, tuple)
        for ml in self.merge_layers:
            assert isinstance(ml, MergeLayer)


def extract_forks_masks_lengths(backbone: bb.Backbone) -> tuple[int, ...]:
    lengths = []

    fork_count = 0

    for layer in backbone.layers:
        if isinstance(layer, gl.Fork):
            fork_count += 1

        if isinstance(layer, gl.Merge):
            lengths.append(fork_count)

    return tuple(lengths)


def create_inputs_merger(mask_len: int, rng: rand.RNG) -> MergeLayer:
    assert mask_len >= 0

    mask = rng.integers(
        low=0,
        high=1,
        endpoint=True,
        size=mask_len,
        dtype=bool,
    )
    mask_as_bools = tuple(bool(b) for b in mask)
    merge_strat = rng.choice(list(MergeStrategy))  # type: ignore
    reshape_strat = rng.choice(list(ReshapeStrategy))  # type: ignore

    return MergeLayer(
        forks_mask=mask_as_bools,
        merge_strategy=merge_strat,
        reshape_strategy=reshape_strat,
    )


def create_connections_schema(
    backbone: bb.Backbone, rng: rand.RNG
) -> ConnectionsSchema:
    masks_lens = extract_forks_masks_lengths(backbone)
    merge_layers = (create_inputs_merger(ml, rng) for ml in masks_lens)
    return ConnectionsSchema(tuple(merge_layers))


def collect_sources(backbone: bb.Backbone) -> list[gl.Layer]:
    return [
        source
        for source, next in itertools.pairwise(backbone.layers)
        if isinstance(next, gl.Fork)
    ]
