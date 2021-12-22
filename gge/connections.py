import dataclasses
import enum

import typeguard

import gge.backbones as bb
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


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class MergeLayer:
    forks_mask: tuple[bool, ...]
    merge_strategy: MergeStrategy
    reshape_strategy: ReshapeStrategy


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class ConnectionsSchema:
    merge_layers: tuple[MergeLayer, ...]


def extract_forks_masks_lengths(backbone: bb.Backbone) -> tuple[int, ...]:
    lengths = []

    fork_count = 0

    for layer in backbone.layers:
        if isinstance(layer, bb.Fork):
            fork_count += 1

        if isinstance(layer, bb.Merge):
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

    merge_strat = rng.choice(list(MergeStrategy))  # type: ignore
    reshape_strat = rng.choice(list(ReshapeStrategy))  # type: ignore

    return MergeLayer(
        forks_mask=tuple(mask),
        merge_strategy=merge_strat,
        reshape_strategy=reshape_strat,
    )


def create_connections_schema(
    backbone: bb.Backbone, rng: rand.RNG
) -> ConnectionsSchema:
    masks_lens = extract_forks_masks_lengths(backbone)
    merge_layers = (create_inputs_merger(ml, rng) for ml in masks_lens)
    return ConnectionsSchema(tuple(merge_layers))
