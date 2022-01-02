import dataclasses
import enum
import itertools

import gge.backbones as bb
import gge.layers as gl
import gge.name_generator as ng
import gge.randomness as rand


@enum.unique
class ReshapeStrategy(enum.Enum):
    DOWNSAMPLE = enum.auto()
    UPSAMPLE = enum.auto()


@enum.unique
class MergeStrategy(enum.Enum):
    ADD = enum.auto()
    CONCAT = enum.auto()


@dataclasses.dataclass(frozen=True)
class MergeParameters:
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
    merge_layers: tuple[MergeParameters, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.merge_layers, tuple)
        for ml in self.merge_layers:
            assert isinstance(ml, MergeParameters)


def extract_forks_masks_lengths(backbone: bb.Backbone) -> tuple[int, ...]:
    lengths = []

    fork_count = 0

    for layer in backbone.layers:
        if not isinstance(layer, gl.MarkerLayer):
            continue

        if layer.mark_type == gl.MarkerType.FORK_POINT:
            fork_count += 1

        if layer.mark_type == gl.MarkerType.MERGE_POINT:
            lengths.append(fork_count)

    return tuple(lengths)


def create_inputs_merger(mask_len: int, rng: rand.RNG) -> MergeParameters:
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

    return MergeParameters(
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


def collect_fork_sources(backbone: bb.Backbone) -> list[gl.Layer]:
    return [
        source
        for source, next in itertools.pairwise(backbone.layers)
        if gl.is_fork_marker(next)
    ]


def downsampling_shortcut(
    source: gl.ConnectedLayer,
    target_shape: gl.Shape,
    name: str,
) -> gl.ConnectedConv2D:
    assert source.output_shape != target_shape
    assert source.output_shape.aspect_ratio == target_shape.aspect_ratio

    ratio = source.output_shape.width / target_shape.width
    stride = int(ratio)
    assert stride == ratio

    reshaper = gl.Conv2D(
        name=name,
        filter_count=target_shape.depth,
        kernel_size=1,
        stride=stride,
    )

    return gl.ConnectedConv2D(input_layer=source, params=reshaper)


def upsampling_shortcut(
    source: gl.ConnectedLayer,
    target_shape: gl.Shape,
    name: str,
) -> gl.ConnectedConv2DTranspose:
    assert source.output_shape != target_shape
    assert source.output_shape.aspect_ratio == target_shape.aspect_ratio

    ratio = target_shape.width / source.output_shape.width
    stride = int(ratio)
    assert stride == ratio

    reshaper = gl.Conv2DTranspose(
        name=name,
        filter_count=target_shape.depth,
        kernel_size=1,
        stride=stride,
    )

    return gl.ConnectedConv2DTranspose(input_layer=source, params=reshaper)


def make_shortcut(
    source: gl.ConnectedLayer,
    target_shape: gl.Shape,
    mode: ReshapeStrategy,
    name_gen: ng.NameGenerator,
) -> gl.ConnectedLayer:

    assert source.output_shape.aspect_ratio == target_shape.aspect_ratio

    if source.output_shape == target_shape:
        return source

    if mode == ReshapeStrategy.DOWNSAMPLE:
        return downsampling_shortcut(
            source=source,
            target_shape=target_shape,
            name=name_gen.gen_name(gl.Conv2D),
        )

    elif mode == ReshapeStrategy.UPSAMPLE:
        return upsampling_shortcut(
            source=source,
            target_shape=target_shape,
            name=name_gen.gen_name(gl.Conv2DTranspose),
        )

    else:
        raise ValueError(f"unexpected ReshapeStrategy: {mode}")


def select_target_shape(
    candidates: list[gl.Shape],
    mode: ReshapeStrategy,
) -> gl.Shape:
    assert len(candidates) > 1
    for a, b in itertools.pairwise(candidates):
        assert a.aspect_ratio == b.aspect_ratio

    if mode == ReshapeStrategy.DOWNSAMPLE:
        return min(candidates, key=lambda shape: shape.width)

    elif mode == ReshapeStrategy.UPSAMPLE:
        return max(candidates, key=lambda shape: shape.width)

    else:
        raise ValueError("unexpected ReshapeStrategy")


def make_merge(
    sources: list[gl.ConnectedLayer],
    reshape_strategy: ReshapeStrategy,
    merge_strategy: MergeStrategy,
    name_gen: ng.NameGenerator,
) -> gl.ConnectedMergeLayer:
    assert len(sources) >= 1

    src_shapes = [src.output_shape for src in sources]
    target_shape = select_target_shape(
        candidates=src_shapes,
        mode=reshape_strategy,
    )

    shorcuts = [
        make_shortcut(
            src,
            target_shape,
            mode=reshape_strategy,
            name_gen=name_gen,
        )
        for src in sources
    ]

    if merge_strategy == MergeStrategy.ADD:
        return gl.ConnectedAdd(tuple(shorcuts))

    elif merge_strategy == MergeStrategy.CONCAT:
        return gl.ConnectedConcatenate(tuple(shorcuts))

    else:
        raise ValueError(f"unknown MergeStrategy: {merge_strategy}")
