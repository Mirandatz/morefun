import enum
import itertools
import typing

import attrs
from loguru import logger

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


@attrs.frozen(cache_hash=True)
class MergeParameters:
    forks_mask: tuple[bool, ...]
    merge_strategy: MergeStrategy
    reshape_strategy: ReshapeStrategy

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.forks_mask, tuple)
        for value in self.forks_mask:
            assert isinstance(value, bool)

        assert isinstance(self.merge_strategy, MergeStrategy)
        assert isinstance(self.reshape_strategy, ReshapeStrategy)


@attrs.frozen(cache_hash=True)
class ConnectionsSchema:
    merge_params: tuple[MergeParameters, ...]

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.merge_params, tuple)
        for ml in self.merge_params:
            assert isinstance(ml, MergeParameters)

    def is_empty(self) -> bool:
        return len(self.merge_params) == 0


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
    source: gl.ConnectableLayer,
    target_shape: gl.Shape,
    name: str,
) -> gl.ConnectedConv2D:
    assert source.output_shape != target_shape
    assert source.output_shape.aspect_ratio == target_shape.aspect_ratio

    stride, rest = divmod(source.output_shape.width, target_shape.width)
    assert rest == 0

    reshaper = gl.Conv2D(
        name=name,
        filter_count=target_shape.depth,
        kernel_size=1,
        stride=stride,
    )

    return gl.ConnectedConv2D(input_layer=source, params=reshaper)


def upsampling_shortcut(
    source: gl.ConnectableLayer,
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
    source: gl.ConnectableLayer,
    target_shape: gl.Shape,
    mode: ReshapeStrategy,
    name_gen: ng.NameGenerator,
) -> gl.ConnectableLayer:

    assert source.output_shape.aspect_ratio == target_shape.aspect_ratio

    if source.output_shape == target_shape:
        return source

    if mode == ReshapeStrategy.DOWNSAMPLE:
        return downsampling_shortcut(
            source=source,
            target_shape=target_shape,
            name="shortcut_" + name_gen.gen_name(gl.Conv2D),
        )

    elif mode == ReshapeStrategy.UPSAMPLE:
        return upsampling_shortcut(
            source=source,
            target_shape=target_shape,
            name="shortcut_" + name_gen.gen_name(gl.Conv2DTranspose),
        )

    else:
        raise ValueError(f"unexpected ReshapeStrategy: {mode}")


def select_target_shape(
    candidates: list[gl.Shape],
    mode: ReshapeStrategy,
) -> gl.Shape:
    assert len(candidates) >= 1
    for a, b in itertools.pairwise(candidates):
        assert a.aspect_ratio == b.aspect_ratio

    if mode == ReshapeStrategy.DOWNSAMPLE:
        return min(candidates, key=lambda shape: shape.width)

    elif mode == ReshapeStrategy.UPSAMPLE:
        return max(candidates, key=lambda shape: shape.width)

    else:
        raise ValueError("unexpected ReshapeStrategy")


def make_merge(
    sources: list[gl.ConnectableLayer],
    reshape_strategy: ReshapeStrategy,
    merge_strategy: MergeStrategy,
    name_gen: ng.NameGenerator,
) -> gl.MultiInputLayer:
    assert len(sources) > 1

    src_shapes = [src.output_shape for src in sources]
    target_shape = select_target_shape(
        candidates=src_shapes,
        mode=reshape_strategy,
    )

    logger.debug(
        f"Merging shapes=<{src_shapes}> into target=<{target_shape}>"
        f" from strategy=<{reshape_strategy}"
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
        return gl.ConnectedAdd(
            input_layers=tuple(shorcuts),
            name=name_gen.gen_name(gl.ConnectedAdd),
        )

    elif merge_strategy == MergeStrategy.CONCAT:
        return gl.ConnectedConcatenate(
            input_layers=tuple(shorcuts),
            name=name_gen.gen_name(gl.ConnectedConcatenate),
        )

    else:
        raise ValueError(f"unknown MergeStrategy: {merge_strategy}")


class StatefulLayerConnector:
    def __init__(
        self,
        input_layer: gl.Input,
        merge_params_iter: typing.Iterator[MergeParameters],
    ) -> None:
        self._merge_params_iter = merge_params_iter
        self._previous_layer: gl.ConnectableLayer = input_layer
        self._name_gen = ng.NameGenerator()
        self._fork_points: list[gl.ConnectableLayer] = []

    @property
    def previous_layer(self) -> gl.ConnectableLayer:
        return self._previous_layer

    def _connect_single_input_layer(
        self,
        layer: gl.ConvertibleToConnectableLayer,
    ) -> None:
        self._previous_layer = layer.to_connectable(self.previous_layer)

    def _register_fork_point(self) -> None:
        self._fork_points.append(self._previous_layer)

    def _get_merge_params(self) -> MergeParameters:
        return next(self._merge_params_iter)

    def _connect_multi_input_layer(self) -> None:
        merge_params = self._get_merge_params()

        assert len(self._fork_points) == len(merge_params.forks_mask)

        chosen_fork_points = itertools.compress(
            data=self._fork_points,
            selectors=merge_params.forks_mask,
        )

        sources = [self._previous_layer] + list(chosen_fork_points)
        # same as traditional `list(set(sources))` but keeps the ordering
        sources_without_duplicates_and_in_same_order = list(dict.fromkeys(sources))

        if len(sources_without_duplicates_and_in_same_order) == 1:
            # we can't create and "merge layer" with only one input,
            # so we just pass-through the output of the previous layer instead
            return

        merge = make_merge(
            sources=sources_without_duplicates_and_in_same_order,
            reshape_strategy=merge_params.reshape_strategy,
            merge_strategy=merge_params.merge_strategy,
            name_gen=self._name_gen,
        )

        self._previous_layer = merge

    def process_layer(self, layer: gl.Layer) -> None:
        if isinstance(layer, gl.ConvertibleToConnectableLayer):
            self._connect_single_input_layer(layer)

        elif gl.is_fork_marker(layer):
            self._register_fork_point()

        elif gl.is_merge_marker(layer):
            self._connect_multi_input_layer()

        else:
            raise ValueError(f"unknown layer type/configuration: {layer}")


def connect_backbone(
    backbone: bb.Backbone,
    conn_schema: ConnectionsSchema,
    input_layer: gl.Input,
) -> gl.ConnectableLayer:
    """
    This function creates connected versions of the layers in the `backbone`
    using `conn_schema` to describe how to generate "fork points" and "merge layers"

    The return value is the "output layer", i.e., the last layer connected
    to the network. Since the network forms a Directed Acyclic Graph, it is possible
    to "navigate the entire network" by recursively visiting the sources of the output layer.
    """

    merge_params_iter = iter(conn_schema.merge_params)

    stateful_connector = StatefulLayerConnector(
        input_layer=input_layer,
        merge_params_iter=merge_params_iter,
    )

    for layer in backbone.layers:
        stateful_connector.process_layer(layer)

    # sanity check
    assert len(list(merge_params_iter)) == 0

    return stateful_connector.previous_layer
