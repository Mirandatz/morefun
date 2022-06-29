import collections
import itertools
import typing

import attrs
import lark
from loguru import logger

import gge.layers as gl
import gge.lower_grammar_parsing as lgp
import gge.name_generator
import gge.randomness as rand


def _raise_if_contains_repeated_names(layers: tuple[gl.Layer, ...]) -> None:
    names = collections.Counter(layer.name for layer in layers)  # type: ignore
    repeated = [name for name, times in names.items() if times > 1]
    if repeated:
        raise ValueError(
            f"layers must have unique names, but the following are repeated: {repeated}"
        )


def _raise_if_contains_sequences_of_forks(layers: tuple[gl.Layer, ...]) -> None:
    for prev, curr in itertools.pairwise(layers):
        if gl.is_fork_marker(prev) and gl.is_fork_marker(curr):
            raise ValueError("backbone must not contain sequences of forks")


@attrs.frozen(cache_hash=True)
class Backbone:
    layers: tuple[gl.Layer, ...]

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.layers, tuple)
        for layer in self.layers:
            assert isinstance(layer, gl.Layer)  # type: ignore

        real_layers = filter(gl.is_real_layer, self.layers)
        if not any(real_layers):
            raise ValueError("layers must contain at least one real layer")

        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must not contain duplicates")

        _raise_if_contains_sequences_of_forks(self.layers)
        _raise_if_contains_repeated_names(self.layers)


@lark.v_args(inline=True)
class BackboneSynthetizer(lgp.LowerGrammarTransformer):
    # This set contains terminals that, when visited/processed,
    # are just converted into `None`.
    # It is used in `BackboneSynthetizer.__default_token__` to remove
    # boilerplate code.
    _USELESS_MARKERS = {
        '"random_flip"',
        '"random_rotation"',
        '"resizing"',
        '"height"',
        '"width"',
        '"conv"',
        '"filter_count"',
        '"kernel_size"',
        '"stride"',
        '"maxpool"',
        '"avgpool"',
        '"pool_size"',
        '"batchnorm"',
        '"relu"',
        '"gelu"',
        '"swish"',
    }

    # This set is also used to remove boilplate code.
    # For more information, see: `BackboneSynthetizer.__default_token__`
    _RULES_OF_MARKERS_VALUE_PAIRS = {
        "height",
        "width",
        "filter_count",
        "kernel_size",
        "stride",
        "pool_size",
    }

    def __init__(self) -> None:
        super().__init__()
        self._name_generator = gge.name_generator.NameGenerator()

    def __default__(
        self,
        data: lark.Token,
        children: list[typing.Any],
        meta: typing.Any,
    ) -> typing.Any:
        self._raise_if_not_running()

        if data.value in BackboneSynthetizer._RULES_OF_MARKERS_VALUE_PAIRS:
            marker, value = children
            return value

        return super().__default__(data, children, meta)

    def __default_token__(self, token: lark.Token) -> typing.Any:
        self._raise_if_not_running()

        if token.value in BackboneSynthetizer._USELESS_MARKERS:
            return None

        return super().__default_token__(token)

    def _create_layer_name(self, prefix: str | type) -> str:
        self._raise_if_not_running()
        return self._name_generator.gen_name(prefix)

    @lark.v_args(inline=False)
    def backbone(self, blocks: typing.Any) -> Backbone:
        self._raise_if_not_running()

        assert isinstance(blocks, list)
        for list_of_layers in blocks:
            assert isinstance(list_of_layers, list)
            for layer in list_of_layers:
                assert isinstance(layer, gl.Layer)  # type: ignore

        layers = tuple(layer for list_of_layers in blocks for layer in list_of_layers)
        return Backbone(layers)

    @lark.v_args(inline=False)
    def block(self, parts: typing.Any) -> list[gl.Layer]:
        self._raise_if_not_running()

        return [x for x in parts if x is not None]

    def MERGE(self, token: lark.Token | None = None) -> gl.MarkerLayer | None:
        self._raise_if_not_running()

        if token is not None:
            return gl.make_merge(self._create_layer_name("merge"))
        else:
            return None

    def FORK(self, token: lark.Token | None = None) -> gl.MarkerLayer | None:
        self._raise_if_not_running()

        if token is not None:
            return gl.make_fork(self._create_layer_name("fork"))
        else:
            return None

    def layer(self, layer: gl.Layer) -> gl.Layer:
        self._raise_if_not_running()
        return layer

    def random_flip(self, marker: None, mode: str) -> gl.RandomFlip:
        self._raise_if_not_running()
        assert mode in gl.FLIP_MODES

        return gl.RandomFlip(
            name=self._create_layer_name(gl.RandomFlip),
            mode=mode,
            seed=rand.get_rng_seed(),
        )

    def random_rotation(self, marker: None, factor: float) -> gl.RandomRotation:
        self._raise_if_not_running()
        assert isinstance(factor, float)
        assert 0 <= factor <= 1

        return gl.RandomRotation(
            name=self._create_layer_name(gl.RandomRotation),
            factor=factor,
            seed=rand.get_rng_seed(),
        )

    def resizing(
        self, marker: None, target_height: int, target_width: int
    ) -> gl.Resizing:
        self._raise_if_not_running()

        return gl.Resizing(
            name=self._create_layer_name(gl.Resizing),
            height=target_height,
            width=target_width,
        )

    def conv(
        self, marker: None, filter_count: int, kernel_size: int, stride: int
    ) -> gl.Conv2D:
        self._raise_if_not_running()

        return gl.Conv2D(
            name=self._create_layer_name(gl.Conv2D),
            filter_count=filter_count,
            kernel_size=kernel_size,
            stride=stride,
        )

    def batchnorm(self, marker: None) -> gl.BatchNorm:
        self._raise_if_not_running()
        return gl.BatchNorm(self._create_layer_name(gl.BatchNorm))

    def maxpool(self, marker: None, pool_size: int, stride: int) -> gl.MaxPool2D:
        self._raise_if_not_running()
        return gl.MaxPool2D(
            name=self._create_layer_name(gl.MaxPool2D),
            pool_size=pool_size,
            stride=stride,
        )

    def avgpool(self, marker: None, pool_size: int, stride: int) -> gl.AvgPool2D:
        self._raise_if_not_running()
        return gl.AvgPool2D(
            name=self._create_layer_name(gl.AvgPool2D),
            pool_size=pool_size,
            stride=stride,
        )

    def activation(self, layer: gl.SingleInputLayer) -> gl.SingleInputLayer:
        self._raise_if_not_running()
        return layer

    def RELU(self, token: lark.Token) -> gl.Relu:
        self._raise_if_not_running()
        return gl.Relu(name=self._create_layer_name(gl.Relu))

    def GELU(self, token: lark.Token) -> gl.Gelu:
        self._raise_if_not_running()
        return gl.Gelu(name=self._create_layer_name(gl.Gelu))

    def SWISH(self, token: lark.Token) -> gl.Swish:
        self._raise_if_not_running()
        return gl.Swish(name=self._create_layer_name(gl.Swish))

    def FLIP_MODE(self, token: lark.Token) -> str:
        self._raise_if_not_running()
        if token.value not in gl.FLIP_MODES:
            raise ValueError(f"unexpected token for FLIP_MODE=<{token.value}>")

        return typing.cast(str, token.value)


def parse(
    tokenstream: str,
    *,
    start: typing.Literal["start", "backbone"] = "start",
) -> Backbone:
    """
    `start` indicates whether `tokenstream`'s first symbol is
    the backbone start symbol or the grammar start symbol.
    """

    logger.debug("parsing backbone tokenstream")

    assert start in ("start", "backbone")
    tree = lgp.parse_tokenstream(
        tokenstream,
        start=start,
        relevant_subtree="backbone",
    )
    backbone = BackboneSynthetizer().transform(tree)
    assert isinstance(backbone, Backbone)

    logger.debug("finished parsing backbone tokenstream")
    return backbone
