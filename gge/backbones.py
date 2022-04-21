import collections
import itertools
import typing

import attrs
import lark
from loguru import logger

import gge.layers as gl
import gge.lower_grammar_parsing as lgp
import gge.name_generator


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
    def __init__(self) -> None:
        super().__init__()
        self._name_generator = gge.name_generator.NameGenerator()

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

    def filter_count(self, marker: None, count: int) -> int:
        self._raise_if_not_running()
        assert count >= 1

        return count

    def kernel_size(self, marker: None, stride: int) -> int:
        self._raise_if_not_running()
        assert stride >= 1

        return stride

    def stride(self, marker: None, size: int) -> int:
        self._raise_if_not_running()
        assert size >= 1

        return size

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

    def pool_size(self, marker: None, pool_size: int) -> int:
        self._raise_if_not_running()
        assert pool_size >= 1
        return pool_size

    def activation(self, layer: gl.SingleInputLayer) -> gl.SingleInputLayer:
        self._raise_if_not_running()
        return layer

    def CONV(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def FILTER_COUNT(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def KERNEL_SIZE(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def STRIDE(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def MAXPOOL(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def AVGPOOL(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def POOL_SIZE(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def BATCHNORM(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def RELU(self, token: lark.Token) -> gl.Relu:
        self._raise_if_not_running()
        return gl.Relu(name=self._create_layer_name(gl.Relu))

    def GELU(self, token: lark.Token) -> gl.Gelu:
        self._raise_if_not_running()
        return gl.Gelu(name=self._create_layer_name(gl.Gelu))

    def SWISH(self, token: lark.Token) -> gl.Swish:
        self._raise_if_not_running()
        return gl.Swish(name=self._create_layer_name(gl.Swish))


def parse(tokenstream: str, start: typing.Literal["start", "backbone"]) -> Backbone:
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
