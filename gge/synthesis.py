import typing
from dataclasses import dataclass
from typing import Any, Union

import lark
import typeguard


@typeguard.typechecked
@dataclass(frozen=True)
class Input:
    pass


@typeguard.typechecked
@dataclass(frozen=True)
class Conv2d:
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


@typeguard.typechecked
@dataclass(frozen=True)
class Dense:
    units: int

    def __post_init__(self) -> None:
        assert self.units > 0


@typeguard.typechecked
@dataclass(frozen=True)
class Dropout:
    rate: float

    def __post_init__(self) -> None:
        assert 0 <= self.rate <= 1


Layer = Union[Input, Conv2d, Dense, Dropout]
MainPath = typing.Tuple[Layer, ...]


class Synthetizer(lark.Transformer[MainPath]):
    def __init__(self) -> None:
        super().__init__()

    def __default__(self, data: Any, children: Any, meta: Any) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: str) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    def start(self, layers: list[Layer]) -> MainPath:
        return tuple(layers)

    @lark.v_args(inline=True)
    def layer(self, layer: Layer) -> Layer:
        return layer

    @lark.v_args(inline=True)
    def conv_layer(self, filter_count: int, kernel_size: int, stride: int) -> Conv2d:
        return Conv2d(
            filter_count=filter_count,
            kernel_size=kernel_size,
            stride=stride,
        )

    @lark.v_args(inline=True, meta=True)
    def filter_count(self, meta: lark.tree.Meta, count: int) -> int:
        if count < 0:
            raise ValueError(
                f"count must be >= 0. line, column=[{meta.line},{meta.column}]"
            )

        return count

    @lark.v_args(inline=True, meta=True)
    def kernel_size(self, meta: lark.tree.Meta, stride: int) -> int:
        if stride < 0:
            raise ValueError(
                f"stride must be >= 0. line, column=[{meta.line},{meta.column}]"
            )

        return stride

    @lark.v_args(inline=True, meta=True)
    def stride(self, meta: lark.tree.Meta, size: int) -> int:
        if size < 0:
            raise ValueError(
                f"size must be >= 0. line, column=[{meta.line},{meta.column}]"
            )

        return size

    @lark.v_args(inline=True, meta=True)
    def dense_layer(self, meta: lark.tree.Meta, units: int) -> Dense:
        if units < 0:
            raise ValueError(
                f"units must be >= 0. line, column=[{meta.line},{meta.column}]"
            )

        return Dense(units)

    @lark.v_args(inline=True, meta=True)
    def dropout_layer(self, meta: lark.tree.Meta, rate: float) -> Dropout:
        if not (0 <= rate <= 1):
            raise ValueError(
                f"rate must be >= 0 and <= 1. line, column=[{meta.line},{meta.column}]"
            )

        return Dropout(rate)

    def INT(self, token: lark.Token) -> int:
        return int(token.value)

    def FLOAT(self, token: lark.Token) -> float:
        return float(token.value)


def main() -> None:
    from pathlib import Path

    DATA_DIR = Path(__file__).parent.parent / "data"
    parser = lark.Lark.open(
        grammar_filename=str(DATA_DIR / "mesagrammar.lark"),
        parser="lalr",
    )
    tree = parser.parse(
        r"""
        conv2d filter_count 2 kernel_size 4 stride 2 dense 4 dropout 0.2 dense 10 dropout 0.6
    """
    )

    mainpath = Synthetizer().transform(tree)
    print(mainpath)

    # print(tree.pretty())


if __name__ == "__main__":
    main()
