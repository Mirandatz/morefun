import typing

import attrs
import lark

import gge.lower_grammar_parsing as lgp


@attrs.frozen
class DataAugmentation:
    """
    rotation_range is measured in degrees
    [width,height]_shift_range is measured as fraction of total [width,heigth]
    """

    rotation_range: float
    width_shift_range: float
    height_shift_range: float
    zoom_range: float
    horizontal_flip: bool
    vertical_flip: bool

    def __attrs_post_init__(self) -> None:
        assert 0 <= self.rotation_range <= 360
        assert 0 <= self.width_shift_range <= 1
        assert 0 <= self.height_shift_range <= 1
        assert 0 <= self.zoom_range <= 1


@lark.v_args(inline=True)
class Transformer(lgp.LowerGrammarTransformer):
    def data_augmentation(
        self,
        marker: None,
        rotation: float,
        width_shift: float,
        height_shift: float,
        zoom: float,
        horizontal_flip: bool,
        vertical_flip: bool,
    ) -> DataAugmentation:
        return DataAugmentation(
            rotation_range=rotation,
            width_shift_range=width_shift,
            height_shift_range=height_shift,
            zoom_range=zoom,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        )

    def rotation(self, marker: None, value: float) -> float:
        return value

    def width_shift(self, marker: None, value: float) -> float:
        return value

    def height_shift(self, marker: None, value: float) -> float:
        return value

    def zoom_shift(self, marker: None, value: float) -> float:
        return value

    def horizontal_flip(self, marker: None, value: bool) -> bool:
        return value

    def vertical_flip(self, marker: None, value: bool) -> bool:
        return value

    def DATA_AUGMENTATION(self, marker: lark.Token) -> None:
        return None

    def ROTATION(self, marker: lark.Token) -> None:
        return None

    def WIDTH_SHIFT(self, marker: lark.Token) -> None:
        return None

    def HEIGHT_SHIFT(self, marker: lark.Token) -> None:
        return None

    def ZOOM(self, marker: lark.Token) -> None:
        return None

    def HORIZONTAL_FLIP(self, marker: lark.Token) -> None:
        return None

    def VERTICAL_FLIP(self, marker: lark.Token) -> None:
        return None


def parse(
    tokenstream: str, start: typing.Literal["start", "data_augmentation"]
) -> DataAugmentation:
    """
    `start` indicates whether `tokenstream`'s first symbol is
    the data augmentation start symbol or the grammar start symbol.
    """

    tree = lgp.parse_tokenstream(
        tokenstream,
        start=start,
        relevant_subtree="data_augmentation",
    )
    data_aug: DataAugmentation = Transformer().transform(tree)
    return data_aug
