import attrs
import hypothesis.strategies as hs
from hypothesis import given

import gge.data_augmentations as gda


@attrs.frozen
class ParsingTestData:
    tokenstream: str
    expected: gda.DataAugmentation


@hs.composite
def data_augmentations(draw: hs.DrawFn) -> ParsingTestData:
    rotation_range = draw(hs.floats(min_value=0, max_value=360))
    width_shift_range = draw(hs.floats(min_value=0, max_value=1))
    height_shift_range = draw(hs.floats(min_value=0, max_value=1))
    zoom_range = draw(hs.floats(min_value=0, max_value=1))
    horizontal_flip = draw(hs.booleans())
    vertical_flip = draw(hs.booleans())

    tokenstream = (
        '"data_augmentation"'
        f'"rotation" {rotation_range}'
        f'"width_shift" {width_shift_range}'
        f'"height_shift" {height_shift_range}'
        f'"zoom" {zoom_range}'
        f'"horizontal_flip" {str(horizontal_flip).lower()}'
        f'"vertical_flip" {str(vertical_flip).lower()}'
    )

    return ParsingTestData(
        tokenstream=tokenstream,
        expected=gda.DataAugmentation(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        ),
    )


@given(test_data=data_augmentations())
def test_parse(test_data: ParsingTestData) -> None:
    """Can parse lower gramar tokenstream into DataAugmentation."""
    assert test_data.expected == gda.parse(
        test_data.tokenstream,
        start="data_augmentation",
    )
