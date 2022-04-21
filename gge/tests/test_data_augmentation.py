import attrs
import hypothesis.strategies as hs
from hypothesis import given

import gge.data_augmentations as gda


@attrs.frozen
class ParsingTestData:
    tokenstream: str
    expected: gda.DataAugmentation


@hs.composite
def data_augmentations(draw: hs.DrawFn) -> gda.DataAugmentation:
    rotation_range = draw(hs.floats(min_value=0, max_value=360))
    width_shift_range = draw(hs.floats(min_value=0, max_value=1))
    height_shift_range = draw(hs.floats(min_value=0, max_value=1))
    zoom_range = draw(hs.floats(min_value=0, max_value=1, exclude_max=True))
    horizontal_flip = draw(hs.booleans())
    vertical_flip = draw(hs.booleans())

    return gda.DataAugmentation(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
    )


def make_parsing_test_data(data_aug: gda.DataAugmentation) -> ParsingTestData:
    tokenstream = (
        '"data_augmentation"'
        f'"rotation" {data_aug.rotation_range}'
        f'"width_shift" {data_aug.width_shift_range}'
        f'"height_shift" {data_aug.height_shift_range}'
        f'"zoom" {data_aug.zoom_range}'
        f'"horizontal_flip" {str(data_aug.horizontal_flip).lower()}'
        f'"vertical_flip" {str(data_aug.vertical_flip).lower()}'
    )
    return ParsingTestData(tokenstream, data_aug)


@given(test_data=data_augmentations().map(make_parsing_test_data))
def test_parse(test_data: ParsingTestData) -> None:
    """Can parse lower gramar tokenstream into DataAugmentation."""
    assert test_data.expected == gda.parse(
        test_data.tokenstream,
        start="data_augmentation",
    )


@given(data_aug=data_augmentations())
def test_tensorflow_parity(data_aug: gda.DataAugmentation) -> None:
    """Can convert DataAugmentation to its Tensorflow equivalent."""
    data_generator = data_aug.to_tensorflow_data_generator()
    assert data_generator.rotation_range == data_aug.rotation_range
    assert data_generator.width_shift_range == data_aug.width_shift_range
    assert data_generator.height_shift_range == data_aug.height_shift_range
    assert data_generator.zoom_range == [
        1 - data_aug.zoom_range,
        1 + data_aug.zoom_range,
    ]
    assert data_generator.horizontal_flip == data_aug.horizontal_flip
    assert data_generator.vertical_flip == data_aug.vertical_flip
