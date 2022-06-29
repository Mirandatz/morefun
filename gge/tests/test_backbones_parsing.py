from hypothesis import given

import gge.backbones as bb
import gge.tests.strategies.lower_grammar as lgs
from gge.tests.fixtures import remove_logger_sinks  # noqa


@given(test_data=lgs.random_flips())
def test_parse_random_flip(test_data: lgs.LayersTestData) -> None:
    """Can parse random flip."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.random_rotations())
def test_parse_random_rotation(test_data: lgs.LayersTestData) -> None:
    """Can parse random rotation."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.resizings())
def test_parse_resizing(test_data: lgs.LayersTestData) -> None:
    """Can parse resizing."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.random_crops())
def test_parse_random_crop(test_data: lgs.LayersTestData) -> None:
    """Can parse random crop."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.convs())
def test_parse_conv(test_data: lgs.LayersTestData) -> None:
    """Can parse conv."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.maxpools())
def test_parse_maxpool(test_data: lgs.LayersTestData) -> None:
    """Can parse maxpool."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.avgpools())
def test_parse_avgpools(test_data: lgs.LayersTestData) -> None:
    """Can parse avg pool2d."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.batchnorms())
def test_parse_batchnorm(test_data: lgs.LayersTestData) -> None:
    """Can parse batchnorm."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.relus())
def test_parse_relu(test_data: lgs.LayersTestData) -> None:
    """Can parse relu."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.gelus())
def test_parse_gelu(test_data: lgs.LayersTestData) -> None:
    """Can parse gelu."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.swishs())
def test_parse_swish(test_data: lgs.LayersTestData) -> None:
    """Can parse swish."""
    actual = bb.parse(test_data.token_stream, start="backbone")
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual
