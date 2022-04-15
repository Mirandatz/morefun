import pytest
from hypothesis import example, given  # noqa

import gge.backbones as bb
import gge.tests.strategies.lower_grammar as lgs


@pytest.fixture(autouse=True)
def disable_logger() -> None:
    from loguru import logger

    logger.remove()


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.convs()))
def test_parse_conv(test_data: lgs.LayersTestData) -> None:
    """Can parse conv."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.maxpools()))
def test_parse_maxpool(test_data: lgs.LayersTestData) -> None:
    """Can parse maxpool."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.avgpools()))
def test_parse_avgpools(test_data: lgs.LayersTestData) -> None:
    """Can parse avg pool2d."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.batchnorms()))
def test_parse_batchnorm(test_data: lgs.LayersTestData) -> None:
    """Can parse batchnorm."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.relus()))
def test_parse_relu(test_data: lgs.LayersTestData) -> None:
    """Can parse relu."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.gelus()))
def test_parse_gelu(test_data: lgs.LayersTestData) -> None:
    """Can parse gelu."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=lgs.add_dummy_optimizer_suffix_to(lgs.swishs()))
def test_parse_swish(test_data: lgs.LayersTestData) -> None:
    """Can parse swish."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual
