import pytest
from hypothesis import example, given  # noqa

import gge.backbones as bb
import gge.tests.strategies.mesagrammars as mesa


@pytest.fixture(autouse=True)
def disable_logger() -> None:
    from loguru import logger

    logger.remove()


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.conv2ds()))
def test_parse_conv2d(test_data: mesa.LayersTestData) -> None:
    """Can parse conv2d."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.maxpools()))
def test_parse_maxpool(test_data: mesa.LayersTestData) -> None:
    """Can parse maxpool."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.avgpools()))
def test_parse_avgpools(test_data: mesa.LayersTestData) -> None:
    """Can parse avg pool2d."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.batchnorms()))
def test_parse_batchnorm(test_data: mesa.LayersTestData) -> None:
    """Can parse batchnorm."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.relus()))
def test_parse_relu(test_data: mesa.LayersTestData) -> None:
    """Can parse relu."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.gelus()))
def test_parse_gelu(test_data: mesa.LayersTestData) -> None:
    """Can parse gelu."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_suffix_to(mesa.swishs()))
def test_parse_swish(test_data: mesa.LayersTestData) -> None:
    """Can parse swish."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual
