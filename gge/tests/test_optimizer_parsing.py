from hypothesis import given

import gge.optimizers as go
import gge.tests.strategies.mesagrammars as ms


@given(test_data=ms.add_dummy_layer_prefix_to(ms.sgds()))
def test_parse_sgd(test_data: ms.OptimizerTestData) -> None:
    """Can parse SGD optimizer."""
    actual = go.parse(test_data.token_stream)
    expected = test_data.parsed
    assert expected == actual


@given(test_data=ms.add_dummy_layer_prefix_to(ms.adams()))
def test_parse_adam(test_data: ms.OptimizerTestData) -> None:
    """Can parse Adam optimizer."""
    actual = go.parse(test_data.token_stream)
    expected = test_data.parsed
    assert expected == actual
