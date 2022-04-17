from hypothesis import given

import gge.optimizers as optim
import gge.tests.strategies.lower_grammar as lgs


@given(test_data=lgs.add_dummy_layer_prefix_to(lgs.sgds()))
def test_parse_sgd(test_data: lgs.OptimizerTestData) -> None:
    """Can parse SGD optimizer."""
    actual = optim.parse(test_data.token_stream)
    expected = test_data.parsed
    assert expected == actual


@given(test_data=lgs.add_dummy_layer_prefix_to(lgs.adams()))
def test_parse_adam(test_data: lgs.OptimizerTestData) -> None:
    """Can parse Adam optimizer."""
    actual = optim.parse(test_data.token_stream)
    expected = test_data.parsed
    assert expected == actual
