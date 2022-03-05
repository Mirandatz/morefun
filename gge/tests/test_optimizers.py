from hypothesis import given

import gge.optimizers as go
import gge.tests.strategies.mesagrammars as ms


@given(test_data=ms.add_dummy_layer_to(ms.sgds()))
def test_sgd(test_data: ms.SGDTestData) -> None:
    """Can parse SGD optimizer."""
    actual = go.parse(test_data.token_stream)
    expected = test_data.parsed
    assert expected == actual
