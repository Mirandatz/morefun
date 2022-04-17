import hypothesis.strategies as hs
from hypothesis import given

import gge.optimizers as optim


@given(
    learning_rate=hs.floats(min_value=0, max_value=9, exclude_min=True),
    momentum=hs.floats(min_value=0, max_value=1),
    nesterov=hs.booleans(),
)
def test_sgd_to_keras(
    learning_rate: float,
    momentum: float,
    nesterov: bool,
) -> None:
    ours = optim.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    theirs = ours.to_keras()

    assert ours.learning_rate == theirs.learning_rate
    assert ours.momentum == theirs.momentum
    assert ours.nesterov == theirs.nesterov


@given(
    learning_rate=hs.floats(min_value=0, max_value=9, exclude_min=True),
    beta1=hs.floats(min_value=0, max_value=9, exclude_min=True),
    beta2=hs.floats(min_value=0, max_value=9, exclude_min=True),
    epsilon=hs.floats(min_value=0, max_value=9, exclude_min=True),
    amsgrad=hs.booleans(),
)
def test_adam_to_keras(
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    amsgrad: bool,
) -> None:
    ours = optim.Adam(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        amsgrad=amsgrad,
    )

    theirs = ours.to_keras()

    assert ours.learning_rate == theirs.learning_rate
    assert ours.beta1 == theirs.beta_1
    assert ours.beta2 == theirs.beta_2
    assert ours.epsilon == theirs.epsilon
    assert ours.amsgrad == theirs.amsgrad
