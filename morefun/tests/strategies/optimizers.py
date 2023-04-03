import hypothesis.strategies as hs

import morefun.neural_networks.optimizers as optim
import morefun.tests.strategies.data_structures as ds


@hs.composite
def sgds(draw: hs.DrawFn) -> optim.SGD:
    return optim.SGD(
        learning_rate=draw(hs.floats(min_value=0, max_value=9, exclude_min=True)),
        momentum=draw(hs.floats(min_value=0, max_value=1)),
        nesterov=draw(hs.booleans()),
    )


def sgd_grammar(sgd: optim.SGD) -> ds.ParsingTestData[optim.SGD]:
    nesterov = "true" if sgd.nesterov else "false"
    grammar = (
        'start : "sgd"'
        f'"learning_rate" "{sgd.learning_rate}"'
        f'"momentum" "{sgd.momentum}"'
        f'"nesterov" "{nesterov}"'
    )

    return ds.ParsingTestData(grammar, sgd)


@hs.composite
def adams(draw: hs.DrawFn) -> optim.Adam:
    return optim.Adam(
        learning_rate=draw(
            hs.floats(
                min_value=0,
                max_value=9,
                exclude_min=True,
            )
        ),
        beta1=draw(
            hs.floats(
                min_value=0,
                max_value=1,
                exclude_min=True,
                exclude_max=True,
            )
        ),
        beta2=draw(
            hs.floats(
                min_value=0,
                max_value=1,
                exclude_min=True,
                exclude_max=True,
            )
        ),
        epsilon=draw(
            hs.floats(
                min_value=0,
                max_value=1,
                exclude_min=True,
                exclude_max=True,
            )
        ),
        amsgrad=draw(hs.booleans()),
    )


def adam_grammar(adam: optim.Adam) -> ds.ParsingTestData[optim.Adam]:
    amsgrad = "true" if adam.amsgrad else "false"

    grammar = (
        'start : "adam"'
        f'"learning_rate" "{adam.learning_rate}"'
        f'"beta1" "{adam.beta1}"'
        f'"beta2" "{adam.beta2}"'
        f'"epsilon" "{adam.epsilon}"'
        f'"amsgrad" "{amsgrad}"'
    )

    return ds.ParsingTestData(grammar, adam)


@hs.composite
def rangers(draw: hs.DrawFn) -> optim.Ranger:
    return optim.Ranger(
        learning_rate=draw(hs.floats(min_value=0, max_value=9, exclude_min=True)),
        beta1=draw(
            hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
        ),
        beta2=draw(
            hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
        ),
        epsilon=draw(
            hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
        ),
        amsgrad=draw(hs.booleans()),
        sync_period=draw(hs.integers(min_value=1, max_value=3)),
        slow_step_size=draw(
            hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
        ),
    )


def ranger_grammar(ranger: optim.Ranger) -> ds.ParsingTestData[optim.Ranger]:
    amsgrad = "true" if ranger.amsgrad else "false"

    grammar = (
        'start : "ranger"'
        f' "learning_rate" "{ranger.learning_rate}"'
        f' "beta1" "{ranger.beta1}"'
        f' "beta2" "{ranger.beta2}"'
        f' "epsilon" "{ranger.epsilon}"'
        f' "amsgrad" "{amsgrad}"'
        f' "sync_period" "{ranger.sync_period}"'
        f' "slow_step_size" "{ranger.slow_step_size}"'
    )

    return ds.ParsingTestData(grammar, ranger)