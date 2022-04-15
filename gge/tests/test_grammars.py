import itertools

import pytest
from hypothesis import given

import gge.grammars as gr
import gge.tests.strategies.metagrammars as ms

START = gr.NonTerminal("start")


@pytest.fixture(autouse=True)
def disable_logger() -> None:
    from loguru import logger

    logger.remove()


def test_start_symbol() -> None:
    grammar = gr.Grammar(
        """
        start     : convblock
        convblock : conv~2
        conv      : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """
    )
    assert START == grammar.start_symbol


def test_terminals() -> None:
    grammar = gr.Grammar(
        """
        start : a act1 b act2
        a : "conv2d" "filter_count" 1 "kernel_size" (2) "stride" (3 | 4)
        b : "conv2d" "filter_count" 5 "kernel_size" 6 "stride" (7)
        act1 : "relu"
        act2 : "swish"
    """
    )
    actual = set(grammar.terminals)

    names = {
        gr.ExpectedTerminal.CONV2D.value,
        gr.ExpectedTerminal.FILTER_COUNT.value,
        gr.ExpectedTerminal.KERNEL_SIZE.value,
        gr.ExpectedTerminal.STRIDE.value,
        gr.ExpectedTerminal.RELU.value,
        gr.ExpectedTerminal.SWISH.value,
    }
    numbers = {gr.Terminal(str(i)) for i in range(1, 8)}
    expected = set.union(names, numbers)

    assert expected == actual


def test_non_terminals() -> None:
    grammar = gr.Grammar(
        """
        start : a b
        a : "conv2d" "filter_count" 1 "kernel_size" (2) "stride" (3 | 4)
        b : "conv2d" "filter_count" 5 "kernel_size" 6 "stride" (7)
    """
    )
    actual = set(grammar.nonterminals)
    expected = {START, gr.NonTerminal("a"), gr.NonTerminal("b")}
    assert expected == actual


def test_start_trivial_expansion() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a     : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
        """
    )
    (actual,) = grammar.expansions(START)
    expected = gr.RuleOption((gr.NonTerminal("a"),))
    assert expected == actual


def test_start_simple_expansion() -> None:
    grammar = gr.Grammar(
        """
        start : a b
        a     : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
        b     : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
        """
    )
    (actual,) = grammar.expansions(START)

    expected = gr.RuleOption(
        (
            gr.NonTerminal("a"),
            gr.NonTerminal("b"),
        )
    )

    assert expected == actual


def test_start_complex_expansion() -> None:
    grammar = gr.Grammar(
        """
        start : a b | c | c a
        a     : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
        b     : "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6
        c     : "conv2d" "filter_count" 7 "kernel_size" 8 "stride" 9
        """
    )
    actual = grammar.expansions(START)

    a = gr.NonTerminal("a")
    b = gr.NonTerminal("b")
    c = gr.NonTerminal("c")

    expected = (
        gr.RuleOption((a, b)),
        gr.RuleOption((c,)),
        gr.RuleOption((c, a)),
    )

    assert expected == actual


def test_complex_symbol_expansion() -> None:
    grammar = gr.Grammar(
        """
        start  : sblock cblock conv1
        sblock : conv1 conv2
        cblock : conv1~1..2 sblock conv2~1..3
               | sblock~2..2 conv1~1
               | conv3

        conv1 : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
        conv2 : "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6
        conv3 : "conv2d" "filter_count" 7 "kernel_size" 8 "stride" 9
        """
    )
    actual = grammar.expansions(gr.NonTerminal("cblock"))

    sb = gr.NonTerminal("sblock")
    c1 = gr.NonTerminal("conv1")
    c2 = gr.NonTerminal("conv2")
    c3 = gr.NonTerminal("conv3")

    expected = (
        gr.RuleOption((c1, sb, c2)),
        gr.RuleOption((c1, sb, c2, c2)),
        gr.RuleOption((c1, sb, c2, c2, c2)),
        gr.RuleOption((c1, c1, sb, c2)),
        gr.RuleOption((c1, c1, sb, c2, c2)),
        gr.RuleOption((c1, c1, sb, c2, c2, c2)),
        gr.RuleOption((sb, sb, c1)),
        gr.RuleOption((c3,)),
    )

    assert expected == actual


def test_non_terminal_with_single_expansion() -> None:
    grammar = gr.Grammar(
        """ start : block
        block : layer
        layer : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """
    )
    (actual,) = grammar.expansions(gr.NonTerminal("block"))
    expected = gr.RuleOption((gr.NonTerminal("layer"),))
    assert actual == expected


def test_int_arg_parenthesis() -> None:
    grammar = gr.Grammar(
        """
        start   : with without
        with    : "conv2d" "filter_count" (1) "kernel_size" (2) "stride" (3)
        without : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
        """
    )
    with_par = grammar.expansions(gr.NonTerminal("with"))
    without_par = grammar.expansions(gr.NonTerminal("without"))
    assert with_par == without_par


def test_multiple_int_arg() -> None:
    grammar = gr.Grammar(
        """
        start : layer
        layer : "conv2d" "filter_count" (1 | 2) "kernel_size" 2 "stride" 3
        """
    )
    actual_first, actual_second = grammar.expansions(gr.NonTerminal("layer"))

    expected_first = gr.RuleOption(
        (
            gr.ExpectedTerminal.CONV2D.value,
            gr.ExpectedTerminal.FILTER_COUNT.value,
            gr.Terminal("1"),
            gr.ExpectedTerminal.KERNEL_SIZE.value,
            gr.Terminal("2"),
            gr.ExpectedTerminal.STRIDE.value,
            gr.Terminal("3"),
        )
    )
    assert expected_first == actual_first

    expected_second = gr.RuleOption(
        (
            gr.ExpectedTerminal.CONV2D.value,
            gr.ExpectedTerminal.FILTER_COUNT.value,
            gr.Terminal("2"),
            gr.ExpectedTerminal.KERNEL_SIZE.value,
            gr.Terminal("2"),
            gr.ExpectedTerminal.STRIDE.value,
            gr.Terminal("3"),
        )
    )
    assert expected_second == actual_second


def test_blockless_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : conv
        conv  : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """
    )
    start_rule, conv_def_rule = grammar.rules

    assert START == start_rule.lhs
    assert gr.RuleOption((gr.NonTerminal("conv"),)) == start_rule.rhs

    assert gr.NonTerminal("conv") == conv_def_rule.lhs
    expected_rhs = gr.RuleOption(
        (
            gr.ExpectedTerminal.CONV2D.value,
            gr.ExpectedTerminal.FILTER_COUNT.value,
            gr.Terminal("1"),
            gr.ExpectedTerminal.KERNEL_SIZE.value,
            gr.Terminal("2"),
            gr.ExpectedTerminal.STRIDE.value,
            gr.Terminal("3"),
        )
    )
    assert expected_rhs == conv_def_rule.rhs


def test_batchnorm_in_block() -> None:
    grammar = gr.Grammar(
        """
        start : block
        block : conv norm
        conv  : "conv2d" "filter_count" (1 | 2) "kernel_size" 2 "stride" 3
        norm  : "batchnorm"
        """
    )

    nt = gr.NonTerminal("norm")
    assert nt in grammar.nonterminals

    (block_exp,) = grammar.expansions(gr.NonTerminal("block"))
    _, actual = block_exp.symbols

    assert nt == actual


def test_batchnorm_def() -> None:
    grammar = gr.Grammar(
        """
        start : block
        block : conv norm
        conv  : "conv2d" "filter_count" (1 | 2) "kernel_size" 2 "stride" 3
        norm  : "batchnorm"
        """
    )

    nt = gr.NonTerminal("norm")
    (actual_expansion,) = grammar.expansions(nt)

    expected_expansion = gr.RuleOption((gr.ExpectedTerminal.BATCHRNOM.value,))

    assert expected_expansion == actual_expansion


def test_pooling_layer_in_block() -> None:
    grammar = gr.Grammar(
        """
        start : block
        block : conv pool
        conv  : "conv2d" "filter_count" 1 "kernel_size" (2) "stride" (3 | 4)
        pool  : "maxpool" "pool_size" 2 "stride" 3
        """
    )

    nt = gr.NonTerminal("pool")
    assert nt in grammar.nonterminals

    (block_exp,) = grammar.expansions(gr.NonTerminal("block"))
    _, actual = block_exp.symbols

    assert nt == actual


@given(
    pool_sizes=ms.int_args(min_value=1, max_value=9),
    strides=ms.int_args(min_value=1, max_value=9),
)
def test_maxpool_def(
    pool_sizes: ms.GrammarArgs,
    strides: ms.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "maxpool"'
        f' "pool_size" {pool_sizes.text}'
        f' "stride" {strides.text}'
    )
    grammar = gr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(gr.NonTerminal("start"))

    # asserts
    test_values = itertools.product(pool_sizes.terminals, strides.terminals)
    for expansion, (pool_size_term, stride_term) in zip(
        expansions, test_values, strict=True
    ):
        expected = gr.RuleOption(
            (
                gr.ExpectedTerminal.MAX_POOL.value,
                gr.ExpectedTerminal.POOL_SIZE.value,
                pool_size_term,
                gr.ExpectedTerminal.STRIDE.value,
                stride_term,
            )
        )
        assert expected == expansion


@given(
    pool_sizes=ms.int_args(min_value=1, max_value=9),
    strides=ms.int_args(min_value=1, max_value=9),
)
def test_avg_pool2d_def(
    pool_sizes: ms.GrammarArgs,
    strides: ms.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "avg_pool2d"'
        f' "pool_size" {pool_sizes.text}'
        f' "stride" {strides.text}'
    )
    grammar = gr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(gr.NonTerminal("start"))

    test_values = itertools.product(pool_sizes.terminals, strides.terminals)
    for expansion, (pool_size_term, stride_term) in zip(
        expansions, test_values, strict=True
    ):
        expected = gr.RuleOption(
            (
                gr.ExpectedTerminal.AVG_POOL.value,
                gr.ExpectedTerminal.POOL_SIZE.value,
                pool_size_term,
                gr.ExpectedTerminal.STRIDE.value,
                stride_term,
            )
        )
        assert expected == expansion


def test_relu_def() -> None:
    grammar = gr.Grammar(
        """
        start : "relu"
        """
    )
    (actual,) = grammar.expansions(gr.NonTerminal("start"))
    expected = gr.RuleOption((gr.ExpectedTerminal.RELU.value,))
    assert expected == actual


@given(
    learning_rate=ms.float_args(min_value=0, exclude_min=True),
    momentum=ms.float_args(min_value=0, exclude_min=True),
    nesterov=ms.bool_args(),
)
def test_sgd_def(
    learning_rate: ms.GrammarArgs,
    momentum: ms.GrammarArgs,
    nesterov: ms.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "sgd"'
        f' "learning_rate" {learning_rate.text}'
        f' "momentum" {momentum.text}'
        f' "nesterov" {nesterov.text}'
    )
    grammar = gr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(gr.NonTerminal("start"))

    test_values = itertools.product(
        learning_rate.terminals,
        momentum.terminals,
        nesterov.terminals,
    )

    for actual_expansion, (lr_term, mom_term, nest_term) in zip(
        expansions,
        test_values,
        strict=True,
    ):
        expected_symbols = (
            gr.ExpectedTerminal.SGD.value,
            gr.ExpectedTerminal.LEARNING_RATE.value,
            lr_term,
            gr.ExpectedTerminal.MOMENTUM.value,
            mom_term,
            gr.ExpectedTerminal.NESTEROV.value,
            nest_term,
        )
        expected_expansion = gr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion


@given(
    learning_rate=ms.float_args(min_value=0, exclude_min=True),
    beta1=ms.float_args(min_value=0, exclude_min=True),
    beta2=ms.float_args(min_value=0, exclude_min=True),
    epsilon=ms.float_args(min_value=0, exclude_min=True),
    amsgrad=ms.bool_args(),
)
def test_adam_def(
    learning_rate: ms.GrammarArgs,
    beta1: ms.GrammarArgs,
    beta2: ms.GrammarArgs,
    epsilon: ms.GrammarArgs,
    amsgrad: ms.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "adam"'
        f' "learning_rate" {learning_rate.text}'
        f' "beta1" {beta1.text}'
        f' "beta2" {beta2.text}'
        f' "epsilon" {epsilon.text}'
        f' "amsgrad" {amsgrad.text}'
    )
    grammar = gr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(gr.NonTerminal("start"))

    test_values = itertools.product(
        learning_rate.terminals,
        beta1.terminals,
        beta2.terminals,
        epsilon.terminals,
        amsgrad.terminals,
    )

    for actual_expansion, terminals in zip(
        expansions,
        test_values,
        strict=True,
    ):
        (
            lr_term,
            beta1_term,
            beta2_term,
            epsilon_term,
            amsgrad_term,
        ) = terminals

        expected_symbols = (
            gr.ExpectedTerminal.ADAM.value,
            gr.ExpectedTerminal.LEARNING_RATE.value,
            lr_term,
            gr.ExpectedTerminal.BETA1.value,
            beta1_term,
            gr.ExpectedTerminal.BETA2.value,
            beta2_term,
            gr.ExpectedTerminal.EPSILON.value,
            epsilon_term,
            gr.ExpectedTerminal.AMSGRAD.value,
            amsgrad_term,
        )
        expected_expansion = gr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion
