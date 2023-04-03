import itertools

import pytest
from hypothesis import given

import gge.grammars.upper_grammars as ugr
import gge.tests.strategies.upper_grammar as ugs

# autouse fixture
from gge.tests.fixtures import remove_logger_sinks  # noqa

START = ugr.NonTerminal("start")


def test_start_symbol() -> None:
    grammar = ugr.Grammar(
        """
        start     : convblock
        convblock : conv~2
        conv      : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
    """
    )
    assert START == grammar.start_symbol


def test_terminals() -> None:
    grammar = ugr.Grammar(
        """
        start : a act1 b act2
        a : "conv" "filter_count" "1" "kernel_size" ("2") "stride" ("3" | "4")
        b : "conv" "filter_count" "5" "kernel_size" "6" "stride" ("7")
        act1 : "relu"
        act2 : "swish"
    """
    )
    actual = set(grammar.terminals)

    names = {
        ugr.ExpectedTerminal.CONV.value,
        ugr.ExpectedTerminal.FILTER_COUNT.value,
        ugr.ExpectedTerminal.KERNEL_SIZE.value,
        ugr.ExpectedTerminal.STRIDE.value,
        ugr.ExpectedTerminal.RELU.value,
        ugr.ExpectedTerminal.SWISH.value,
    }
    numbers = {ugr.Terminal(f'"{i}"') for i in range(1, 8)}
    expected = set.union(names, numbers)

    assert expected == actual


def test_non_terminals() -> None:
    grammar = ugr.Grammar(
        """
        start : a b
        a : "conv" "filter_count" "1" "kernel_size" ("2") "stride" ("3" | "4")
        b : "conv" "filter_count" "5" "kernel_size" "6" "stride" ("7")
    """
    )
    actual = set(grammar.nonterminals)
    expected = {START, ugr.NonTerminal("a"), ugr.NonTerminal("b")}
    assert expected == actual


def test_start_trivial_expansion() -> None:
    grammar = ugr.Grammar(
        """
        start : a
        a     : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        """
    )
    (actual,) = grammar.expansions(START)
    expected = ugr.RuleOption((ugr.NonTerminal("a"),))
    assert expected == actual


def test_start_simple_expansion() -> None:
    grammar = ugr.Grammar(
        """
        start : a b
        a     : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        b     : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        """
    )
    (actual,) = grammar.expansions(START)

    expected = ugr.RuleOption(
        (
            ugr.NonTerminal("a"),
            ugr.NonTerminal("b"),
        )
    )

    assert expected == actual


def test_start_complex_expansion() -> None:
    grammar = ugr.Grammar(
        """
        start : a b | c | c a
        a     : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        b     : "conv" "filter_count" "4" "kernel_size" "5" "stride" "6"
        c     : "conv" "filter_count" "7" "kernel_size" "8" "stride" "9"
        """
    )
    actual = grammar.expansions(START)

    a = ugr.NonTerminal("a")
    b = ugr.NonTerminal("b")
    c = ugr.NonTerminal("c")

    expected = (
        ugr.RuleOption((a, b)),
        ugr.RuleOption((c,)),
        ugr.RuleOption((c, a)),
    )

    assert expected == actual


def test_complex_symbol_expansion() -> None:
    grammar = ugr.Grammar(
        """
        start  : sblock cblock conv1
        sblock : conv1 conv2
        cblock : conv1~1..2 sblock conv2~1..3
               | sblock~2..2 conv1~1
               | conv3

        conv1 : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        conv2 : "conv" "filter_count" "4" "kernel_size" "5" "stride" "6"
        conv3 : "conv" "filter_count" "7" "kernel_size" "8" "stride" "9"
        """
    )
    actual = grammar.expansions(ugr.NonTerminal("cblock"))

    sb = ugr.NonTerminal("sblock")
    c1 = ugr.NonTerminal("conv1")
    c2 = ugr.NonTerminal("conv2")
    c3 = ugr.NonTerminal("conv3")

    expected = (
        ugr.RuleOption((c1, sb, c2)),
        ugr.RuleOption((c1, sb, c2, c2)),
        ugr.RuleOption((c1, sb, c2, c2, c2)),
        ugr.RuleOption((c1, c1, sb, c2)),
        ugr.RuleOption((c1, c1, sb, c2, c2)),
        ugr.RuleOption((c1, c1, sb, c2, c2, c2)),
        ugr.RuleOption((sb, sb, c1)),
        ugr.RuleOption((c3,)),
    )

    assert expected == actual


def test_non_terminal_with_single_expansion() -> None:
    grammar = ugr.Grammar(
        """ start : block
        block : layer
        layer : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
    """
    )
    (actual,) = grammar.expansions(ugr.NonTerminal("block"))
    expected = ugr.RuleOption((ugr.NonTerminal("layer"),))
    assert actual == expected


def test_int_arg_parenthesis() -> None:
    grammar = ugr.Grammar(
        """
        start   : with without
        with    : "conv" "filter_count" ("1") "kernel_size" ("2") "stride" ("3")
        without : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        """
    )
    with_par = grammar.expansions(ugr.NonTerminal("with"))
    without_par = grammar.expansions(ugr.NonTerminal("without"))
    assert with_par == without_par


def test_multiple_int_arg() -> None:
    grammar = ugr.Grammar(
        """
        start : layer
        layer : "conv" "filter_count" ("1" | "2") "kernel_size" "2" "stride" "3"
        """
    )
    actual_first, actual_second = grammar.expansions(ugr.NonTerminal("layer"))

    expected_first = ugr.RuleOption(
        (
            ugr.ExpectedTerminal.CONV.value,
            ugr.ExpectedTerminal.FILTER_COUNT.value,
            ugr.Terminal('"1"'),
            ugr.ExpectedTerminal.KERNEL_SIZE.value,
            ugr.Terminal('"2"'),
            ugr.ExpectedTerminal.STRIDE.value,
            ugr.Terminal('"3"'),
        )
    )
    assert expected_first == actual_first

    expected_second = ugr.RuleOption(
        (
            ugr.ExpectedTerminal.CONV.value,
            ugr.ExpectedTerminal.FILTER_COUNT.value,
            ugr.Terminal('"2"'),
            ugr.ExpectedTerminal.KERNEL_SIZE.value,
            ugr.Terminal('"2"'),
            ugr.ExpectedTerminal.STRIDE.value,
            ugr.Terminal('"3"'),
        )
    )
    assert expected_second == actual_second


def test_blockless_grammar() -> None:
    grammar = ugr.Grammar(
        """
        start : conv
        conv  : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
    """
    )
    start_rule, conv_def_rule = grammar.rules

    assert START == start_rule.lhs
    assert ugr.RuleOption((ugr.NonTerminal("conv"),)) == start_rule.rhs

    assert ugr.NonTerminal("conv") == conv_def_rule.lhs
    expected_rhs = ugr.RuleOption(
        (
            ugr.ExpectedTerminal.CONV.value,
            ugr.ExpectedTerminal.FILTER_COUNT.value,
            ugr.Terminal('"1"'),
            ugr.ExpectedTerminal.KERNEL_SIZE.value,
            ugr.Terminal('"2"'),
            ugr.ExpectedTerminal.STRIDE.value,
            ugr.Terminal('"3"'),
        )
    )
    assert expected_rhs == conv_def_rule.rhs


def test_batchnorm_in_block() -> None:
    grammar = ugr.Grammar(
        """
        start : block
        block : conv norm
        conv  : "conv" "filter_count" ("1" | "2") "kernel_size" "2" "stride" "3"
        norm  : "batchnorm"
        """
    )

    nt = ugr.NonTerminal("norm")
    assert nt in grammar.nonterminals

    (block_exp,) = grammar.expansions(ugr.NonTerminal("block"))
    _, actual = block_exp.symbols

    assert nt == actual


def test_batchnorm_def() -> None:
    grammar = ugr.Grammar(
        """
        start : block
        block : conv norm
        conv  : "conv" "filter_count" ("1" | "2") "kernel_size" "2" "stride" "3"
        norm  : "batchnorm"
        """
    )

    nt = ugr.NonTerminal("norm")
    (actual_expansion,) = grammar.expansions(nt)

    expected_expansion = ugr.RuleOption((ugr.ExpectedTerminal.BATCHRNOM.value,))

    assert expected_expansion == actual_expansion


def test_pooling_layer_in_block() -> None:
    grammar = ugr.Grammar(
        """
        start : block
        block : conv pool
        conv  : "conv" "filter_count" "1" "kernel_size" ("2") "stride" ("3" | "4")
        pool  : "maxpool" "pool_size" "2" "stride" "3"
        """
    )

    nt = ugr.NonTerminal("pool")
    assert nt in grammar.nonterminals

    (block_exp,) = grammar.expansions(ugr.NonTerminal("block"))
    _, actual = block_exp.symbols

    assert nt == actual


@pytest.mark.slow
@given(
    pool_sizes=ugs.int_args(min_value=1, max_value=9),
    strides=ugs.int_args(min_value=1, max_value=9),
)
def test_maxpool_def(
    pool_sizes: ugs.GrammarArgs,
    strides: ugs.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "maxpool"'
        f' "pool_size" {pool_sizes.tokenstream}'
        f' "stride" {strides.tokenstream}'
    )
    grammar = ugr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(ugr.NonTerminal("start"))

    # asserts
    test_values = itertools.product(pool_sizes.parsed, strides.parsed)
    for expansion, (pool_size_term, stride_term) in zip(
        expansions, test_values, strict=True
    ):
        expected = ugr.RuleOption(
            (
                ugr.ExpectedTerminal.MAXPOOL.value,
                ugr.ExpectedTerminal.POOL_SIZE.value,
                pool_size_term,
                ugr.ExpectedTerminal.STRIDE.value,
                stride_term,
            )
        )
        assert expected == expansion


@pytest.mark.slow
@given(
    pool_sizes=ugs.int_args(min_value=1, max_value=9),
    strides=ugs.int_args(min_value=1, max_value=9),
)
def test_avgpool_def(
    pool_sizes: ugs.GrammarArgs,
    strides: ugs.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "avgpool"'
        f' "pool_size" {pool_sizes.tokenstream}'
        f' "stride" {strides.tokenstream}'
    )
    grammar = ugr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(ugr.NonTerminal("start"))

    test_values = itertools.product(pool_sizes.parsed, strides.parsed)
    for expansion, (pool_size_term, stride_term) in zip(
        expansions, test_values, strict=True
    ):
        expected = ugr.RuleOption(
            (
                ugr.ExpectedTerminal.AVGPOOL.value,
                ugr.ExpectedTerminal.POOL_SIZE.value,
                pool_size_term,
                ugr.ExpectedTerminal.STRIDE.value,
                stride_term,
            )
        )
        assert expected == expansion


def test_relu_def() -> None:
    grammar = ugr.Grammar(
        """
        start : "relu"
        """
    )
    (actual,) = grammar.expansions(ugr.NonTerminal("start"))
    expected = ugr.RuleOption((ugr.ExpectedTerminal.RELU.value,))
    assert expected == actual


@pytest.mark.slow
@given(
    learning_rate=ugs.float_args(min_value=0, exclude_min=True),
    momentum=ugs.float_args(min_value=0, exclude_min=True),
    nesterov=ugs.bool_args(),
)
def test_sgd_def(
    learning_rate: ugs.GrammarArgs,
    momentum: ugs.GrammarArgs,
    nesterov: ugs.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "sgd"'
        f' "learning_rate" {learning_rate.tokenstream}'
        f' "momentum" {momentum.tokenstream}'
        f' "nesterov" {nesterov.tokenstream}'
    )
    grammar = ugr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(ugr.NonTerminal("start"))

    test_values = itertools.product(
        learning_rate.parsed,
        momentum.parsed,
        nesterov.parsed,
    )

    for actual_expansion, (lr_term, mom_term, nest_term) in zip(
        expansions,
        test_values,
        strict=True,
    ):
        expected_symbols = (
            ugr.ExpectedTerminal.SGD.value,
            ugr.ExpectedTerminal.LEARNING_RATE.value,
            lr_term,
            ugr.ExpectedTerminal.MOMENTUM.value,
            mom_term,
            ugr.ExpectedTerminal.NESTEROV.value,
            nest_term,
        )
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion


@pytest.mark.slow
@given(
    learning_rate=ugs.float_args(min_value=0, exclude_min=True),
    beta1=ugs.float_args(min_value=0, exclude_min=True),
    beta2=ugs.float_args(min_value=0, exclude_min=True),
    epsilon=ugs.float_args(min_value=0, exclude_min=True),
    amsgrad=ugs.bool_args(),
)
def test_adam_def(
    learning_rate: ugs.GrammarArgs,
    beta1: ugs.GrammarArgs,
    beta2: ugs.GrammarArgs,
    epsilon: ugs.GrammarArgs,
    amsgrad: ugs.GrammarArgs,
) -> None:
    # setup
    raw_grammar = (
        'start : "adam"'
        f' "learning_rate" {learning_rate.tokenstream}'
        f' "beta1" {beta1.tokenstream}'
        f' "beta2" {beta2.tokenstream}'
        f' "epsilon" {epsilon.tokenstream}'
        f' "amsgrad" {amsgrad.tokenstream}'
    )
    grammar = ugr.Grammar(raw_grammar)

    # function under test
    expansions = grammar.expansions(ugr.NonTerminal("start"))

    test_values = itertools.product(
        learning_rate.parsed,
        beta1.parsed,
        beta2.parsed,
        epsilon.parsed,
        amsgrad.parsed,
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
            ugr.ExpectedTerminal.ADAM.value,
            ugr.ExpectedTerminal.LEARNING_RATE.value,
            lr_term,
            ugr.ExpectedTerminal.BETA1.value,
            beta1_term,
            ugr.ExpectedTerminal.BETA2.value,
            beta2_term,
            ugr.ExpectedTerminal.EPSILON.value,
            epsilon_term,
            ugr.ExpectedTerminal.AMSGRAD.value,
            amsgrad_term,
        )
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion


@given(mode=ugs.flip_modes())
def test_random_flip(mode: ugs.GrammarArgs) -> None:
    """Can parse middle grammar layer definition: random_flip."""

    raw_grammar = f'start : "random_flip" {mode.tokenstream}'
    grammar = ugr.Grammar(raw_grammar)

    expansions = grammar.expansions(ugr.NonTerminal("start"))
    for actual_expansion, terminal in zip(expansions, mode.parsed, strict=True):
        expected_symbols = (ugr.ExpectedTerminal.RANDOM_FLIP.value, terminal)
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion


@given(rotation=ugs.float_args(min_value=0))
def test_random_rotation(rotation: ugs.GrammarArgs) -> None:
    """Can parse middle grammar layer definition: random_rotation."""

    raw_grammar = f'start : "random_rotation" {rotation.tokenstream}'
    grammar = ugr.Grammar(raw_grammar)

    expansions = grammar.expansions(ugr.NonTerminal("start"))
    for actual_expanion, terminal in zip(expansions, rotation.parsed):
        expected_symbols = (ugr.ExpectedTerminal.RANDOM_ROTATION.value, terminal)
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expanion


@given(
    height=ugs.int_args(min_value=1, max_value=3),
    width=ugs.int_args(min_value=1, max_value=3),
)
def test_resizing(height: ugs.GrammarArgs, width: ugs.GrammarArgs) -> None:
    """Can parse middle grammar layer definition: resizing."""

    raw_grammar = (
        f'start : "resizing" "height" {height.tokenstream} "width" {width.tokenstream}'
    )
    grammar = ugr.Grammar(raw_grammar)

    expansions = grammar.expansions(ugr.NonTerminal("start"))
    test_values = itertools.product(height.parsed, width.parsed)
    for actual_expansion, terminals in zip(
        expansions,
        test_values,
        strict=True,
    ):
        height_term, width_term = terminals
        expected_symbols = (
            ugr.ExpectedTerminal.RESIZING.value,
            ugr.ExpectedTerminal.HEIGHT.value,
            height_term,
            ugr.ExpectedTerminal.WIDTH.value,
            width_term,
        )
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion


@given(
    height=ugs.int_args(min_value=1, max_value=3),
    width=ugs.int_args(min_value=1, max_value=3),
)
def test_random_crop(height: ugs.GrammarArgs, width: ugs.GrammarArgs) -> None:
    """Can parse middle grammar layer definition: random_crop."""

    raw_grammar = f'start : "random_crop" "height" {height.tokenstream} "width" {width.tokenstream}'
    grammar = ugr.Grammar(raw_grammar)

    expansions = grammar.expansions(ugr.NonTerminal("start"))
    test_values = itertools.product(height.parsed, width.parsed)
    for actual_expansion, terminals in zip(
        expansions,
        test_values,
        strict=True,
    ):
        height_term, width_term = terminals
        expected_symbols = (
            ugr.ExpectedTerminal.RANDOM_CROP.value,
            ugr.ExpectedTerminal.HEIGHT.value,
            height_term,
            ugr.ExpectedTerminal.WIDTH.value,
            width_term,
        )
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expansion


@given(
    factor=ugs.float_args(min_value=-1, max_value=1),
)
def test_random_translation(factor: ugs.GrammarArgs) -> None:
    """Can parse middle grammar layer definition: random_translation."""

    raw_grammar = f'start : "random_translation" {factor.tokenstream}'
    grammar = ugr.Grammar(raw_grammar)

    expansions = grammar.expansions(ugr.NonTerminal("start"))
    for actual_expanion, terminal in zip(expansions, factor.parsed):
        expected_symbols = (ugr.ExpectedTerminal.RANDOM_TRANSLATION.value, terminal)
        expected_expansion = ugr.RuleOption(expected_symbols)
        assert expected_expansion == actual_expanion


def test_prelu() -> None:
    """Can parse middle grammar layer definition: prelu."""
    grammar = ugr.Grammar('start : "prelu"')
    (actual_expansion,) = grammar.expansions(ugr.NonTerminal("start"))
    expected_expansion = ugr.RuleOption((ugr.ExpectedTerminal.PRELU.value,))
    assert expected_expansion == actual_expansion
