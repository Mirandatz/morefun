from hypothesis import given

import gge.grammars as gr
import gge.tests.metagrammar_strategies as ms

START = gr.NonTerminal("start")

CONV2D = gr.Terminal('"conv2d"')
FILTER_COUNT = gr.Terminal('"filter_count"')
KERNEL_SIZE = gr.Terminal('"kernel_size"')
STRIDE = gr.Terminal('"stride"')

BATCHRNOM = gr.Terminal('"batchnorm"')

RELU = gr.Terminal('"relu"')
GELU = gr.Terminal('"gelu"')
SWISH = gr.Terminal('"swish"')

POOL2D = gr.Terminal('"pool2d"')
TYPE = gr.Terminal('"type"')
MAX = gr.Terminal('"max"')


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

    names = {CONV2D, FILTER_COUNT, KERNEL_SIZE, STRIDE, RELU, SWISH}
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
            CONV2D,
            FILTER_COUNT,
            gr.Terminal("1"),
            KERNEL_SIZE,
            gr.Terminal("2"),
            STRIDE,
            gr.Terminal("3"),
        )
    )
    assert expected_first == actual_first

    expected_second = gr.RuleOption(
        (
            CONV2D,
            FILTER_COUNT,
            gr.Terminal("2"),
            KERNEL_SIZE,
            gr.Terminal("2"),
            STRIDE,
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
            CONV2D,
            FILTER_COUNT,
            gr.Terminal("1"),
            KERNEL_SIZE,
            gr.Terminal("2"),
            STRIDE,
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

    expected_expansion = gr.RuleOption((BATCHRNOM,))

    assert expected_expansion == actual_expansion


def test_pooling_layer_in_block() -> None:
    grammar = gr.Grammar(
        """
        start : block
        block : conv pool
        conv  : "conv2d" "filter_count" 1 "kernel_size" (2) "stride" (3 | 4)
        pool  : "pool2d" "max" "stride" 2
        """
    )

    nt = gr.NonTerminal("pool")
    assert nt in grammar.nonterminals

    (block_exp,) = grammar.expansions(gr.NonTerminal("block"))
    _, actual = block_exp.symbols

    assert nt == actual


def test_pooling_layer_def() -> None:
    grammar = gr.Grammar(
        """
        start : block
        block : conv pool
        conv  : "conv2d" "filter_count" 1 "kernel_size" (2) "stride" (3 | 4)
        pool  : "pool2d" "max" "stride" 2
        """
    )

    (actual,) = grammar.expansions(gr.NonTerminal("pool"))
    expected = gr.RuleOption((POOL2D, MAX, gr.Terminal("2")))

    assert expected == actual


@given(data=ms.pool2ds())
def test_pool_layer_def(data: ms.Pool2DTestData) -> None:
    raw_grammar = ms.make_raw_grammar(data)
    grammar = gr.Grammar(raw_grammar)

    (actual,) = grammar.expansions(data.nonterminal)
    expected = data.expansion()
    assert expected == actual


def test_relu_def() -> None:
    grammar = gr.Grammar(
        """
        start : "relu"
        """
    )
    (actual,) = grammar.expansions(gr.NonTerminal("start"))
    expected = gr.RuleOption((RELU,))
    assert expected == actual
