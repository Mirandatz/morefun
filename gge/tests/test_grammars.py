from pathlib import Path

import pytest

from gge.grammars import Grammar, NonTerminal, RuleOption, Terminal

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def raw_grammar() -> str:
    return """
    start: simple_block complex_block simple_dense

    simple_block : simple_conv simple_dense
    simple_conv  : "conv2d" "filter_count" (4) "kernel_size" (3) "stride" (1)
    simple_dense : "dense" (16)

    complex_block : complex_conv~2 complex_dense_block~1..3
                  | simple_conv complex_dense_block~2..2 simple_conv
                  | simple_conv

    complex_conv  : "conv2d" "filter_count" (4 | 8 | 6) "kernel_size" (3 | 5) "stride" (1 | 2)

    complex_dense_block : complex_dense_layer~3..5 _droperino
    complex_dense_layer : "dense" (16 | 32 | 64)
    _droperino          : "dropout" (0.3 | 0.5 | 0.7)
    """


@pytest.fixture
def raw_metagrammar() -> str:
    metagrammar_path = DATA_DIR / "metagrammar.lark"
    return metagrammar_path.read_text()


@pytest.fixture
def sample_grammar(raw_grammar: str, raw_metagrammar: str) -> Grammar:
    return Grammar(
        raw_grammar=raw_grammar,
        raw_metagrammar=raw_metagrammar,
    )


def test_start_symbol(sample_grammar: Grammar) -> None:
    expected = NonTerminal("start")
    actual = sample_grammar.start_symbol

    assert expected == actual


def test_terminals(sample_grammar: Grammar) -> None:
    terminals = [
        '"conv2d"',
        '"filter_count"',
        "4",
        '"kernel_size"',
        "3",
        '"stride"',
        "1",
        '"dense"',
        "16",
        "8",
        "6",
        "5",
        "2",
        "32",
        "64",
        '"dropout"',
        "0.3",
        "0.5",
        "0.7",
    ]
    expected = {Terminal(t) for t in terminals}
    assert expected == set(sample_grammar.terminals)


def test_non_terminals(sample_grammar: Grammar) -> None:
    texts = [
        "start",
        "simple_block",
        "complex_block",
        "simple_dense",
        "simple_conv",
        "complex_conv",
        "complex_dense_block",
        "complex_dense_layer",
        "_droperino",
    ]
    expected = {NonTerminal(t) for t in texts}
    actual = set(sample_grammar.nonterminals)

    assert expected == actual


def test_start_symbol_expansion(sample_grammar: Grammar) -> None:
    start = NonTerminal("start")

    expansions = sample_grammar.expansions(start)
    assert len(expansions) == 1

    actual = expansions[0]
    expected = RuleOption(
        (
            NonTerminal("simple_block"),
            NonTerminal("complex_block"),
            NonTerminal("simple_dense"),
        )
    )

    assert expected == actual


def test_complex_symbol_expansion(sample_grammar: Grammar) -> None:
    nt = NonTerminal("complex_block")
    actual = sample_grammar.expansions(nt)

    cc = NonTerminal("complex_conv")
    cdb = NonTerminal("complex_dense_block")
    sc = NonTerminal("simple_conv")

    expected = (
        RuleOption((cc, cc, cdb)),
        RuleOption((cc, cc, cdb, cdb)),
        RuleOption((cc, cc, cdb, cdb, cdb)),
        RuleOption((sc, cdb, cdb, sc)),
        RuleOption((sc,)),
    )

    assert expected == actual


def test_non_terminal_with_single_expansion(sample_grammar: Grammar) -> None:
    nt = NonTerminal("simple_dense")
    exps = sample_grammar.expansions(nt)
    assert len(exps) == 1

    actual = exps[0]
    expected = RuleOption((Terminal('"dense"'), Terminal("16")))

    assert actual == expected
