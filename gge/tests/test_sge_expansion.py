import pytest

from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar, NonTerminal
from gge.tests.grammar_fixtures import raw_metagrammar

assert raw_metagrammar is not None


def test_simple_grammar(raw_metagrammar: str) -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c
        b : "dense" (2)
        c : "dense" (4)
        """,
        raw_metagrammar=raw_metagrammar,
    )

    a, b, c = [
        NonTerminal(text)
        for text in [
            "a",
            "b",
            "c",
        ]
    ]

    # shorter alias
    test_func = sge.max_nr_of_times_nonterminal_can_be_expanded

    assert 1 == test_func(a, grammar)
    assert 1 == test_func(b, grammar)
    assert 1 == test_func(c, grammar)


def test_simple_repetition(raw_metagrammar: str) -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c~5 b
        b : "dense" (4)
        c : "dense" (9)
        """,
        raw_metagrammar=raw_metagrammar,
    )

    a, b, c = [
        NonTerminal(text)
        for text in [
            "a",
            "b",
            "c",
        ]
    ]

    # shorter alias
    test_func = sge.max_nr_of_times_nonterminal_can_be_expanded

    assert 1 == test_func(a, grammar)
    assert 2 == test_func(b, grammar)
    assert 5 == test_func(c, grammar)


@pytest.mark.xfail
def test_ranged_repetion(raw_metagrammar: str) -> None:
    grammar = Grammar(
        raw_grammar=r"""
        a : c~5..7
        c : "c"
        """,
        raw_metagrammar=raw_metagrammar,
    )

    a, b, c = [NonTerminal(text) for text in ["a", "b", "c"]]

    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(a, grammar)
    assert 2 == sge.max_nr_of_times_nonterminal_can_be_expanded(b, grammar)
    assert 7 == sge.max_nr_of_times_nonterminal_can_be_expanded(c, grammar)
