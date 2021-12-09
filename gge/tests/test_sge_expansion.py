from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar, NonTerminal


def test_simple_grammar() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c
        b : "dense" (2)
        c : "dense" (4)
        """
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


def test_simple_repetition() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c~5 b
        b : "dense" (4)
        c : "dense" (9)
        """
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


def test_ranged_repetion() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b~5..7
        b : "dense" (37)
        """
    )

    start, a, b = [NonTerminal(text) for text in ["start", "a", "b"]]

    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(start, grammar)
    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(a, grammar)
    assert 7 == sge.max_nr_of_times_nonterminal_can_be_expanded(b, grammar)
