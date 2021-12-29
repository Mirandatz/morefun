from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar, NonTerminal


def test_simple_grammar() -> None:
    grammar = Grammar(
        """
        start : a
        a : b c
        b : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "relu"
        c : "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6 "relu"
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
        """
        start : a
        a : b c~5 b
        b : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "relu"
        c : "conv2d" "filter_count" 3 "kernel_size" 2 "stride" 1 "gelu"
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
        """
        start : a
        a : b~5..7
        b : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "relu"
        """
    )

    start, a, b = [NonTerminal(text) for text in ["start", "a", "b"]]

    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(start, grammar)
    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(a, grammar)
    assert 7 == sge.max_nr_of_times_nonterminal_can_be_expanded(b, grammar)
