import gge.grammars as gr
import gge.structured_grammatical_evolution as sge

# auto-import fixture
from gge.tests.fixtures import remove_logger_sinks  # noqa


def test_can_expand_nonrecursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c
        b : "conv" "filter_count" "64" "kernel_size" "5" "stride" "2"
        c : "conv" "filter_count" "128" "kernel_size" "3" "stride" "1"
        """
    )

    a = gr.NonTerminal("a")
    b = gr.NonTerminal("b")
    c = gr.NonTerminal("c")

    assert not sge.can_expand(a, a, grammar)
    assert sge.can_expand(a, b, grammar)
    assert sge.can_expand(a, c, grammar)

    assert not sge.can_expand(b, a, grammar)
    assert not sge.can_expand(b, b, grammar)
    assert not sge.can_expand(b, c, grammar)

    assert not sge.can_expand(c, a, grammar)
    assert not sge.can_expand(c, b, grammar)
    assert not sge.can_expand(c, c, grammar)


def test_can_expand_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c | a
        b : b | c
        c : "conv" "filter_count" ("2") "kernel_size" ("1") "stride" ("2")
        """
    )

    a = gr.NonTerminal("a")
    b = gr.NonTerminal("b")
    c = gr.NonTerminal("c")

    assert sge.can_expand(a, a, grammar)
    assert sge.can_expand(a, b, grammar)
    assert sge.can_expand(a, c, grammar)

    assert not sge.can_expand(b, a, grammar)
    assert sge.can_expand(b, b, grammar)
    assert sge.can_expand(b, c, grammar)

    assert not sge.can_expand(c, a, grammar)
    assert not sge.can_expand(c, b, grammar)
    assert not sge.can_expand(c, c, grammar)


def test_can_expand_complex_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c d
        b : c d
        c : d a
        d : "conv" "filter_count" ("2") "kernel_size" ("1") "stride" ("2")
        """
    )

    a = gr.NonTerminal("a")
    b = gr.NonTerminal("b")
    c = gr.NonTerminal("c")
    d = gr.NonTerminal("d")

    assert sge.can_expand(a, a, grammar)
    assert sge.can_expand(a, b, grammar)
    assert sge.can_expand(a, c, grammar)
    assert sge.can_expand(a, d, grammar)

    assert sge.can_expand(b, a, grammar)
    assert sge.can_expand(b, b, grammar)
    assert sge.can_expand(b, c, grammar)
    assert sge.can_expand(b, b, grammar)

    assert sge.can_expand(c, a, grammar)
    assert sge.can_expand(c, b, grammar)
    assert sge.can_expand(c, c, grammar)
    assert sge.can_expand(c, b, grammar)

    assert not sge.can_expand(d, a, grammar)
    assert not sge.can_expand(d, b, grammar)
    assert not sge.can_expand(d, c, grammar)
    assert not sge.can_expand(d, b, grammar)


def test_can_expand_nasty_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b | c a | k
        b : c | k
        c : k
        k : "conv" "filter_count" ("2") "kernel_size" ("1") "stride" ("2")
        """
    )
    a = gr.NonTerminal("a")
    b = gr.NonTerminal("b")
    c = gr.NonTerminal("c")

    assert sge.can_expand(a, a, grammar)
    assert sge.can_expand(a, b, grammar)
    assert sge.can_expand(a, c, grammar)

    assert sge.can_expand(b, c, grammar)
    assert not sge.can_expand(b, a, grammar)
    assert not sge.can_expand(b, b, grammar)

    assert not sge.can_expand(c, a, grammar)
    assert not sge.can_expand(c, b, grammar)
    assert not sge.can_expand(c, c, grammar)


def test_max_nr_of_times_nonterminal_can_be_expanded_simple_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c
        b : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        c : "conv" "filter_count" "4" "kernel_size" "5" "stride" "6"
        """
    )

    a, b, c = [
        gr.NonTerminal(text)
        for text in [
            "a",
            "b",
            "c",
        ]
    ]

    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(a, grammar)
    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(b, grammar)
    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(c, grammar)


def test_max_nr_of_times_nonterminal_can_be_expanded_simple_repetition() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c~5 b
        b : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        c : "conv" "filter_count" "3" "kernel_size" "2" "stride" "1"
        """
    )

    a, b, c = [
        gr.NonTerminal(text)
        for text in [
            "a",
            "b",
            "c",
        ]
    ]

    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(a, grammar)
    assert 2 == sge.max_nr_of_times_nonterminal_can_be_expanded(b, grammar)
    assert 5 == sge.max_nr_of_times_nonterminal_can_be_expanded(c, grammar)


def test_max_nr_of_times_nonterminal_can_be_expanded_ranged_repetion() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b~5..7
        b : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        """
    )

    start, a, b = [gr.NonTerminal(text) for text in ["start", "a", "b"]]

    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(start, grammar)
    assert 1 == sge.max_nr_of_times_nonterminal_can_be_expanded(a, grammar)
    assert 7 == sge.max_nr_of_times_nonterminal_can_be_expanded(b, grammar)


def test_check_recursion_nonrecursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c
        b : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        c : "conv" "filter_count" "4" "kernel_size" "5" "stride" "6"
        """
    )

    assert not sge.grammar_is_recursive(grammar)


def test_check_recursion_simple_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c | a
        b : b | c
        c : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        """
    )

    assert sge.grammar_is_recursive(grammar)


def test_check_recursion_complex_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c d
        b : c d
        c : d a
        d : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        """
    )

    assert sge.grammar_is_recursive(grammar)


def test_check_recursion_nasty_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b | c d | k
        b : c | k | j
        c : k
        d : c | b
        j : a
        k : "conv" "filter_count" ("2") "kernel_size" ("5") "stride" ("2")
        """
    )

    assert sge.grammar_is_recursive(grammar)
