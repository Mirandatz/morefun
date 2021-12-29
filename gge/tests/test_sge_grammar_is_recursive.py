from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar


def test_nonrecursive_grammar() -> None:
    grammar = Grammar(
        """
        start : a
        a : b c
        b : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "gelu"
        c : "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6 "gelu"
        """
    )

    assert not sge.grammar_is_recursive(grammar)


def test_simple_recursive_grammar() -> None:
    grammar = Grammar(
        """
        start : a
        a : b c | a
        b : b | c
        c : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "relu"
        """
    )

    assert sge.grammar_is_recursive(grammar)


def test_complex_recursive_grammar() -> None:
    grammar = Grammar(
        """
        start : a
        a : b c d
        b : c d
        c : d a
        d : "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "swish"
        """
    )

    assert sge.grammar_is_recursive(grammar)


def test_nasty_recursive_grammar() -> None:
    grammar = Grammar(
        """
        start : a
        a : b | c d | k
        b : c | k | j
        c : k
        d : c | b
        j : a
        k : "conv2d" "filter_count" (2) "kernel_size" (5) "stride" (2) "gelu"
        """
    )

    assert sge.grammar_is_recursive(grammar)
