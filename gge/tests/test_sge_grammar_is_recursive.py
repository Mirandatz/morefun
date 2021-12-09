from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar


def test_nonrecursive_grammar() -> None:
    grammar = Grammar(
        raw_grammar=r"""
            start : a
            a : b c
            b : "dense" (40)
            c : "dropout" (0.3)"""
    )

    assert not sge.grammar_is_recursive(grammar)


def test_simple_recursive_grammar() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c | a
        b : b | c
        c : "dense" (30)
        """
    )

    assert sge.grammar_is_recursive(grammar)


def test_complex_recursive_grammar() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c d
        b : c d
        c : d a
        d : "dense" (2)
        """
    )

    assert sge.grammar_is_recursive(grammar)


def test_nasty_recursive_grammar() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b | c d | k
        b : c | k | j
        c : k
        d : c | b
        j : a
        k : "conv2d" "filter_count" (2) "kernel_size" (5) "stride" (2)
        """
    )

    assert sge.grammar_is_recursive(grammar)
