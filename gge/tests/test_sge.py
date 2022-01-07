import gge.grammars as gr
import gge.structured_grammatical_evolution as sge


def test_nonrecursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c
        b : "conv2d" "filter_count" 64 "kernel_size" 5 "stride" 2
        c : "conv2d" "filter_count" 128 "kernel_size" 3 "stride" 1
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


def test_simple_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c | a
        b : b | c
        c : "conv2d" "filter_count" (2) "kernel_size" (1) "stride" (2)
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


def test_complex_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c d
        b : c d
        c : d a
        d : "conv2d" "filter_count" (2) "kernel_size" (1) "stride" (2)
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


def test_nasty_recursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b | c a | k
        b : c | k
        c : k
        k : "conv2d" "filter_count" (2) "kernel_size" (1) "stride" (2)
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
