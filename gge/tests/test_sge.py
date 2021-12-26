import gge.grammars as gr
import gge.structured_grammatical_evolution as sge


def test_nonrecursive_grammar() -> None:
    grammar = gr.Grammar(
        """
        start : a
        a : b c
        b : "dense" (3)
        c : "dense" (8)
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
        c : "dropout" (0.3)
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
        d : "conv2d" "filter_count" (2) "kernel_size" (1) "stride" (2) "relu"
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
        raw_grammar=r"""
        start : a
        a : b | c a | k
        b : c | k
        c : k
        k : "dropout" (0.0)
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
