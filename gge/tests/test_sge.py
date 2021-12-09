from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar, NonTerminal


def test_nonrecursive_grammar() -> None:
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c
        b : "dense" (3)
        c : "dense" (8)
        """
    )

    a = NonTerminal("a")
    b = NonTerminal("b")
    c = NonTerminal("c")

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
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c | a
        b : b | c
        c : "dropout" (0.3)
        """
    )

    a = NonTerminal("a")
    b = NonTerminal("b")
    c = NonTerminal("c")

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
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b c d
        b : c d
        c : d a
        d : "conv2d" "filter_count" (2) "kernel_size" (1) "stride" (2)
        """
    )

    a = NonTerminal("a")
    b = NonTerminal("b")
    c = NonTerminal("c")
    d = NonTerminal("d")

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
    grammar = Grammar(
        raw_grammar=r"""
        start : a
        a : b | c a | k
        b : c | k
        c : k
        k : "dropout" (0.0)
        """
    )
    a = NonTerminal("a")
    b = NonTerminal("b")
    c = NonTerminal("c")

    assert sge.can_expand(a, a, grammar)
    assert sge.can_expand(a, b, grammar)
    assert sge.can_expand(a, c, grammar)

    assert sge.can_expand(b, c, grammar)
    assert not sge.can_expand(b, a, grammar)
    assert not sge.can_expand(b, b, grammar)

    assert not sge.can_expand(c, a, grammar)
    assert not sge.can_expand(c, b, grammar)
    assert not sge.can_expand(c, c, grammar)
