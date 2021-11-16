from gge import structured_grammatical_evolution as sge
from gge.grammars import Grammar, NonTerminal


class Test_can_expand:
    def test_nonrecursive_grammar(self) -> None:
        grammar = Grammar(
            """
            a : b c
            b : "b"
            c : "c"
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

    def test_simple_recursive_grammar(self) -> None:
        grammar = Grammar(
            """
            a : b c | a
            b : b | c | "b"
            c : "c"
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

    def test_complex_recursive_grammar(self) -> None:
        grammar = Grammar(
            """
        a : b c d
        b : c d
        c : d a
        d : "d"
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

    def test_nasty_recursive_grammar(self) -> None:
        grammar = Grammar(
            """
        a : b | c a | "k"
        b : c | "k"
        c : "k"
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


class Test_grammar_is_recursive:
    def test_nonrecursive_grammar(self) -> None:
        grammar = Grammar(
            """
        a : b c
        b : "b"
        c : "c"
        """
        )

        assert not sge.grammar_is_recursive(grammar)

    def test_simple_recursive_grammar(self) -> None:
        grammar = Grammar(
            """
            a : b c | a
            b : b | c | "b"
            c : "c"
            """
        )

        assert sge.grammar_is_recursive(grammar)

    def test_complex_recursive_grammar(self) -> None:
        grammar = Grammar(
            """
        a : b c d
        b : c d
        c : d a
        d : "d"
        """
        )

        assert sge.grammar_is_recursive(grammar)

    def test_nasty_recursive_grammar(self) -> None:
        grammar = Grammar(
            """
        a : b | c d? | "k"
        b : c | "k" | j
        c : "k"
        d : c | b
        j : a
        """
        )

        assert sge.grammar_is_recursive(grammar)


class Test_max_nr_of_times_nonterminal_can_be_expanded:
    def test_simple_grammar(self) -> None:
        grammar = Grammar(
            """
        a : b c
        b : "b"
        c : "c"
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

    def test_optional(self) -> None:
        grammar = Grammar(
            """
        a : b? c b?
        b : "b"
        c : "c"
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
        assert 1 == test_func(c, grammar)

    def test_simple_repetition(self) -> None:
        grammar = Grammar(
            """
        a : b? c~5 b?
        b : "b"
        c : "c"
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

    def test_ranged_repetion(self) -> None:
        grammar = Grammar(
            """
        a : b? c~5..7 b?
        b : "b"
        c : "c"
        """
        )

        a, b, c = [NonTerminal(text) for text in ["a", "b", "c"]]

        # shorter alias
        test_func = sge.max_nr_of_times_nonterminal_can_be_expanded

        assert 1 == test_func(a, grammar)
        assert 2 == test_func(b, grammar)
        assert 7 == test_func(c, grammar)

    def test_ranged_inside_ranged(self) -> None:
        grammar = Grammar(
            """
        a : b? c~5..7 b?
        b : "b"
        c : d~2..4
        d : "d"
        """
        )

        a, b, c, d = [NonTerminal(text) for text in ["a", "b", "c", "d"]]

        # shorter alias
        test_func = sge.max_nr_of_times_nonterminal_can_be_expanded

        assert 1 == test_func(a, grammar)
        assert 2 == test_func(b, grammar)
        assert 7 == test_func(c, grammar)
        assert 4 * 7 == test_func(d, grammar)

    def test_optional_inside_range(self) -> None:
        grammar = Grammar(
            """
        a : b? c~5..7 b?
        b : "b"
        c : d d? d
        d : "d"
        """
        )

        a, b, c, d = [NonTerminal(text) for text in ["a", "b", "c", "d"]]

        # shorter alias
        test_func = sge.max_nr_of_times_nonterminal_can_be_expanded

        assert 1 == test_func(a, grammar)
        assert 2 == test_func(b, grammar)
        assert 7 == test_func(c, grammar)
        assert 3 * 7 == test_func(d, grammar)

    def test_range_inside_optional(self) -> None:
        grammar = Grammar(
            """
        a : b? c? b?
        b : "b"
        c : d~2..5
        d : "d"
        """
        )

        a, b, c, d = [NonTerminal(text) for text in ["a", "b", "c", "d"]]

        # shorter alias
        test_func = sge.max_nr_of_times_nonterminal_can_be_expanded

        assert 1 == test_func(a, grammar)
        assert 2 == test_func(b, grammar)
        assert 1 == test_func(c, grammar)
        assert 5 == test_func(d, grammar)
