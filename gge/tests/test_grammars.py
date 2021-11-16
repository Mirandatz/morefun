import sys
from typing import Any

import lark
import pytest

from gge.grammars import Grammar, NonTerminal, Terminal


@pytest.fixture
def raw_grammar() -> str:
    return """
    startberg : simple "0.3" complex "1e-07"
    simple    : "another_terminal"
    complex   : simple? "maybe_text"? repeaters
    repeaters : a~1..3 b~5
    a         : "a"
    b         : "b"
    """


@pytest.fixture
def sample_grammar(raw_grammar: str) -> Grammar:
    return Grammar(raw_grammar)


@pytest.fixture
def sample_parser(raw_grammar: str) -> lark.lark:  # type: ignore
    return lark.Lark(raw_grammar, start="startberg", keep_all_tokens=True)


class TestGrammar:
    @pytest.mark.skipif(
        "typeguard" not in sys.modules, reason="requires the Pandas library"
    )
    def test_terminal_constructor_invalid_types(self) -> None:
        invalid_types: list[Any] = [None, 2, 2.3, [], ["a"], tuple(), tuple("l")]
        for text in invalid_types:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                Terminal(text)

    def test_start_symbol(self, sample_grammar: Grammar) -> None:
        expected = NonTerminal("startberg")
        actual = sample_grammar.start_symbol

        assert expected == actual

    def test_terminals(self, sample_grammar: Grammar) -> None:
        texts = ["0.3", "1e-07", "another_terminal", "maybe_text", "a", "b"]
        expected = {Terminal(t) for t in texts}
        assert expected == set(sample_grammar.terminals)

    def test_non_terminals(self, sample_grammar: Grammar) -> None:
        texts = ["startberg", "simple", "complex", "repeaters", "a", "b"]
        expected = {NonTerminal(t) for t in texts}
        actual = set(sample_grammar.nonterminals)

        assert expected == actual

    def test_start_symbol_expansion(self, sample_grammar: Grammar) -> None:
        start = NonTerminal("startberg")

        expansions = sample_grammar.expansions(start)
        assert len(expansions) == 1

        actual = expansions[0]

        expected = (
            NonTerminal("simple"),
            Terminal("0.3"),
            NonTerminal("complex"),
            Terminal("1e-07"),
        )

        assert expected == actual

    def test_complex_symbol_expansion(self, sample_grammar: Grammar) -> None:
        nt = NonTerminal("complex")
        actual = sample_grammar.expansions(nt)

        s = NonTerminal("simple")
        mt = Terminal("maybe_text")
        r = NonTerminal("repeaters")

        expected = (
            (r,),
            (mt, r),
            (s, r),
            (s, mt, r),
        )

        assert expected == actual

    def test_non_terminal_with_single_expansion(
        self,
        sample_grammar: Grammar,
    ) -> None:
        nt = NonTerminal("a")
        exps = sample_grammar.expansions(nt)
        assert len(exps) == 1

        actual = exps[0]
        expected = (Terminal("a"),)

        assert actual == expected

    def test_raise_if_non_terminal_not_used(self) -> None:
        raw = """
        a : b c
        b : "B"
        c : "C"
        d : "D"
        """

        with pytest.raises(ValueError):
            Grammar(raw)

    def test_repeated_symbol_expansion(self, sample_grammar: Grammar) -> None:
        actual = sample_grammar.expansions(NonTerminal("repeaters"))

        a = NonTerminal("a")
        b = NonTerminal("b")
        expected = (
            (
                a,
                b,
                b,
                b,
                b,
                b,
            ),
            (a, a, b, b, b, b, b),
            (a, a, a, b, b, b, b, b),
        )

        assert expected == actual


class TestCompatibilityWithLark:
    def test_terminals(
        self,
        sample_grammar: Grammar,
        sample_parser: lark.Lark,
    ) -> None:
        expected = set(t.pattern.value for t in sample_parser.terminals)
        actual = set(t.text for t in sample_grammar.terminals)

        assert expected == actual

    def test_nonterminals(
        self,
        sample_grammar: Grammar,
        sample_parser: lark.Lark,
    ) -> None:
        expected = set(rule.origin.name for rule in sample_parser.rules)  # type: ignore
        actual = set(t.text for t in sample_grammar.nonterminals)

        assert expected == actual
