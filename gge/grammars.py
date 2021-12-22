import functools
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import lark
import typeguard

METAGRAMMAR_PATH = Path(__file__).parent.parent / "data" / "metagrammar.lark"


def _can_be_parsed_as_float(value: str) -> bool:
    try:
        _ = float(value)
        return True
    except ValueError:
        return False


def _is_valid_name(value: str) -> bool:
    assert len(value) >= 1

    first = value[0]
    if not (first == "_" or first.isalpha()):
        return False

    chars_are_valid = (c == "_" or c.isalnum() for c in value[1:])
    return all(chars_are_valid)


@typeguard.typechecked
@dataclass(order=True, frozen=True)
class NonTerminal:
    text: str

    def __post_init__(self) -> None:
        error_msg = (
            "text must be non empty, "
            "can only contain lowercase alphanum and underscores, "
            "and the first character must be an underscore or alpha. "
            f"invalid text: {self.text}"
        )

        if not _is_valid_name(self.text):
            raise ValueError(error_msg)

        if not self.text.islower():
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        return f"NT({self.text})"


@typeguard.typechecked
@dataclass(order=True, frozen=True)
class Terminal:
    text: str

    def __post_init__(self) -> None:
        if _can_be_parsed_as_float(self.text):
            return

        if _is_valid_name(self.text):
            return

        error_msg = (
            "text must be non empty, "
            "can only contain lowercase alphanum and underscores, "
            "and the first character must be an underscore or alpha "
            "(unless the entire text is surround by double quotes). "
            f"invalid text: {self.text}"
        )

        # the rest of the code checks if a text is a quoted "valid name"
        if len(self.text) <= 2:
            raise ValueError(error_msg)

        if not (self.text[0] == self.text[-1] == '"'):
            raise ValueError(error_msg)

        core_is_valid = all(c.isalnum() or c == "_" for c in self.text[1:-2])
        if not core_is_valid:
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        return f"T({self.text})"


Symbol = Union[Terminal, NonTerminal]


@typeguard.typechecked
@dataclass(order=True, frozen=True)
class RuleOption:
    """
    represents one possible expansion of a rule, i.e. "the stuff separated by `|`"
    to exemplify, consider the following rule:
    expr = x | y | x * y
    it contains 3 options:
    option0 = x
    option1 = y
    option2 = x * y
    """

    symbols: tuple[Symbol, ...]

    def __post_init__(self) -> None:
        assert len(self.symbols) >= 1

    def __repr__(self) -> str:
        all_options = ",".join(repr(s) for s in self.symbols)
        return f"RuleOption({all_options})"


@typeguard.typechecked
@dataclass(order=True, frozen=True)
class ProductionRule:
    lhs: NonTerminal
    rhs: RuleOption

    def __repr__(self) -> str:
        return f"Rule({self.lhs}->{self.rhs})"


class Grammar:
    def __init__(
        self,
        raw_grammar: str,
        *,
        metagrammar_path: Path = METAGRAMMAR_PATH,
    ) -> None:
        assert metagrammar_path.is_file()

        parser = lark.Lark.open(str(metagrammar_path), parser="lalr")
        tree = parser.parse(raw_grammar)
        transformer = GrammarTransformer()
        transformer.transform(tree)

        self._raw_grammar = raw_grammar

        self._nonterminals = transformer.get_nonterminals()
        self._terminals = transformer.get_terminals()
        self._rules = transformer.get_rules()
        self._start_symbol = transformer.get_start_symbol()

        self._as_tuple = (
            self._nonterminals,
            self._terminals,
            self._rules,
            self._start_symbol,
        )

        self._hash = hash(self._as_tuple)

        validate_grammar_components(
            N=self.nonterminals,
            T=self.terminals,
            P=self.rules,
            S=self.start_symbol,
        )

    @property
    def nonterminals(self) -> tuple[NonTerminal, ...]:
        return self._nonterminals

    @property
    def terminals(self) -> tuple[Terminal, ...]:
        return self._terminals

    @property
    def rules(self) -> tuple[ProductionRule, ...]:
        return self._rules

    @property
    def start_symbol(self) -> NonTerminal:
        return self._start_symbol

    @property
    def raw_grammar(self) -> str:
        return self._raw_grammar

    @functools.cache
    def expansions(self, nt: NonTerminal) -> tuple[RuleOption, ...]:
        if nt not in self.nonterminals:
            raise ValueError("nt is not contained in the nonterminals of the grammar")

        relevant_rhss = (r.rhs for r in self.rules if r.lhs == nt)
        rhss_as_tuple = tuple(relevant_rhss)

        assert len(rhss_as_tuple) >= 1

        return rhss_as_tuple

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if id(self) == id(other):
            return True

        if not isinstance(other, Grammar):
            return NotImplemented

        return self._as_tuple == other._as_tuple


def _nonterminals_and_lhss_are_consistent(
    nonterminals: tuple[NonTerminal, ...],
    rules: tuple[ProductionRule, ...],
) -> None:
    nts_set = set(nonterminals)
    lhss_set = set(r.lhs for r in rules)

    only_on_nts = nts_set.difference(lhss_set)
    if only_on_nts:
        raise ValueError(
            "all nonterminals must appear on the lhs of a rule at least once, "
            f"but the following do not: {only_on_nts}"
        )

    only_on_lhss = lhss_set.difference(only_on_nts)
    if only_on_lhss:
        raise ValueError(
            "all symbols that appear on the lhs of a rule must be registered "
            f"as a nonterminal, but the following are not: {only_on_lhss}"
        )


def _nonterminals_and_rhss_are_consistent(
    nonterminals: tuple[NonTerminal, ...],
    rules: tuple[ProductionRule, ...],
    start: NonTerminal,
) -> None:
    all_rhs_options = (r.rhs.symbols for r in rules)
    all_rhs_symbols = itertools.chain(*all_rhs_options)
    all_rhs_nonterminals = set(
        nt for nt in all_rhs_symbols if isinstance(nt, NonTerminal)
    )

    # start is always used therefore should not be checked
    nts_set = set(nonterminals)
    nts_set.remove(start)

    only_on_nts = nts_set.difference(all_rhs_nonterminals)
    if only_on_nts:
        raise ValueError(
            "all nonterminals must be used "
            "(i.e. appear on the rhs of a rule) "
            f"at least once, but the following do not: {only_on_nts}"
        )

    only_on_rhss = all_rhs_nonterminals.difference(nts_set)
    if only_on_rhss:
        raise ValueError(
            "all nonterminals that appear on the rhs of a rule must be "
            "registered as a nonterminal, "
            f"but the following are not: {only_on_rhss}"
        )


def _terminals_and_rhss_are_consistent(
    terminals: tuple[Terminal, ...],
    rules: tuple[ProductionRule, ...],
) -> None:
    all_rhs_options = (r.rhs.symbols for r in rules)
    all_rhs_symbols = itertools.chain(*all_rhs_options)
    all_rhs_terminals = set(nt for nt in all_rhs_symbols if isinstance(nt, Terminal))

    terms_set = set(terminals)

    only_on_terms = terms_set.difference(all_rhs_terminals)
    if only_on_terms:
        raise ValueError(
            "all terminals must appear on the rhs of a rule "
            f"at least once, but the following do not: {only_on_terms}"
        )

    only_on_rhss = all_rhs_terminals.difference(terms_set)
    if only_on_rhss:
        raise ValueError(
            "all terminals that appear on the rhs of a rule "
            "must be registered as a terminal, "
            f"but the following are not: {only_on_rhss}"
        )


@typeguard.typechecked
def validate_grammar_components(
    N: tuple[NonTerminal, ...],
    T: tuple[Terminal, ...],
    P: tuple[ProductionRule, ...],
    S: NonTerminal,
) -> None:
    assert len(N) > 0
    assert len(T) > 0
    assert len(P) > 0

    assert len(N) == len(set(N))
    assert len(T) == len(set(T))
    assert len(P) == len(set(P))

    assert S in N

    _nonterminals_and_lhss_are_consistent(N, P)
    _nonterminals_and_rhss_are_consistent(N, P, S)
    _terminals_and_rhss_are_consistent(T, P)


class GrammarTransformer(lark.Transformer[list[ProductionRule]]):
    def __init__(self) -> None:
        super().__init__()

        self._frozen = True

        self._terminals: Optional[list[Terminal]] = None
        self._nonterminals: Optional[list[NonTerminal]] = None
        self._rules: Optional[list[ProductionRule]] = None
        self._start: Optional[NonTerminal] = None

    def __default__(self, data: Any, children: Any, meta: Any) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: Any) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    def get_nonterminals(self) -> tuple[NonTerminal, ...]:
        assert self._frozen
        assert self._nonterminals is not None

        return tuple(self._nonterminals)

    def get_terminals(self) -> tuple[Terminal, ...]:
        assert self._frozen
        assert self._terminals is not None

        return tuple(self._terminals)

    def get_rules(self) -> tuple[ProductionRule, ...]:
        assert self._frozen
        assert self._rules is not None

        return tuple(self._rules)

    def get_start_symbol(self) -> NonTerminal:
        assert self._frozen
        assert self._start is not None

        return self._start

    def transform(self, tree: lark.Tree) -> list[ProductionRule]:
        assert self._frozen

        self._frozen = False

        self._terminals = []
        self._nonterminals = []
        self._rules = []

        ret = super().transform(tree)
        self._frozen = True

        return ret

    def _register_terminal(self, text: str) -> Terminal:
        assert not self._frozen
        assert self._terminals is not None

        term = Terminal(text)

        if term not in self._terminals:
            self._terminals.append(term)

        return term

    def _register_nonterminal(self, text: str) -> NonTerminal:
        assert not self._frozen
        assert self._nonterminals is not None

        nonterm = NonTerminal(text)

        if nonterm not in self._nonterminals:
            self._nonterminals.append(nonterm)

        return nonterm

    def start(self, list_of_list_of_rules: Any) -> list[ProductionRule]:
        assert not self._frozen
        assert self._start is None

        self._start = NonTerminal("start")

        flattened = [
            rule for list_of_rules in list_of_list_of_rules for rule in list_of_rules
        ]
        self._rules = flattened

        return flattened

    def rule(self, rule_parts: Any) -> list[ProductionRule]:
        assert not self._frozen

        lhs, *list_of_list_of_options = rule_parts

        return [
            ProductionRule(lhs, rhs)
            for list_of_options in list_of_list_of_options
            for rhs in list_of_options
        ]

    def block(self, list_of_list_of_options: Any) -> list[RuleOption]:
        assert not self._frozen

        return [
            option
            for list_of_options in list_of_list_of_options
            for option in list_of_options
        ]

    def block_option(
        self,
        repeated_symbols: list[Iterable[Symbol]],
    ) -> list[RuleOption]:
        assert not self._frozen

        combinations = itertools.product(*repeated_symbols)
        unpacked = [itertools.chain(*inner) for inner in combinations]
        return [RuleOption(tuple(symbols)) for symbols in unpacked]

    def symbol_range(
        self, parts: tuple[NonTerminal, Optional[int], Optional[int]]
    ) -> list[list[NonTerminal]]:
        assert not self._frozen

        name, a, b = parts

        if a is not None and b is not None:
            start = a
            stop = b

        elif a is not None and b is None:
            start = a
            stop = a

        elif a is None and b is None:
            start = 1
            stop = 1

        else:
            raise ValueError(f"unexpected symbol_range configuration: {parts}")

        assert start >= 0
        assert stop >= start

        return [[name] * count for count in range(start, stop + 1)]

    def layer(self, list_of_lists_of_options: Any) -> list[RuleOption]:
        assert not self._frozen

        return [
            opt
            for list_of_options in list_of_lists_of_options
            for opt in list_of_options
        ]

    def conv_layer(self, parts: Any) -> list[RuleOption]:
        assert not self._frozen

        marker, filter_counts, kernel_sizes, strides = parts
        option_data = itertools.product(filter_counts, kernel_sizes, strides)
        return [RuleOption((marker, *fc, *ks, *st)) for fc, ks, st in option_data]

    def filter_count(self, parts: list[Terminal]) -> list[tuple[Terminal, Terminal]]:
        assert not self._frozen

        marker, *counts = parts
        return [(marker, c) for c in counts]

    def kernel_size(self, parts: list[Terminal]) -> list[tuple[Terminal, Terminal]]:
        assert not self._frozen

        marker, *sizes = parts
        return [(marker, s) for s in sizes]

    def stride(self, parts: list[Terminal]) -> list[tuple[Terminal, Terminal]]:
        assert not self._frozen

        marker, *strides = parts
        return [(marker, s) for s in strides]

    def dense_layer(self, parts: list[Terminal]) -> list[RuleOption]:
        assert not self._frozen

        marker, *units = parts
        return [RuleOption((marker, ut)) for ut in units]

    def dropout_layer(self, parts: list[Terminal]) -> list[RuleOption]:
        assert not self._frozen

        marker, *rates = parts
        return [RuleOption((marker, rt)) for rt in rates]

    def NONTERMINAL(self, token: lark.Token) -> NonTerminal:
        assert not self._frozen
        return self._register_nonterminal(token.value)

    def MERGE(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def FORK(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def CONV2D(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def FILTER_COUNT(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def KERNEL_SIZE(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def STRIDE(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def DENSE(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def DROPOUT(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def RANGE_BOUND(self, token: lark.Token) -> int:
        assert not self._frozen
        return int(token.value)

    def INT_ARG(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)

    def FLOAT_ARG(self, token: lark.Token) -> Terminal:
        assert not self._frozen
        return self._register_terminal(token.value)
