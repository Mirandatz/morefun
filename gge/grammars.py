import dataclasses
import functools
import itertools
import pathlib
import typing

import lark

import gge.transformers as gge_transformers

METAGRAMMAR_PATH = pathlib.Path(__file__).parent.parent / "data" / "metagrammar.lark"


@functools.cache
def get_metagrammar() -> str:
    return METAGRAMMAR_PATH.read_text()


def get_metagrammar_parser() -> lark.Lark:
    return lark.Lark(get_metagrammar(), parser="lalr", maybe_placeholders=True)


def extract_ast(grammar_text: str) -> lark.Tree:
    parser = get_metagrammar_parser()
    return parser.parse(grammar_text)


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


@dataclasses.dataclass(order=True, frozen=True)
class NonTerminal:
    text: str

    def __post_init__(self) -> None:
        assert isinstance(self.text, str)

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


@dataclasses.dataclass(order=True, frozen=True)
class Terminal:
    text: str

    def __post_init__(self) -> None:
        assert isinstance(self.text, str)

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

        unquoted = self.text.strip('"')
        if not _is_valid_name(unquoted):
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        return f"T({self.text})"


Symbol: typing.TypeAlias = Terminal | NonTerminal


@dataclasses.dataclass(order=True, frozen=True)
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
        assert isinstance(self.symbols, tuple)
        for s in self.symbols:
            assert isinstance(s, Symbol)

        assert len(self.symbols) >= 1

    def __repr__(self) -> str:
        all_options = ",".join(repr(s) for s in self.symbols)
        return f"RuleOption({all_options})"


@dataclasses.dataclass(order=True, frozen=True)
class ProductionRule:
    lhs: NonTerminal
    rhs: RuleOption

    def __post_init__(self) -> None:
        assert isinstance(self.lhs, NonTerminal)
        assert isinstance(self.rhs, RuleOption)

    def __repr__(self) -> str:
        return f"Rule({self.lhs}->{self.rhs})"


@dataclasses.dataclass(frozen=True)
class GrammarComponents:
    nonterminals: tuple[NonTerminal, ...]
    terminals: tuple[Terminal, ...]
    rules: tuple[ProductionRule, ...]
    start_symbol: NonTerminal

    def __post_init__(self) -> None:
        assert isinstance(self.nonterminals, tuple)
        for nt in self.nonterminals:
            assert isinstance(nt, NonTerminal)

        assert isinstance(self.terminals, tuple)
        for term in self.terminals:
            assert isinstance(term, Terminal)

        assert isinstance(self.rules, tuple)
        for r in self.rules:
            assert isinstance(r, ProductionRule)

        assert isinstance(self.start_symbol, NonTerminal)

        N = self.nonterminals
        T = self.terminals
        P = self.rules
        S = self.start_symbol

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


def _nonterminals_and_lhss_are_consistent(
    nonterminals: tuple[NonTerminal, ...],
    rules: tuple[ProductionRule, ...],
) -> None:
    nts_set = set(nonterminals)
    lhss_set = set(r.lhs for r in rules)

    only_on_nts = set.difference(nts_set, lhss_set)
    if only_on_nts:
        raise ValueError(
            "all nonterminals must appear on the lhs of a rule at least once, "
            f"but the following do not: {only_on_nts}"
        )

    only_on_lhss = set.difference(lhss_set, nts_set)
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

    only_on_nts = set.difference(nts_set, all_rhs_nonterminals)
    if only_on_nts:
        raise ValueError(
            "all nonterminals must be used "
            "(i.e. appear on the rhs of a rule) "
            f"at least once, but the following do not: {only_on_nts}"
        )

    only_on_rhss = set.difference(all_rhs_nonterminals, nts_set)
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

    only_on_terms = set.difference(terms_set, all_rhs_terminals)
    if only_on_terms:
        raise ValueError(
            "all terminals must appear on the rhs of a rule "
            f"at least once, but the following do not: {only_on_terms}"
        )

    only_on_rhss = set.difference(all_rhs_terminals, terms_set)
    if only_on_rhss:
        raise ValueError(
            "all terminals that appear on the rhs of a rule "
            "must be registered as a terminal, "
            f"but the following are not: {only_on_rhss}"
        )


class Grammar:
    def __init__(self, raw_grammar: str) -> None:
        tree = extract_ast(raw_grammar)
        components = GrammarTransformer().transform(tree)

        self._raw_grammar = raw_grammar

        self._nonterminals = components.nonterminals
        self._terminals = components.terminals
        self._rules = components.rules
        self._start_symbol = components.start_symbol

        self._as_tuple = (
            self._nonterminals,
            self._terminals,
            self._rules,
            self._start_symbol,
        )

        self._hash = hash(self._as_tuple)

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


MarkerValuePair = tuple[Terminal, Terminal]


class GrammarTransformer(gge_transformers.SinglePassTransformer[GrammarComponents]):
    def __init__(self) -> None:
        super().__init__()

        self._terminals: list[Terminal] = []
        self._nonterminals: list[NonTerminal] = []
        self._rules: list[ProductionRule] = []
        self._start: NonTerminal = NonTerminal("start")

    def transform(self, tree: lark.Tree) -> GrammarComponents:
        super().transform(tree)

        return GrammarComponents(
            nonterminals=tuple(self._nonterminals),
            terminals=tuple(self._terminals),
            rules=tuple(self._rules),
            start_symbol=self._start,
        )

    def _register_terminal(self, text: str) -> Terminal:
        self._raise_if_not_running()
        assert type(text) == str

        term = Terminal(text)

        if term not in self._terminals:
            self._terminals.append(term)

        return term

    def _register_nonterminal(self, text: str) -> NonTerminal:
        self._raise_if_not_running()
        assert type(text) == str

        nonterm = NonTerminal(text)

        if nonterm not in self._nonterminals:
            self._nonterminals.append(nonterm)

        return nonterm

    def start(self, list_of_list_of_rules: typing.Any) -> list[ProductionRule]:
        self._raise_if_not_running()

        flattened = [
            rule for list_of_rules in list_of_list_of_rules for rule in list_of_rules
        ]
        self._rules.extend(flattened)

        return flattened

    def rule(self, rule_parts: typing.Any) -> list[ProductionRule]:
        self._raise_if_not_running()

        lhs, *list_of_list_of_options = rule_parts

        return [
            ProductionRule(lhs, rhs)
            for list_of_options in list_of_list_of_options
            for rhs in list_of_options
        ]

    def block(self, list_of_list_of_options: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()

        return [
            option
            for list_of_options in list_of_list_of_options
            for option in list_of_options
        ]

    def block_option(
        self,
        repeated_symbols: list[list[Terminal | NonTerminal]],
    ) -> list[RuleOption]:
        self._raise_if_not_running()

        combinations = itertools.product(*repeated_symbols)
        unpacked = [itertools.chain(*inner) for inner in combinations]
        return [RuleOption(tuple(symbols)) for symbols in unpacked]

    @lark.v_args(inline=True)
    def maybe_merge(self, term: typing.Optional[Terminal]) -> list[list[Terminal]]:
        self._raise_if_not_running()

        if term is None:
            return [[]]
        else:
            return [[term]]

    @lark.v_args(inline=True)
    def maybe_fork(self, term: typing.Optional[Terminal]) -> list[list[Terminal]]:
        self._raise_if_not_running()

        if term is None:
            return [[]]
        else:
            return [[term]]

    @lark.v_args(inline=True)
    def symbol_range(
        self,
        nt: NonTerminal,
        a: typing.Optional[int] = None,
        b: typing.Optional[int] = None,
    ) -> list[list[NonTerminal]]:
        self._raise_if_not_running()

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
            raise ValueError(f"unexpected symbol_range configuration: {(nt,a,b)}")

        assert start >= 0
        assert stop >= start

        return [[nt] * count for count in range(start, stop + 1)]

    def layer(self, list_of_lists_of_options: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()

        return [
            opt
            for list_of_options in list_of_lists_of_options
            for opt in list_of_options
        ]

    def conv_layer(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()

        marker, *params = parts
        combinations = itertools.product(*params)
        return [RuleOption((marker, *fc, *ks, *st)) for fc, ks, st in combinations]

    def filter_count(self, parts: list[Terminal]) -> list[tuple[Terminal, Terminal]]:
        self._raise_if_not_running()

        marker, *counts = parts
        return [(marker, c) for c in counts]

    def kernel_size(self, parts: list[Terminal]) -> list[tuple[Terminal, Terminal]]:
        self._raise_if_not_running()

        marker, *sizes = parts
        return [(marker, s) for s in sizes]

    def stride(self, parts: list[Terminal]) -> list[tuple[Terminal, Terminal]]:
        self._raise_if_not_running()

        marker, *strides = parts
        return [(marker, s) for s in strides]

    def activation(self, activations: list[Terminal]) -> list[Terminal]:
        self._raise_if_not_running()
        return activations

    @lark.v_args(inline=True)
    def batchnorm_layer(self, term: Terminal) -> list[RuleOption]:
        return [RuleOption((term,))]

    @lark.v_args(inline=True)
    def pooling_layer(
        self,
        marker: Terminal,
        types: list[MarkerValuePair],
        strides: list[MarkerValuePair],
    ) -> list[RuleOption]:
        combinations = itertools.product(types, strides)
        return [RuleOption((marker, *tp, *st)) for tp, st in combinations]

    @lark.v_args(inline=True)
    def pooling_type_seq(
        self,
        marker: Terminal,
        *values: Terminal,
    ) -> list[tuple[Terminal, Terminal]]:
        return [(marker, v) for v in values]

    @lark.v_args(inline=True)
    def pooling_stride_seq(
        self,
        marker: Terminal,
        *values: Terminal,
    ) -> list[tuple[Terminal, Terminal]]:
        return [(marker, v) for v in values]

    def BATCHNORM(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def NONTERMINAL(self, token: lark.Token) -> NonTerminal:
        self._raise_if_not_running()
        return self._register_nonterminal(token.value)

    def MERGE(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def FORK(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def CONV2D(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def FILTER_COUNT(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def KERNEL_SIZE(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def STRIDE(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def ACTIVATION(self, term: Terminal) -> Terminal:
        return term

    def RELU(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def GELU(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def SWISH(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def RANGE_BOUND(self, token: lark.Token) -> int:
        self._raise_if_not_running()
        return int(token.value)

    def INT_ARG(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def FLOAT_ARG(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def POOLING_LAYER(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def POOLING_TYPE(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def POOLING_STRIDE(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def POOLING_MAX(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def POOLING_AVG(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)
