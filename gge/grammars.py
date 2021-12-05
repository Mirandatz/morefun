import functools
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import lark

_LarkTreeNode = Union[lark.Tree, str]


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

        if len(self.text) == 0:
            raise ValueError(error_msg)

        if not self.text.islower():
            raise ValueError(error_msg)

        first = self.text[0]
        if not (first == "_" or first.isalpha()):
            raise ValueError(error_msg)

        core_is_valid = all(c.isalnum() or c == "_" for c in self.text[1:])
        if not core_is_valid:
            raise ValueError(error_msg)


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


Symbol = Union[Terminal, NonTerminal]


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


@dataclass(order=True, frozen=True)
class ProductionRule:
    lhs: NonTerminal
    rhs: RuleOption


class Grammar:
    def __init__(
        self,
        raw_grammar: str,
        raw_metagrammar: str,
    ) -> None:
        n, t, p, s = extract_grammar_components(
            raw_grammar,
            raw_metagrammar,
        )
        validate_grammar_components(n, t, p, s)

        self._raw_grammar = raw_grammar

        self._nonterminals = n
        self._terminals = t
        self._rules = p
        self._start_symbol = s

        self._as_tuple = (n, t, p, s)
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


def _extract_lhs(rule: lark.Tree) -> NonTerminal:
    assert rule.data == "rule"
    lhs = rule.children[0]
    assert isinstance(lhs, lark.Token)
    return NonTerminal(lhs.value)


def _conv_layer_marker(node: _LarkTreeNode) -> Terminal:
    assert isinstance(node, lark.Token)
    unquoted = node.value.replace('"', "")
    return Terminal(unquoted)


def _filter_counts(node: _LarkTreeNode) -> tuple[Terminal, list[Terminal]]:
    assert isinstance(node, lark.Tree)
    assert node.data == "filter_count"

    marker_token, *count_nodes = node.children

    assert isinstance(marker_token, lark.Token)
    unquoted_marker = marker_token.value.replace('"', "")
    marker = Terminal(unquoted_marker)

    counts: list[Terminal] = []
    for cn in count_nodes:
        assert isinstance(cn, lark.Token)
        cn_terminal = Terminal(cn.value)
        counts.append(cn_terminal)

    return marker, counts


def _kernel_sizes(
    node: _LarkTreeNode,
) -> tuple[Terminal, list[Terminal]]:
    assert isinstance(node, lark.Tree)
    assert node.data == "kernel_size"

    marker_node, *size_nodes = node.children

    assert isinstance(marker_node, lark.Token)
    unquoted = marker_node.value.replace('"', "")
    marker = Terminal(unquoted)

    sizes: list[Terminal] = []
    for sn in size_nodes:
        assert isinstance(sn, lark.Token)
        sn_term = Terminal(sn.value)
        sizes.append(sn_term)

    return marker, sizes


def _strides(node: _LarkTreeNode) -> tuple[Terminal, list[Terminal]]:
    assert isinstance(node, lark.Tree)
    assert node.data == "stride"

    marker_node, *stride_nodes = node.children

    assert isinstance(marker_node, lark.Token)
    unquoted = marker_node.value.replace('"', "")
    marker = Terminal(unquoted)

    strides: list[Terminal] = []
    for sn in stride_nodes:
        assert isinstance(sn, lark.Token)
        sn_term = Terminal(sn.value)
        strides.append(sn_term)

    return marker, strides


def _rhs_from_conv_layer(node: lark.Tree) -> Iterable[RuleOption]:
    assert node.data == "conv_layer"

    (
        conv_layer_marker_node,
        filter_count_node,
        kernel_size_node,
        stride_node,
    ) = node.children

    conv_marker = _conv_layer_marker(conv_layer_marker_node)
    filter_count_marker, filter_counts = _filter_counts(filter_count_node)
    kernel_size_marker, kernel_sizes = _kernel_sizes(kernel_size_node)
    stride_marker, strides = _strides(stride_node)

    for fc, ks, st in itertools.product(filter_counts, kernel_sizes, strides):
        symbols = (
            conv_marker,
            filter_count_marker,
            fc,
            kernel_size_marker,
            ks,
            stride_marker,
            st,
        )
        yield RuleOption(symbols)


def _rhs_from_dense_layer(node: lark.Tree) -> Iterable[RuleOption]:
    assert node.data == "dense_layer"

    marker_node, *units_tokens = node.children

    assert isinstance(marker_node, lark.Token)
    marker_terminal = Terminal(marker_node.value.replace('"', ""))

    for ut in units_tokens:
        assert isinstance(ut, lark.Token)
        units_terminal = Terminal(ut.value)

        symbols = (marker_terminal, units_terminal)
        yield RuleOption(symbols)


def _rhs_from_dropout_layer(node: lark.Tree) -> Iterable[RuleOption]:
    assert node.data == "dropout_layer"

    marker_token, *rate_tokens = node.children

    assert isinstance(marker_token, lark.Token)
    marker_terminal = Terminal(marker_token.value.replace('"', ""))

    for rt in rate_tokens:
        assert isinstance(rt, lark.Token)
        rate_terminal = Terminal(rt.value)

        symbols = (marker_terminal, rate_terminal)
        yield RuleOption(symbols)


def _rhss_from_layer(node: lark.Tree) -> Iterable[RuleOption]:
    assert node.data == "layer"

    assert len(node.children) == 1
    child = node.children[0]
    assert isinstance(child, lark.Tree)

    if child.data == "conv_layer":
        yield from _rhs_from_conv_layer(child)

    elif child.data == "dense_layer":
        yield from _rhs_from_dense_layer(child)

    elif child.data == "dropout_layer":
        yield from _rhs_from_dropout_layer(child)

    else:
        raise ValueError(f"unexpected node type: {child.data}")


def _symbol_range(node: _LarkTreeNode) -> Iterable[list[NonTerminal]]:
    assert isinstance(node, lark.Tree)
    assert node.data == "symbol_range"

    name_token, a, b = node.children

    assert isinstance(name_token, lark.Token)
    name = name_token.value
    nt = NonTerminal(name)

    assert a is None or isinstance(a, lark.Token)
    assert b is None or isinstance(b, lark.Token)

    if a is not None and b is not None:
        start = int(a.value)
        stop = int(b.value)
    elif a is not None and b is None:
        start = int(a.value)
        stop = int(a.value)
    elif a is None and b is None:
        start = 1
        stop = 1
    else:
        raise ValueError(f"unexpected symbol_range configuration: {node}")

    assert start >= 0
    assert stop >= start

    for count in range(start, stop + 1):
        yield [nt] * count


def _rhss_from_block_option(node: _LarkTreeNode) -> Iterable[RuleOption]:
    assert isinstance(node, lark.Tree)
    assert node.data == "block_option"

    repeated_symbols = [_symbol_range(sr) for sr in node.children]

    for current_expansions in itertools.product(*repeated_symbols):
        symbols = []
        for ce in current_expansions:
            symbols.extend(ce)

        yield RuleOption(symbols=tuple(symbols))


def _rhss_from_block(node: lark.Tree) -> Iterable[RuleOption]:
    assert node.data == "block"

    for option in node.children:
        yield from _rhss_from_block_option(option)


def _rhss_from_rule(rule: lark.Tree) -> Iterable[RuleOption]:
    assert rule.data == "rule"

    _, *options = rule.children

    for opt in options:
        assert isinstance(opt, lark.Tree)

        if opt.data == "layer":
            yield from _rhss_from_layer(opt)

        elif opt.data == "block":
            yield from _rhss_from_block(opt)

        else:
            raise ValueError(f"unknown node type {opt.data}")


def extract_rules(tree: lark.Tree) -> Iterable[ProductionRule]:
    assert tree.data == "start"

    for rule in tree.children:
        assert isinstance(rule, lark.Tree)

        lhs = _extract_lhs(rule)

        for rhs in _rhss_from_rule(rule):
            yield ProductionRule(lhs, rhs)


def extract_nonterminals(tree: lark.Tree) -> set[NonTerminal]:
    assert tree.data == "start"

    nonterminals: set[NonTerminal] = set()

    for rule in tree.children:
        assert isinstance(rule, lark.Tree)
        assert rule.data == "rule"

        lhs = rule.children[0]
        assert isinstance(lhs, lark.Token)

        nt = NonTerminal(text=lhs.value)
        nonterminals.add(nt)

    return nonterminals


_TERMINAL_TOKEN_TYPES = {
    "CONV2D",
    "FILTER_COUNT",
    "KERNEL_SIZE",
    "STRIDE",
    "DENSE",
    "DROPOUT",
    "INT",
    "FLOAT",
}


def _is_terminal_token(node: _LarkTreeNode) -> bool:
    if not isinstance(node, lark.Token):
        return False

    return node.type in _TERMINAL_TOKEN_TYPES


def extract_terminals(
    tree: lark.Tree,
    nonterminals: set[NonTerminal],
) -> set[Terminal]:
    assert tree.data == "start"

    terminals: set[Terminal] = set()
    for node in tree.scan_values(pred=_is_terminal_token):
        token_text = node.replace('"', "")
        terminals.add(Terminal(token_text))

    # sanity check
    nonterminals_texts = set(nt.text for nt in nonterminals)
    for term in terminals:
        if term.text in nonterminals_texts:
            raise ValueError(f"text is both a terminal and a nonterminal: {term.text}")

    return terminals


def extract_grammar_components(
    raw_grammar: str,
    raw_metagrammar: str,
) -> tuple[
    tuple[NonTerminal, ...],
    tuple[Terminal, ...],
    tuple[ProductionRule, ...],
    NonTerminal,
]:
    """
    Returns the standard components of a grammar = (N, T, P, S)
    N = Nonterminals
    T = Terminals
    P = Production rules
    S = Start symbol
    """
    assert raw_grammar
    assert raw_metagrammar

    parser = lark.Lark(raw_metagrammar)
    tree = parser.parse(raw_grammar)

    rules = extract_rules(tree)
    nonterminals = extract_nonterminals(tree)
    terminals = extract_terminals(tree, nonterminals)
    start = NonTerminal("start")

    return tuple(nonterminals), tuple(terminals), tuple(rules), start


def _all_nonterminals_are_defined(
    nonterminals: tuple[NonTerminal, ...],
    rules: tuple[ProductionRule, ...],
) -> bool:
    nts_set = set(nonterminals)
    lhss_set = set(r.lhs for r in rules)
    return nts_set.issubset(lhss_set)


def _all_lhs_are_on_nonterminals(
    nonterminals: tuple[NonTerminal, ...],
    rules: tuple[ProductionRule, ...],
) -> bool:
    nts_set = set(nonterminals)
    lhss_set = set(r.lhs for r in rules)
    return lhss_set.issubset(nts_set)


def _all_nonterminals_are_used(
    nonterminals: tuple[NonTerminal, ...],
    rules: tuple[ProductionRule, ...],
    start: NonTerminal,
) -> bool:
    all_rhs_options = (r.rhs.symbols for r in rules)
    all_rhs_symbols = itertools.chain(*all_rhs_options)
    all_rhs_nonterminals = set(
        nt for nt in all_rhs_symbols if isinstance(nt, NonTerminal)
    )

    # start is always used therefore should not be checked
    nts_set = set(nonterminals)
    nts_set.remove(start)

    return nts_set.issubset(all_rhs_nonterminals)


def _assert_all_terminals_are_used(
    terminals: tuple[Terminal, ...],
    rules: tuple[ProductionRule, ...],
) -> None:
    all_rhs_options = (r.rhs.symbols for r in rules)
    all_rhs_symbols = itertools.chain(*all_rhs_options)
    used_terminals = set(nt for nt in all_rhs_symbols if isinstance(nt, Terminal))

    for t in terminals:
        assert t in used_terminals, t


def _all_terminals_were_found(
    terminals: tuple[Terminal, ...],
    rules: tuple[ProductionRule, ...],
) -> bool:
    all_rhs_options = (r.rhs.symbols for r in rules)
    all_rhs_symbols = itertools.chain(*all_rhs_options)
    all_rhs_terminals = set(nt for nt in all_rhs_symbols if isinstance(nt, Terminal))

    terms_set = set(terminals)

    return all_rhs_terminals.issubset(terms_set)


def _all_symbols_on_rhs_are_terminal_or_nonterminal(
    rules: tuple[ProductionRule, ...]
) -> bool:
    all_rhs_options = (r.rhs.symbols for r in rules)
    all_rhs_symbols = itertools.chain(*all_rhs_options)
    return all(isinstance(s, (Terminal, NonTerminal)) for s in all_rhs_symbols)


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

    assert all(isinstance(nt, NonTerminal) for nt in N)
    assert all(isinstance(t, Terminal) for t in T)
    assert S in N

    assert _all_nonterminals_are_defined(N, P)
    assert _all_nonterminals_are_used(N, P, S)
    assert _all_lhs_are_on_nonterminals(N, P)

    _assert_all_terminals_are_used(T, P)
    assert _all_terminals_were_found(T, P)

    assert _all_symbols_on_rhs_are_terminal_or_nonterminal(P)


class GrammarTransformer(lark.Transformer[list[ProductionRule]]):
    def __init__(self) -> None:
        super().__init__()

        self._frozen = True

        self._terminals: Optional[list[Terminal]] = None
        self._nonterminals: Optional[list[NonTerminal]] = None
        self._rules: Optional[list[ProductionRule]] = None
        self._start: Optional[NonTerminal] = None

    def __default__(
        self,
        data: str,
        children: list[Any],
        meta: lark.tree.Meta,
    ) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(
        self,
        token_text: str,
    ) -> None:
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

    def start(self, rules: list[ProductionRule]) -> list[ProductionRule]:
        assert not self._frozen
        assert self._start is None

        self._start = NonTerminal("start")
        self._rules = rules

        return rules

    def rule(
        self,
        rule_parts: tuple[NonTerminal, list[RuleOption]],
    ) -> list[ProductionRule]:
        assert not self._frozen

        lhs, options = rule_parts
        return [ProductionRule(lhs=lhs, rhs=opt) for opt in options]

    def block(self, block_options: list[RuleOption]) -> list[RuleOption]:
        return block_options

    def block_option(
        self,
        repeated_symbols: list[Iterable[Symbol]],
    ) -> list[RuleOption]:
        return [
            RuleOption(tuple(symbols))
            for symbols in itertools.product(*repeated_symbols)
        ]

    def symbol_range(
        self, parts: tuple[NonTerminal, Optional[int], Optional[int]]
    ) -> list[list[NonTerminal]]:
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

    def layer(self, options: list[RuleOption]) -> list[RuleOption]:
        return options

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


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    metagrammar = (data_dir / "metagrammar.lark").read_text()
    # grammar = (data_dir / "simple_grammar.lark").read_text()
    grammar = r"""
        start: simple_block complex_block simple_dense

        simple_block : simple_conv simple_dense
        simple_conv  : "conv2d" "filter_count" (4) "kernel_size" (3) "stride" (1)
        simple_dense : "dense" (16)

        complex_block : complex_conv~2 complex_dense_block~1..3
                    | simple_conv complex_dense_block~2..2 simple_conv
                    | simple_conv

        complex_conv  : "conv2d" "filter_count" (4 | 8 | 6) "kernel_size" (3 | 5) "stride" (1 | 2)

        complex_dense_block : complex_dense_layer~3..5 _droperino
        complex_dense_layer : "dense" (16 | 32 | 64)
        _droperino          : "dropout" (0.3 | 0.5 | 0.7)
    """

    parser = lark.Lark(metagrammar)
    tree = parser.parse(grammar)

    extractor = GrammarTransformer()
    rules = extractor.transform(tree)

    for r in rules:
        print(r)

    for t in extractor.get_terminals():
        print(t)


if __name__ == "__main__":
    main()
