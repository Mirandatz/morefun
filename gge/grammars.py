import functools
import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Union

RULE_SIDE_DELIMITER = ":"
RULE_OPTION_DELIMITER = "|"


def _can_be_parsed_as_float(value: str) -> bool:
    try:
        _ = float(value)
        return True
    except ValueError:
        return False


def _can_be_stored_as_text(value: str) -> bool:
    chars_are_valid = (c.isalnum() or c == "_" for c in value)
    return len(value) > 0 and all(chars_are_valid)


@dataclass(order=True, frozen=True)
class NonTerminal:
    text: str

    def __post_init__(self) -> None:
        assert _can_be_stored_as_text(self.text)


@dataclass(order=True, frozen=True)
class Terminal:
    text: str

    def __post_init__(self) -> None:
        assert _can_be_parsed_as_float(self.text) or _can_be_stored_as_text(self.text)


Symbol = Union[Terminal, NonTerminal]


@dataclass(order=True, frozen=True)
class QuantifiedSymbol:
    """
    Represents a combination of a `symbol` with a "repetition quantifier", such as
    `symbol~2..5` -> min = 2, max = 5
    `symbol~1000` -> min = 1000, max = 1000
    `symbol?`     -> min = 0, max = 1
    `symbol`      -> min = 1, max = 1
    """

    symbol: Symbol
    min_count_inclusive: int
    max_count_inclusive: int

    def __post_init__(self) -> None:
        assert self.min_count_inclusive >= 0
        assert self.max_count_inclusive >= 0
        assert self.max_count_inclusive >= self.min_count_inclusive


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


@dataclass(order=True, frozen=True)
class ExpandableRuleOption:
    """
    Almost the same thing as `RuleOption`, but contains `QuantifiedSymbols`, instead of raw symbols, e.g.:
    thousand_bs_or_maybe_c = b~1000 | c?
    expandable_option0 = b~1000
    expandable_option1 = c?
    """

    quantified_symbols: tuple[QuantifiedSymbol, ...]


@dataclass(order=True, frozen=True)
class ProductionRule:
    lhs: NonTerminal
    rhs: tuple[Symbol, ...]


class Grammar:
    def __init__(self, raw_grammar: str) -> None:
        n, t, p, s = extract_grammar_components(raw_grammar)
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
    def expansions(self, nt: NonTerminal) -> tuple[tuple[Symbol, ...], ...]:
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


def expand_quantified_symbol(unit: QuantifiedSymbol) -> Iterable[tuple[Symbol, ...]]:
    for i in range(
        unit.min_count_inclusive,
        unit.max_count_inclusive + 1,
    ):
        expansion = tuple(itertools.repeat(unit.symbol, i))
        yield expansion


def expand_rule_option(option: ExpandableRuleOption) -> Iterable[RuleOption]:
    all_symbols_expansions = [
        expand_quantified_symbol(u) for u in option.quantified_symbols
    ]
    for current_expansions in itertools.product(*all_symbols_expansions):
        symbols = itertools.chain(*current_expansions)
        yield RuleOption(symbols=tuple(symbols))


def extract_rule_lhs(trimmed_line: str) -> tuple[NonTerminal, str]:
    """
    Returns the symbol on the left-hand side of a rule and the raw right-hand side of the rule
    """
    match = re.search(pattern=r"^\s*(\w+)\s*:", string=trimmed_line)
    if not match:
        raise ValueError(f"Unable to find lhs of rule: {trimmed_line}")

    non_terminal_text = match.group(1)
    non_terminal = NonTerminal(text=non_terminal_text)

    match_end = match.end(0)
    rest = trimmed_line[match_end:]

    return non_terminal, rest


def try_extract_nonterminal(trimmed_rhs: str) -> Optional[tuple[NonTerminal, str]]:
    match = re.match(pattern=r"^\s*(\w+)", string=trimmed_rhs)
    if not match:
        return None

    non_terminal_text = match.group(1)
    non_terminal = NonTerminal(non_terminal_text)

    match_end = match.end(0)
    rest = trimmed_rhs[match_end:]

    return non_terminal, rest


def try_extract_terminal(trimmed_rhs: str) -> Optional[tuple[Terminal, str]]:
    text_pattern = r"\w+"
    int_pattern = r"[-+]?\d+"
    float_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    spaces_pattern = r"^\s*"
    terminal_pattern = (
        f'{spaces_pattern}"({text_pattern}|{float_pattern}|{int_pattern})"'
    )
    match = re.search(pattern=terminal_pattern, string=trimmed_rhs)
    if not match:
        return None

    terminal_text = match.group(1)
    terminal = Terminal(terminal_text)

    match_end = match.end(0)
    rest = trimmed_rhs[match_end:]

    return terminal, rest


def extract_symbol(text: str) -> tuple[Symbol, str]:
    maybe_terminal = try_extract_terminal(text)
    if maybe_terminal:
        terminal, rest = maybe_terminal
        return terminal, rest

    maybe_nonterminal = try_extract_nonterminal(text)
    if maybe_nonterminal:
        nonterminal, rest = maybe_nonterminal
        return nonterminal, rest

    raise ValueError(f"unable to extract a symbol from text: {text}")


def extract_repetition_range(text: str) -> tuple[int, int, str]:
    if len(text) == 0 or text[0].isspace():
        # if there is no repetition range, the symbol must appear exactly once
        inclusive_min = 1
        inclusive_max = 1
        rest = text.lstrip()
        return inclusive_min, inclusive_max, rest

    elif text[0] == "?":
        inclusive_min = 0
        inclusive_max = 1
        rest = text[1:].lstrip()
        return inclusive_min, inclusive_max, rest

    elif match := re.search(pattern=r"^~(\d+)\.\.(\d+)", string=text):
        inclusive_min = int(match.group(1))
        inclusive_max = int(match.group(2))
        rest = text[match.end(0) :]
        return inclusive_min, inclusive_max, rest

    elif match := re.search(pattern=r"^~(\d+)", string=text):
        inclusive_min = int(match.group(1))
        inclusive_max = inclusive_min
        rest = text[match.end(0) :]
        return inclusive_min, inclusive_max, rest

    else:
        raise ValueError(f"Unable to parse repetition range `{text}`")


def extract_quantified_symbols(raw_rule_option: str) -> Iterable[QuantifiedSymbol]:
    if RULE_SIDE_DELIMITER in raw_rule_option:
        raise ValueError(f"Unexpected symbol `{RULE_SIDE_DELIMITER}`")
    if RULE_OPTION_DELIMITER in raw_rule_option:
        raise ValueError(f"unexpected symbol `{RULE_OPTION_DELIMITER}`")

    while raw_rule_option:
        symbol, raw_rule_option = extract_symbol(raw_rule_option)
        inclusive_min, inclusive_max, raw_rule_option = extract_repetition_range(
            raw_rule_option
        )
        qf = QuantifiedSymbol(
            symbol=symbol,
            min_count_inclusive=inclusive_min,
            max_count_inclusive=inclusive_max,
        )
        yield qf


def extract_rule_options(trimmed_rule_rhs: str) -> Iterable[ExpandableRuleOption]:
    if RULE_SIDE_DELIMITER in trimmed_rule_rhs:
        raise ValueError(f"Unexpected symbol `{RULE_SIDE_DELIMITER}`")
    if "\n" in trimmed_rule_rhs:
        raise ValueError("Expected single line as argument")

    for option in trimmed_rule_rhs.split(RULE_OPTION_DELIMITER):
        quantified_symbols = tuple(extract_quantified_symbols(option))
        parsed_option = ExpandableRuleOption(quantified_symbols)
        yield parsed_option


def create_rules(
    lhs: NonTerminal, expandable_options: Iterable[ExpandableRuleOption]
) -> Iterable[ProductionRule]:
    for exp_opt in expandable_options:
        for expansion in expand_rule_option(exp_opt):
            r = ProductionRule(lhs=lhs, rhs=tuple(expansion.symbols))
            yield r


def parse_grammar_line(line: str) -> Iterable[ProductionRule]:
    if not line:
        raise ValueError("Line can not be empty")
    if "\n" in line:
        raise ValueError("Expected single line as argument")

    trimmed = line.strip()
    lhs, trimmed_rhs = extract_rule_lhs(trimmed)
    expandable_rule_options = extract_rule_options(trimmed_rhs)
    return create_rules(lhs, expandable_rule_options)


def extract_grammar_components(
    raw_grammar: str,
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
    if not raw_grammar:
        raise ValueError("raw_grammar can not be empty")

    lines = raw_grammar.splitlines()
    stripped_lines = [ln.strip() for ln in lines]
    relevant_lines = [ln for ln in stripped_lines if ln]

    rules: list[ProductionRule] = []

    for line in relevant_lines:
        extracted_rules = parse_grammar_line(line)
        rules.extend(extracted_rules)

    # we are using dicts instead of sets because the order of insertion in dicts is preserved
    unique_nonterminals: dict[NonTerminal, None] = {}
    unique_terminals: dict[Terminal, None] = {}

    for rule in rules:
        unique_nonterminals[rule.lhs] = None

        for symbol in rule.rhs:
            if isinstance(symbol, NonTerminal):
                unique_nonterminals[symbol] = None
            elif isinstance(symbol, Terminal):
                unique_terminals[symbol] = None
            else:
                raise ValueError(f"Unknown symbol type `{type(symbol)}`")

    nonterminals = tuple(unique_nonterminals.keys())
    terminals = tuple(unique_terminals.keys())
    rules_tuple = tuple(rules)
    start_symbol = rules_tuple[0].lhs

    return nonterminals, terminals, rules_tuple, start_symbol


def validate_grammar_components(
    nonterminals: tuple[NonTerminal, ...],
    terminals: tuple[Terminal, ...],
    rules: tuple[ProductionRule, ...],
    start_symbol: NonTerminal,
) -> None:
    if not nonterminals:
        raise ValueError("nonterminals can not be empty")
    if len(nonterminals) != len(set(nonterminals)):
        raise ValueError("nonterminals can not contain duplicates")

    if not terminals:
        raise ValueError("terminals can not be empty")
    if len(terminals) != len(set(terminals)):
        raise ValueError("terminals can not contain duplicates")

    # no need to check if terminals and nonterminals are disjoint because they have different types
    if not all((isinstance(nt, NonTerminal) for nt in nonterminals)):
        raise ValueError("nonterminals must contain only objects of type NonTerminal")
    if not all((isinstance(t, Terminal) for t in terminals)):
        raise ValueError("terminals must contain only objects of type Terminal")

    if start_symbol not in nonterminals:
        raise ValueError("start_symbol must be contained in nonterminals")

    for nt in nonterminals:
        if not any((nt == r.lhs for r in rules)):
            raise ValueError(
                "All nonterminals must appear at least once in the lhs of a rule"
            )

        if nt == start_symbol:
            continue

        nts_on_rhss = (s for r in rules for s in r.rhs if isinstance(s, NonTerminal))
        if nt not in nts_on_rhss:
            raise ValueError(
                "all nonterminals (except the start symbol) must appear at least once "
                "on the rhs of a rule"
            )

    for nt in nonterminals:
        if nt == start_symbol:
            continue

    for r in rules:
        if r.lhs not in nonterminals:
            raise ValueError("All rule.lhs must be contained in nonterminals")

        for symbol in r.rhs:
            if isinstance(symbol, Terminal):
                if symbol in terminals:
                    continue
                else:
                    raise ValueError(
                        "All terminals on the rhs of a rule must be contained in terminals"
                    )
            elif isinstance(symbol, NonTerminal):
                if symbol in nonterminals:
                    continue
                else:
                    raise ValueError(
                        "All nonterminals on the rhs of a rule must be contained in nonterminals"
                    )
            else:
                raise ValueError(f"Unknown symbol type `{type(symbol)}`")
