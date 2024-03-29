import enum
import functools
import itertools
import typing

import attrs
import lark
from loguru import logger

import morefun.grammars.transformers as mf_transformers
import morefun.paths


def get_upper_grammar_parser() -> lark.Lark:
    logger.debug("start parsing upper_grammar")

    upper_grammar_path = morefun.paths.get_grammars_dir() / "upper_grammar.lark"

    parser = lark.Lark.open(
        str(upper_grammar_path),
        parser="lalr",
        maybe_placeholders=True,
    )
    logger.debug("finished parsing upper_grammar")
    return parser


def extract_ast(grammar_text: str) -> lark.Tree[typing.Any]:
    parser = get_upper_grammar_parser()
    return parser.parse(grammar_text)


def _can_be_parsed_as_float(value: str) -> bool:
    try:
        assert value[0] == '"'
        assert value[-1] == '"'
        unquoted = value[1:-1]
        _ = float(unquoted)
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


@attrs.frozen(cache_hash=True, order=True)
class NonTerminal:
    text: str

    def __attrs_post_init__(self) -> None:
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


@attrs.frozen(cache_hash=True, order=True)
class Terminal:
    text: str

    def __attrs_post_init__(self) -> None:
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
            f"invalid text=<{self.text}>"
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


@attrs.frozen(cache_hash=True, order=True)
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

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.symbols, tuple)
        for s in self.symbols:
            assert isinstance(s, Symbol)  # type: ignore

        assert len(self.symbols) >= 1

    def __repr__(self) -> str:
        all_options = ",".join(repr(s) for s in self.symbols)
        return f"RuleOption({all_options})"


@attrs.frozen(cache_hash=True, order=True)
class ProductionRule:
    lhs: NonTerminal
    rhs: RuleOption

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.lhs, NonTerminal)
        assert isinstance(self.rhs, RuleOption)

    def __repr__(self) -> str:
        return f"Rule({self.lhs}->{self.rhs})"


@attrs.frozen(cache_hash=True)
class GrammarComponents:
    nonterminals: tuple[NonTerminal, ...]
    terminals: tuple[Terminal, ...]
    rules: tuple[ProductionRule, ...]
    start_symbol: NonTerminal

    def __attrs_post_init__(self) -> None:
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

        # Uncomment to disallow the usage repeated rules.
        # Repeating a rule increases its likelihood of being chosen
        # by SGE.
        # assert len(P) == len(set(P))

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

        logger.debug(f"Non-terminal=<{nt}> expands into rhss=<{rhss_as_tuple}>")

        return rhss_as_tuple

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if id(self) == id(other):
            return True

        if not isinstance(other, Grammar):
            return NotImplemented

        return self._as_tuple == other._as_tuple


class ExpectedTerminal(enum.Enum):
    RANDOM_FLIP = Terminal('"random_flip"')
    RANDOM_ROTATION = Terminal('"random_rotation"')

    RESIZING = Terminal('"resizing"')
    HEIGHT = Terminal('"height"')
    WIDTH = Terminal('"width"')

    RANDOM_CROP = Terminal('"random_crop"')

    RANDOM_TRANSLATION = Terminal('"random_translation"')

    DENSE = Terminal('"dense"')

    CONV = Terminal('"conv"')
    FILTER_COUNT = Terminal('"filter_count"')
    KERNEL_SIZE = Terminal('"kernel_size"')
    STRIDE = Terminal('"stride"')

    BATCHRNOM = Terminal('"batchnorm"')

    MAXPOOL = Terminal('"maxpool"')
    AVGPOOL = Terminal('"avgpool"')
    POOL_SIZE = Terminal('"pool_size"')

    RELU = Terminal('"relu"')
    GELU = Terminal('"gelu"')
    SWISH = Terminal('"swish"')
    PRELU = Terminal('"prelu"')

    SGD = Terminal('"sgd"')
    LEARNING_RATE = Terminal('"learning_rate"')
    MOMENTUM = Terminal('"momentum"')
    NESTEROV = Terminal('"nesterov"')

    ADAM = Terminal('"adam"')
    BETA1 = Terminal('"beta1"')
    BETA2 = Terminal('"beta2"')
    EPSILON = Terminal('"epsilon"')
    AMSGRAD = Terminal('"amsgrad"')

    RANGER = Terminal('"ranger"')
    SYNC_PERIOD = Terminal('"sync_period"')
    SLOW_STEP_SIZE = Terminal('"slow_step_size"')


MarkerValuePair = tuple[Terminal, Terminal]


def make_list_of_marker_value_pairs(parts: list[Terminal]) -> list[MarkerValuePair]:
    """
    During the "tree visit/transformation" we often must transform
    lists that look like `["a", 1, 2, 3]`
    into lists that look like `[["a", 1], ["a", 2], ["a", 3]]`.
    This function performs such transformation.
    """

    marker, *values = parts

    assert isinstance(marker, Terminal)
    assert isinstance(values, list)
    for v in values:
        assert isinstance(v, Terminal)

    return [(marker, v) for v in values]


def make_list_of_options(parts: typing.Any) -> list[RuleOption]:
    """
    During the "tree visit/transformation" we often must transform
    lists that look like
    [
        "a",
        [("b", 1), ("b", 2), ("b", 3)],
        [("c", 7), ("c", 8), ("c", 8)]
    ]
    into lists that look like
    [
        RuleOption(("a", "b", 1, "c", 7)),
        RuleOption(("a", "b", 1, "c", 8)),
        RuleOption(("a", "b", 1, "c", 9)),
        ...
    ]
    This function performs such transformation.
    """

    marker, *params = parts

    assert isinstance(marker, Terminal)
    assert isinstance(params, list)
    for param_mvps in params:
        assert isinstance(param_mvps, list)
        for pvmp in param_mvps:
            param_marker, param_value = pvmp
            assert isinstance(param_marker, Terminal)
            assert isinstance(param_value, Terminal)

    rule_options = []
    for combination in itertools.product(*params):
        expanded = list(itertools.chain.from_iterable(combination))
        opt = RuleOption((marker, *expanded))
        rule_options.append(opt)

    return rule_options


class GrammarTransformer(mf_transformers.SinglePassTransformer):
    # This set contains names of grammar rules that always appear in the form
    # `rule_name: marker value` and which the node-visiting process consists in
    # calling `make_list_of_marker_value_pairs`.
    # This set is used  to remove a lot of boilerplate code.
    # For more information, see:
    # - `make_list_of_marker_value_pairs`.
    # - `GrammarTransformer.__default__`
    _rules_of_marker_value_pairs = {
        # resizing
        "height",
        "width",
        # conv
        "filter_count",
        "kernel_size",
        "strides",
        # pooling
        "pool_sizes",
        # sgd
        "learning_rate",
        "momentum",
        "nesterov",
        # adam
        "beta1",
        "beta2",
        "epsilon",
        "amsgrad",
        # ranger
        "sync_period",
        "slow_step_size",
    }

    # This set is also used to remove boilplate code.
    # For more information, see: `GrammarTransformer.__default_token__`
    _known_terminals = {terminal.value.text for terminal in ExpectedTerminal}

    def __init__(self) -> None:
        super().__init__()
        self._terminals: list[Terminal] = []
        self._nonterminals: list[NonTerminal] = []
        self._rules: list[ProductionRule] = []
        self._start: NonTerminal = NonTerminal("start")

    def __default__(
        self,
        data: lark.Token,
        children: list[typing.Any],
        meta: typing.Any,
    ) -> typing.Any:
        self._raise_if_not_running()

        if data.value in self._rules_of_marker_value_pairs:
            return make_list_of_marker_value_pairs(children)

        return super().__default__(data, children, meta)

    def __default_token__(self, token: lark.Token) -> typing.Any:
        self._raise_if_not_running()

        if token.value in self._known_terminals:
            return self._register_terminal(token.value)

        if token.value in ['"fork"', '"merge"']:
            # raise NotImplementedError("WRITE TESTS")
            return self._register_terminal(token.value)

        return super().__default_token__(token)

    def transform(self, tree: lark.Tree[GrammarComponents]) -> GrammarComponents:
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

    def start(
        self, list_of_list_of_rules: list[list[ProductionRule]]
    ) -> list[ProductionRule]:
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

    def block(
        self, list_of_list_of_options: list[list[RuleOption]]
    ) -> list[RuleOption]:
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

    @lark.v_args(inline=False)
    def random_flip(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()

        marker, *modes = parts
        assert isinstance(marker, Terminal)
        assert isinstance(modes, list)
        assert all(isinstance(m, Terminal) for m in modes)
        assert len(modes) >= 1

        rule_options = []
        for m in modes:
            symbols = marker, m
            opt = RuleOption(symbols)
            rule_options.append(opt)

        return rule_options

    @lark.v_args(inline=False)
    def random_rotation(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()

        marker, *rotations = parts
        assert isinstance(marker, Terminal)
        assert isinstance(rotations, list)
        assert all(isinstance(r, Terminal) for r in rotations)
        assert len(rotations) >= 1

        return [RuleOption((marker, r)) for r in rotations]

    def resizing(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    def random_crop(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    def random_translation(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        marker, *factors = parts
        assert isinstance(marker, Terminal)
        assert isinstance(factors, list)
        assert all(isinstance(f, Terminal) for f in factors)
        assert len(factors) >= 1

        return [RuleOption((marker, f)) for f in factors]

    # this rule exists only to make the upper_grammar.lark cleaner
    @lark.v_args(inline=True)
    def layer_and_maybe_emptylines(self, whatever: typing.Any) -> typing.Any:
        self._raise_if_not_running()
        return whatever

    def layer(
        self, list_of_lists_of_options: list[list[RuleOption]]
    ) -> list[RuleOption]:
        self._raise_if_not_running()

        return [
            opt
            for list_of_options in list_of_lists_of_options
            for opt in list_of_options
        ]

    def dense_layer(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()

        marker, *num_neurons = parts
        assert isinstance(marker, Terminal)
        assert isinstance(num_neurons, list)
        assert all(isinstance(nn, Terminal) for nn in num_neurons)
        assert len(num_neurons) >= 1
        return [RuleOption((marker, r)) for r in num_neurons]

    def conv_layer(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    @lark.v_args(inline=True)
    def activation_layer(self, activation: NonTerminal) -> list[RuleOption]:
        self._raise_if_not_running()
        return [RuleOption((activation,))]

    @lark.v_args(inline=True)
    def batchnorm_layer(self, term: Terminal) -> list[RuleOption]:
        return [RuleOption((term,))]

    def max_pooling_layer(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    def avg_pooling_layer(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    # this rule exists only to make the upper_grammar.lark cleaner
    @lark.v_args(inline=True)
    def optimizer_and_maybe_emptylines(self, whatever: typing.Any) -> typing.Any:
        self._raise_if_not_running()
        return whatever

    @lark.v_args(inline=False)
    def optimizer(
        self,
        list_of_lists_of_options: list[list[RuleOption]],
    ) -> list[RuleOption]:
        self._raise_if_not_running()
        return [
            opt
            for list_of_options in list_of_lists_of_options
            for opt in list_of_options
        ]

    def sgd(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    def adam(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    def ranger(self, parts: typing.Any) -> list[RuleOption]:
        self._raise_if_not_running()
        return make_list_of_options(parts)

    def NONTERMINAL(self, token: lark.Token) -> NonTerminal:
        self._raise_if_not_running()
        return self._register_nonterminal(token.value)

    def RANGE_BOUND(self, token: lark.Token) -> int:
        self._raise_if_not_running()
        return int(token.value)

    def INT_ARG(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def FLOAT_ARG(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        return self._register_terminal(token.value)

    def BOOL_ARG(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        assert isinstance(token, lark.Token)
        return self._register_terminal(token.value)

    def FLIP_MODE(self, token: lark.Token) -> Terminal:
        self._raise_if_not_running()
        assert isinstance(token, lark.Token)
        return self._register_terminal(token.value)
