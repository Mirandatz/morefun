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
    represents a combination of a `symbol` with a `repetition quantifier`, such as
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
        raise ValueError("unable to find lhs of rule")

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

    raise ValueError("unable to extract a symbol")


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
        raise ValueError(f"unable to parse repetition range `{text}`")


def extract_quantified_symbols(raw_rule_option: str) -> Iterable[QuantifiedSymbol]:
    if RULE_SIDE_DELIMITER in raw_rule_option:
        raise ValueError(f"unexpected symbol `{RULE_SIDE_DELIMITER}`")
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
        raise ValueError(f"unexpected symbol `{RULE_SIDE_DELIMITER}`")
    if "\n" in trimmed_rule_rhs:
        raise ValueError("expected single line as argument")

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
        raise ValueError("line can not be empty")
    if "\n" in line:
        raise ValueError("expected single line as argument")

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
                raise ValueError(f"unknown symbol type `{type(symbol)}`")

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
                "all nonterminals must appear at least once in the lhs of a rule"
            )

        if nt == start_symbol:
            continue

        # flat_list = [item for sublist in t for item in sublist]
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
            raise ValueError("all rule.lhs must be contained in nonterminals")

        for symbol in r.rhs:
            if isinstance(symbol, Terminal):
                if symbol in terminals:
                    continue
                else:
                    raise ValueError(
                        "all terminals on the rhs of a rule must be contained in terminals"
                    )
            elif isinstance(symbol, NonTerminal):
                if symbol in nonterminals:
                    continue
                else:
                    raise ValueError(
                        "all nonterminals on the rhs of ar ule must be contained in nonterminals"
                    )
            else:
                raise ValueError(f"uknown symbol type `{type(symbol)}`")


if __name__ == "__main__":
    raw = """

        neural_network    : defaults architecture optimizer

        defaults          : default_act
        default_act       : "relu" | "mish"

        architecture      : init_downsampling repr_learning classification

        shortcut_start       : "BLARGHERRRINOOOO"
        shortcut_ends        : output_shape shortcut_start_index~1..10 merge_strat
        output_shape         : rng_seed
        shortcut_start_index : rng_seed
        merge_strat          : "add" | "multiply" | "concatenate"


        init_downsampling : init_block~1..2 shortcut_start
        init_block        : init_conv init_act init_pool init_norm
        init_conv         : init_filter_count init_kernel_size init_strides
        init_filter_count : "32" | "64" | "128" | "256"
        init_kernel_size  : "5" | "7"
        init_strides      : "1" | "2"
        init_act          : "default"
        init_pool         : "max" | "avg"
        init_norm         : "batch_norm"

        repr_learning     : skipy_block~10..15
        skipy_block       : shortcut_ends atomic_block shortcut_start
        atomic_block      : conv_block | pooling_layer

        conv_block        : cb_conv cb_act cb_norm
        cb_conv           : cb_filter_count cb_kernel_size cb_strides
        cb_filter_count   : "32" | "64" | "128" | "256"
        cb_kernel_size    : "3" | "5"
        cb_strides        : "1" | "2"
        cb_act            : "default"
        cb_norm           : "batch_norm"

        pooling_layer     : pl_type pl_size pl_strides
        pl_type           : "max" | "avg"
        pl_size           : "2" | "3"
        pl_strides        : "1" | "2"

        classification    : shortcut_ends flatten_layer dense_blocks
        flatten_layer     : "flatten_layer"
        dense_blocks      : dense_block~1..2
        dense_block       : db_units db_act db_dropout_rate
        db_units          : "32" | "64" | "128"
        db_act            : "default"
        db_dropout_rate   : "0.45" | "0.55" | "0.65"

        optimizer         : adam | radam | ranger

        adam              : learning_rate beta_1 beta_2 epsilon
        learning_rate     : "0.001" | "0.01"
        beta_1            : "0.9"
        beta_2            : "0.999"
        epsilon           : "1e-07"

        radam             : learning_rate beta_1 beta_2 epsilon weight_decay total_steps
        weight_decay      : "0.0"
        total_steps       : "2000" | "4000" | "8000"

        ranger            : inner_optimizer sync_period
        inner_optimizer   : adam | radam
        sync_period       : "6" | "8" | "10"

        rng_seed : "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "13" | "14" | "15" | "16" | "17" | "18" | "19" | "20" | "21" | "22" | "23" | "24" | "25" | "26" | "27" | "28" | "29" | "30" | "31" | "32" | "33" | "34" | "35" | "36" | "37" | "38" | "39" | "40" | "41" | "42" | "43" | "44" | "45" | "46" | "47" | "48" | "49" | "50" | "51" | "52" | "53" | "54" | "55" | "56" | "57" | "58" | "59" | "60" | "61" | "62" | "63" | "64" | "65" | "66" | "67" | "68" | "69" | "70" | "71" | "72" | "73" | "74" | "75" | "76" | "77" | "78" | "79" | "80" | "81" | "82" | "83" | "84" | "85" | "86" | "87" | "88" | "89" | "90" | "91" | "92" | "93" | "94" | "95" | "96" | "97" | "98" | "99" | "100" | "101" | "102" | "103" | "104" | "105" | "106" | "107" | "108" | "109" | "110" | "111" | "112" | "113" | "114" | "115" | "116" | "117" | "118" | "119" | "120" | "121" | "122" | "123" | "124" | "125" | "126" | "127" | "128" | "129" | "130" | "131" | "132" | "133" | "134" | "135" | "136" | "137" | "138" | "139" | "140" | "141" | "142" | "143" | "144" | "145" | "146" | "147" | "148" | "149" | "150" | "151" | "152" | "153" | "154" | "155" | "156" | "157" | "158" | "159" | "160" | "161" | "162" | "163" | "164" | "165" | "166" | "167" | "168" | "169" | "170" | "171" | "172" | "173" | "174" | "175" | "176" | "177" | "178" | "179" | "180" | "181" | "182" | "183" | "184" | "185" | "186" | "187" | "188" | "189" | "190" | "191" | "192" | "193" | "194" | "195" | "196" | "197" | "198" | "199" | "200" | "201" | "202" | "203" | "204" | "205" | "206" | "207" | "208" | "209" | "210" | "211" | "212" | "213" | "214" | "215" | "216" | "217" | "218" | "219" | "220" | "221" | "222" | "223" | "224" | "225" | "226" | "227" | "228" | "229" | "230" | "231" | "232" | "233" | "234" | "235" | "236" | "237" | "238" | "239" | "240" | "241" | "242" | "243" | "244" | "245" | "246" | "247" | "248" | "249" | "250" | "251" | "252" | "253" | "254" | "255" | "256" | "257" | "258" | "259" | "260" | "261" | "262" | "263" | "264" | "265" | "266" | "267" | "268" | "269" | "270" | "271" | "272" | "273" | "274" | "275" | "276" | "277" | "278" | "279" | "280" | "281" | "282" | "283" | "284" | "285" | "286" | "287" | "288" | "289" | "290" | "291" | "292" | "293" | "294" | "295" | "296" | "297" | "298" | "299" | "300" | "301" | "302" | "303" | "304" | "305" | "306" | "307" | "308" | "309" | "310" | "311" | "312" | "313" | "314" | "315" | "316" | "317" | "318" | "319" | "320" | "321" | "322" | "323" | "324" | "325" | "326" | "327" | "328" | "329" | "330" | "331" | "332" | "333" | "334" | "335" | "336" | "337" | "338" | "339" | "340" | "341" | "342" | "343" | "344" | "345" | "346" | "347" | "348" | "349" | "350" | "351" | "352" | "353" | "354" | "355" | "356" | "357" | "358" | "359" | "360" | "361" | "362" | "363" | "364" | "365" | "366" | "367" | "368" | "369" | "370" | "371" | "372" | "373" | "374" | "375" | "376" | "377" | "378" | "379" | "380" | "381" | "382" | "383" | "384" | "385" | "386" | "387" | "388" | "389" | "390" | "391" | "392" | "393" | "394" | "395" | "396" | "397" | "398" | "399" | "400" | "401" | "402" | "403" | "404" | "405" | "406" | "407" | "408" | "409" | "410" | "411" | "412" | "413" | "414" | "415" | "416" | "417" | "418" | "419" | "420" | "421" | "422" | "423" | "424" | "425" | "426" | "427" | "428" | "429" | "430" | "431" | "432" | "433" | "434" | "435" | "436" | "437" | "438" | "439" | "440" | "441" | "442" | "443" | "444" | "445" | "446" | "447" | "448" | "449" | "450" | "451" | "452" | "453" | "454" | "455" | "456" | "457" | "458" | "459" | "460" | "461" | "462" | "463" | "464" | "465" | "466" | "467" | "468" | "469" | "470" | "471" | "472" | "473" | "474" | "475" | "476" | "477" | "478" | "479" | "480" | "481" | "482" | "483" | "484" | "485" | "486" | "487" | "488" | "489" | "490" | "491" | "492" | "493" | "494" | "495" | "496" | "497" | "498" | "499" | "500" | "501" | "502" | "503" | "504" | "505" | "506" | "507" | "508" | "509" | "510" | "511" | "512" | "513" | "514" | "515" | "516" | "517" | "518" | "519" | "520" | "521" | "522" | "523" | "524" | "525" | "526" | "527" | "528" | "529" | "530" | "531" | "532" | "533" | "534" | "535" | "536" | "537" | "538" | "539" | "540" | "541" | "542" | "543" | "544" | "545" | "546" | "547" | "548" | "549" | "550" | "551" | "552" | "553" | "554" | "555" | "556" | "557" | "558" | "559" | "560" | "561" | "562" | "563" | "564" | "565" | "566" | "567" | "568" | "569" | "570" | "571" | "572" | "573" | "574" | "575" | "576" | "577" | "578" | "579" | "580" | "581" | "582" | "583" | "584" | "585" | "586" | "587" | "588" | "589" | "590" | "591" | "592" | "593" | "594" | "595" | "596" | "597" | "598" | "599" | "600" | "601" | "602" | "603" | "604" | "605" | "606" | "607" | "608" | "609" | "610" | "611" | "612" | "613" | "614" | "615" | "616" | "617" | "618" | "619" | "620" | "621" | "622" | "623" | "624" | "625" | "626" | "627" | "628" | "629" | "630" | "631" | "632" | "633" | "634" | "635" | "636" | "637" | "638" | "639" | "640" | "641" | "642" | "643" | "644" | "645" | "646" | "647" | "648" | "649" | "650" | "651" | "652" | "653" | "654" | "655" | "656" | "657" | "658" | "659" | "660" | "661" | "662" | "663" | "664" | "665" | "666" | "667" | "668" | "669" | "670" | "671" | "672" | "673" | "674" | "675" | "676" | "677" | "678" | "679" | "680" | "681" | "682" | "683" | "684" | "685" | "686" | "687" | "688" | "689" | "690" | "691" | "692" | "693" | "694" | "695" | "696" | "697" | "698" | "699" | "700" | "701" | "702" | "703" | "704" | "705" | "706" | "707" | "708" | "709" | "710" | "711" | "712" | "713" | "714" | "715" | "716" | "717" | "718" | "719" | "720" | "721" | "722" | "723" | "724" | "725" | "726" | "727" | "728" | "729" | "730" | "731" | "732" | "733" | "734" | "735" | "736" | "737" | "738" | "739" | "740" | "741" | "742" | "743" | "744" | "745" | "746" | "747" | "748" | "749" | "750" | "751" | "752" | "753" | "754" | "755" | "756" | "757" | "758" | "759" | "760" | "761" | "762" | "763" | "764" | "765" | "766" | "767" | "768" | "769" | "770" | "771" | "772" | "773" | "774" | "775" | "776" | "777" | "778" | "779" | "780" | "781" | "782" | "783" | "784" | "785" | "786" | "787" | "788" | "789" | "790" | "791" | "792" | "793" | "794" | "795" | "796" | "797" | "798" | "799" | "800" | "801" | "802" | "803" | "804" | "805" | "806" | "807" | "808" | "809" | "810" | "811" | "812" | "813" | "814" | "815" | "816" | "817" | "818" | "819" | "820" | "821" | "822" | "823" | "824" | "825" | "826" | "827" | "828" | "829" | "830" | "831" | "832" | "833" | "834" | "835" | "836" | "837" | "838" | "839" | "840" | "841" | "842" | "843" | "844" | "845" | "846" | "847" | "848" | "849" | "850" | "851" | "852" | "853" | "854" | "855" | "856" | "857" | "858" | "859" | "860" | "861" | "862" | "863" | "864" | "865" | "866" | "867" | "868" | "869" | "870" | "871" | "872" | "873" | "874" | "875" | "876" | "877" | "878" | "879" | "880" | "881" | "882" | "883" | "884" | "885" | "886" | "887" | "888" | "889" | "890" | "891" | "892" | "893" | "894" | "895" | "896" | "897" | "898" | "899" | "900" | "901" | "902" | "903" | "904" | "905" | "906" | "907" | "908" | "909" | "910" | "911" | "912" | "913" | "914" | "915" | "916" | "917" | "918" | "919" | "920" | "921" | "922" | "923" | "924" | "925" | "926" | "927" | "928" | "929" | "930" | "931" | "932" | "933" | "934" | "935" | "936" | "937" | "938" | "939" | "940" | "941" | "942" | "943" | "944" | "945" | "946" | "947" | "948" | "949" | "950" | "951" | "952" | "953" | "954" | "955" | "956" | "957" | "958" | "959" | "960" | "961" | "962" | "963" | "964" | "965" | "966" | "967" | "968" | "969" | "970" | "971" | "972" | "973" | "974" | "975" | "976" | "977" | "978" | "979" | "980" | "981" | "982" | "983" | "984" | "985" | "986" | "987" | "988" | "989" | "990" | "991" | "992" | "993" | "994" | "995" | "996" | "997" | "998" | "999" | "1000"

    """
    g = Grammar(raw)
    print(len(g.rules))
