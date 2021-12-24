import collections
import dataclasses
import functools

import typeguard

import gge.grammars as gg
import gge.randomness as rand


# order=True because we want to store genes in a consistent order
@typeguard.typechecked
@dataclasses.dataclass(frozen=True, order=True)
class Gene:
    nonterminal: gg.NonTerminal
    expansions_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        assert len(self.expansions_indices) > 0
        assert all(i >= 0 for i in self.expansions_indices)


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Genotype:
    genes: tuple[Gene, ...]

    def __post_init__(self) -> None:
        sorted_genes = tuple(sorted(self.genes))
        if sorted_genes != self.genes:
            raise ValueError("genes must be sorted")

        if len(self._nonterminals_map) != len(sorted_genes):
            raise ValueError("can not have two genes associated with same nonterminal")

    @functools.cached_property
    def _nonterminals_map(self) -> dict[gg.NonTerminal, Gene]:
        return {g.nonterminal: g for g in self.genes}

    def get_associated_gene(self, non_terminal: gg.NonTerminal) -> Gene:
        return self._nonterminals_map[non_terminal]


class Genemancer:
    def __init__(self, grammar: gg.Grammar):
        assert not grammar_is_recursive(grammar)

        sizes_of_genes_lists = {
            nt: max_nr_of_times_nonterminal_can_be_expanded(nt, grammar)
            for nt in grammar.nonterminals
        }

        max_values_in_genes_lists = {
            nt: len(grammar.expansions(nt)) for nt in grammar.nonterminals
        }

        self._grammar = grammar
        self._sizes_of_genes_lists = sizes_of_genes_lists
        self._max_values_in_genes_lists = max_values_in_genes_lists

    @property
    def grammar(self) -> gg.Grammar:
        return self._grammar

    def create_gene(self, nt: gg.NonTerminal, rng: rand.RNG) -> Gene:
        rules_indices = rng.integers(
            low=0,
            high=self._max_values_in_genes_lists[nt],
            size=self._sizes_of_genes_lists[nt],
        )

        indices_as_tuple = tuple(int(i) for i in rules_indices)
        return Gene(nt, indices_as_tuple)

    def create_genotype(self, rng: rand.RNG) -> Genotype:
        genes = (self.create_gene(nt, rng) for nt in self.grammar.nonterminals)
        sorted_genes = sorted(genes)

        return Genotype(tuple(sorted_genes))

    def map_to_tokenstream(self, genotype: Genotype) -> str:

        """
        This function implements the "genotype mapping" procedure of the GE literature.
        Partially. Because the mapping process usually translates a phenotype to a genotype in a
        complex monolithic process, but in this project, we adopt a "multi-stage mapping approach":
        first we generate a sequence of tokens from the genotype (using this function),
        then we generate a tree from the tokens (using the Lark library);
        finally, we visit the nodes of the tree and synthesize the phenotype.
        """

        tokenstream = ""

        gene_consumption_tracker = {g: 0 for g in genotype.genes}

        to_process: collections.deque[gg.Symbol] = collections.deque()
        to_process.append(self.grammar.start_symbol)

        while to_process:
            symbol = to_process.popleft()

            # sanity check
            assert isinstance(symbol, gg.Terminal | gg.NonTerminal)

            if isinstance(symbol, gg.Terminal):
                tokenstream += symbol.text
                continue

            gene = genotype.get_associated_gene(symbol)
            expansions = self.grammar.expansions(symbol)

            gene_pos = gene_consumption_tracker[gene]
            gene_consumption_tracker[gene] += 1
            exp_choice = gene.expansions_indices[gene_pos]

            chosen_exp = expansions[exp_choice]

            for s in reversed(chosen_exp.symbols):
                to_process.appendleft(s)

        return tokenstream


def max_nr_of_times_nonterminal_can_be_expanded(
    target: gg.NonTerminal,
    grammar: gg.Grammar,
) -> int:
    assert not grammar_is_recursive(grammar)

    if target == grammar.start_symbol:
        return 1

    max_expansions: collections.Counter[gg.NonTerminal] = collections.Counter()

    for rule in grammar.rules:
        if target not in rule.rhs.symbols:
            continue

        lhs = rule.lhs
        rhs = rule.rhs

        times_on_rhs = rhs.symbols.count(target)
        previous_max = max_expansions.get(lhs, 0)
        new_max = max(previous_max, times_on_rhs)
        max_expansions[lhs] = new_max

    for lhs in max_expansions.keys():
        max_expansions[lhs] *= max_nr_of_times_nonterminal_can_be_expanded(
            target=lhs,
            grammar=grammar,
        )

    return sum(max_expansions.values())


def grammar_is_recursive(grammar: gg.Grammar) -> bool:
    return any(
        can_expand(
            source=nt,
            target=nt,
            grammar=grammar,
        )
        for nt in grammar.nonterminals
    )


def can_expand(
    source: gg.NonTerminal,
    target: gg.NonTerminal,
    grammar: gg.Grammar,
) -> bool:
    explored: set[gg.NonTerminal] = set()
    targets_to_explore = [target]

    while targets_to_explore:
        current_target = targets_to_explore.pop()
        explored.add(current_target)

        relevant_rules = [r for r in grammar.rules if current_target in r.rhs.symbols]
        relevant_lhss = [r.lhs for r in relevant_rules]

        if any(lhs == source for lhs in relevant_lhss):
            return True

        targets_to_explore.extend([lhs for lhs in relevant_lhss if lhs not in explored])

    return False
