import collections
import itertools
from dataclasses import dataclass

from gge.grammars import Grammar, NonTerminal, Terminal
from gge.randomness import RNG


# order=True because we want to store genes in a consistent order
@dataclass(order=True, frozen=True)
class Gene:
    nonterminal: NonTerminal
    expansions_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        assert len(self.expansions_indices) > 0
        assert all(i >= 0 for i in self.expansions_indices)


class Genotype:
    _instances_created = itertools.count()

    def __init__(self, genes: tuple[Gene, ...]):
        assert genes == tuple(sorted(genes))

        nonterminals_map = {g.nonterminal: g for g in genes}
        assert len(nonterminals_map) == len(genes)

        self._id = next(Genotype._instances_created)
        self._genes = genes
        self._nonterminals_map = nonterminals_map
        self._hash = hash(self._genes)

    @property
    def id(self) -> int:
        return self._id

    @property
    def genes(self) -> tuple[Gene, ...]:
        return self._genes

    def get_associated_gene(self, non_terminal: NonTerminal) -> Gene:
        return self._nonterminals_map[non_terminal]

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Genotype):
            return NotImplemented

        if self.id == other.id:
            return True

        return self.genes == other.genes

    def __str__(self) -> str:
        gene_strs = (str(g) for g in self.genes)
        return "\n".join(gene_strs)

    def __repr__(self) -> str:
        return self.__str__()


class Genemancer:
    def __init__(self, grammar: Grammar):
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
    def grammar(self) -> Grammar:
        return self._grammar

    def create_gene(self, nt: NonTerminal, rng: RNG) -> Gene:
        rules_indices = rng.integers(
            low=0,
            high=self._max_values_in_genes_lists[nt],
            size=self._sizes_of_genes_lists[nt],
        )

        indices_as_tuple = tuple(int(i) for i in rules_indices)
        return Gene(nt, indices_as_tuple)

    def create_genotype(self, rng: RNG) -> Genotype:
        genes = [self.create_gene(nt, rng) for nt in self.grammar.nonterminals]
        return Genotype(tuple(genes))

    def map_to_tokens(self, genotype: Genotype) -> tuple[str, ...]:
        terminals: collections.deque[Terminal] = collections.deque()

        gene_consumption_tracker = {g: 0 for g in genotype.genes}
        to_process = [self.grammar.start_symbol]

        while to_process:
            nt = to_process.pop()

            gene = genotype.get_associated_gene(nt)
            expansions = self.grammar.expansions(nt)

            gene_pos = gene_consumption_tracker[gene]
            gene_consumption_tracker[gene] += 1
            exp_choice = gene.expansions_indices[gene_pos]

            chosen_exp = expansions[exp_choice]

            for symbol in chosen_exp:
                if type(symbol) == NonTerminal:
                    to_process.append(symbol)

                elif type(symbol) == Terminal:
                    terminals.appendleft(symbol)

                else:
                    raise TypeError(f"unknown symbol type `{type(symbol)}`")

        return tuple(t.text for t in terminals)


def max_nr_of_times_nonterminal_can_be_expanded(
    target: NonTerminal,
    grammar: Grammar,
) -> int:
    assert not grammar_is_recursive(grammar)
    assert target not in grammar.terminals  # type: ignore

    if target == grammar.start_symbol:
        return 1

    max_expansions: dict[NonTerminal, int] = {}

    for rule in grammar.rules:
        if target not in rule.rhs:
            continue

        lhs = rule.lhs
        rhs = rule.rhs

        times_on_rhs = rhs.count(target)
        previous_max = max_expansions.get(lhs, 0)
        new_max = max(previous_max, times_on_rhs)
        max_expansions[lhs] = new_max

    for lhs in max_expansions.keys():
        max_expansions[lhs] *= max_nr_of_times_nonterminal_can_be_expanded(
            target=lhs,
            grammar=grammar,
        )

    return sum(max_expansions.values())


def grammar_is_recursive(grammar: Grammar) -> bool:
    return any(
        can_expand(
            source=nt,
            target=nt,
            grammar=grammar,
        )
        for nt in grammar.nonterminals
    )


def can_expand(
    source: NonTerminal,
    target: NonTerminal,
    grammar: Grammar,
) -> bool:
    explored: set[NonTerminal] = set()
    targets_to_explore = [target]

    while targets_to_explore:
        current_target = targets_to_explore.pop()
        explored.add(current_target)

        relevant_rules = [r for r in grammar.rules if current_target in r.rhs]
        relevant_lhss = [r.lhs for r in relevant_rules]

        if any(lhs == source for lhs in relevant_lhss):
            return True

        targets_to_explore.extend([lhs for lhs in relevant_lhss if lhs not in explored])

    return False
