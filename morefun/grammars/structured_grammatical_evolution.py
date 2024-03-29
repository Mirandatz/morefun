import collections
import functools

import attrs
from loguru import logger

import morefun.grammars.upper_grammars as ugr
import morefun.randomness as rand


@attrs.frozen(cache_hash=True, order=True)
class Gene:
    """
    Genes are created with order=True because we want to store then
    in a consistent (i.e. sorted) fashion inside `Genotype`s.
    """

    nonterminal: ugr.NonTerminal
    expansions_indices: tuple[int, ...]

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.nonterminal, ugr.NonTerminal)

        assert isinstance(self.expansions_indices, tuple)
        for ei in self.expansions_indices:
            assert isinstance(ei, int)

        assert len(self.expansions_indices) > 0
        assert all(i >= 0 for i in self.expansions_indices)


# slots=False to enable functools.cache_property
@attrs.frozen(cache_hash=True, slots=False)
class Genotype:
    genes: tuple[Gene, ...]

    def __attrs_post_init__(self) -> None:
        sorted_genes = tuple(sorted(self.genes))
        if sorted_genes != self.genes:
            raise ValueError("genes must be sorted")

        if len(self._nonterminals_map) != len(sorted_genes):
            raise ValueError("can not have two genes associated with same nonterminal")

    @functools.cached_property
    def _nonterminals_map(self) -> dict[ugr.NonTerminal, Gene]:
        return {g.nonterminal: g for g in self.genes}

    def get_associated_gene(self, non_terminal: ugr.NonTerminal) -> Gene:
        return self._nonterminals_map[non_terminal]


class GenotypeSkeleton:
    def __init__(
        self,
        sizes_of_gene_lists: dict[ugr.NonTerminal, int],
        max_values_in_gene_lists: dict[ugr.NonTerminal, int],
    ) -> None:
        assert sizes_of_gene_lists.keys() == max_values_in_gene_lists.keys()
        assert len(sizes_of_gene_lists) > 0
        for size in sizes_of_gene_lists.values():
            assert size > 0
        for max_val in max_values_in_gene_lists.values():
            assert max_val >= 0

        self._sizes = sizes_of_gene_lists.copy()
        self._max_values = max_values_in_gene_lists.copy()

    def get_gene_list_size(self, nt: ugr.NonTerminal) -> int:
        return self._sizes[nt]

    def get_gene_list_max_value(self, nt: ugr.NonTerminal) -> int:
        return self._max_values[nt]


@attrs.frozen(cache_hash=True)
class SGEParameters:
    grammar: ugr.Grammar

    @functools.cached_property
    def genotype_skeleton(self) -> GenotypeSkeleton:
        return make_genotype_skeleton(self.grammar)


@functools.cache
def make_genotype_skeleton(grammar: ugr.Grammar) -> GenotypeSkeleton:
    assert not grammar_is_recursive(grammar)

    sizes_of_genes_lists = {
        nt: max_nr_of_times_nonterminal_can_be_expanded(nt, grammar)
        for nt in grammar.nonterminals
    }

    max_values_in_genes_lists = {
        nt: len(grammar.expansions(nt)) for nt in grammar.nonterminals
    }

    return GenotypeSkeleton(
        sizes_of_gene_lists=sizes_of_genes_lists,
        max_values_in_gene_lists=max_values_in_genes_lists,
    )


def create_gene(
    nt: ugr.NonTerminal,
    skeleton: GenotypeSkeleton,
    rng: rand.RNG,
) -> Gene:
    rules_indices = rng.integers(
        low=0,
        high=skeleton.get_gene_list_max_value(nt),
        size=skeleton.get_gene_list_size(nt),
    )
    indices_as_tuple = tuple(int(i) for i in rules_indices)

    gene = Gene(nt, indices_as_tuple)
    logger.debug(f"Creating new gene=<{gene}>")
    return gene


def create_genotype(grammar: ugr.Grammar, rng: rand.RNG) -> Genotype:
    skeleton = make_genotype_skeleton(grammar)
    genes = (create_gene(nt, skeleton, rng) for nt in grammar.nonterminals)
    sorted_genes = sorted(genes)

    genotype = Genotype(tuple(sorted_genes))
    logger.debug(f"Creating new genotype=<{genotype}>")
    return genotype


def map_to_tokenstream(genotype: Genotype, grammar: ugr.Grammar) -> str:
    logger.trace("map_to_tokenstream")
    tokenstream = []

    gene_consumption_tracker = {g: 0 for g in genotype.genes}

    to_process: collections.deque[ugr.Terminal | ugr.NonTerminal] = collections.deque()
    to_process.append(grammar.start_symbol)

    while to_process:
        symbol = to_process.popleft()
        logger.debug(f"Processing symbol=<{symbol}>")

        # sanity check
        assert isinstance(symbol, ugr.Terminal | ugr.NonTerminal)

        if isinstance(symbol, ugr.Terminal):
            tokenstream.append(symbol.text)
            logger.debug(f"Terminal={symbol.text} added to tokenstream")
            continue

        gene = genotype.get_associated_gene(symbol)
        expansions = grammar.expansions(symbol)

        gene_pos = gene_consumption_tracker[gene]
        gene_consumption_tracker[gene] += 1
        exp_choice = gene.expansions_indices[gene_pos]

        chosen_exp = expansions[exp_choice]

        symbols_to_add = reversed(chosen_exp.symbols)
        logger.debug(f"Queueing symbols=<{symbols_to_add}>")
        for s in symbols_to_add:
            to_process.appendleft(s)

    return "".join(tokenstream)


def max_nr_of_times_nonterminal_can_be_expanded(
    target: ugr.NonTerminal,
    grammar: ugr.Grammar,
) -> int:
    assert not grammar_is_recursive(grammar)

    if target == grammar.start_symbol:
        return 1

    max_expansions: collections.Counter[ugr.NonTerminal] = collections.Counter()

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


@functools.cache
def grammar_is_recursive(grammar: ugr.Grammar) -> bool:
    return any(
        can_expand(
            source=nt,
            target=nt,
            grammar=grammar,
        )
        for nt in grammar.nonterminals
    )


def can_expand(
    source: ugr.NonTerminal,
    target: ugr.NonTerminal,
    grammar: ugr.Grammar,
) -> bool:
    explored: set[ugr.NonTerminal] = set()
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
