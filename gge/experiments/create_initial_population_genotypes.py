import multiprocessing as mp
import typing

import attrs
import tensorflow as tf
from loguru import logger

import gge.composite_genotypes as cg
import gge.grammars as gr
import gge.layers as gl
import gge.novelty as novel
import gge.phenotypes as pheno
import gge.randomness as rand


@attrs.frozen
class Individual:
    genotype: cg.CompositeGenotype
    phenotype: pheno.Phenotype


@attrs.frozen
class IndividualFilter:
    max_network_depth: int
    max_wide_layers: int
    max_layer_width: int
    max_network_params: int

    def __attrs_post_init__(self) -> None:
        assert self.max_network_depth >= 1
        assert self.max_wide_layers >= 0
        assert self.max_layer_width >= 1
        assert self.max_network_params >= 1


def is_network_too_deep(
    ind: Individual,
    max_depth: int,
) -> bool:
    assert max_depth > 0

    learning_layers = [
        layer
        for layer in ind.phenotype.backbone.layers
        if isinstance(layer, gl.Conv2D | gl.Conv2DTranspose)
    ]
    num_real_layers = len(learning_layers)
    network_too_deep = num_real_layers > max_depth

    logger.info(
        f"genotype {ind.genotype} describes a network with {num_real_layers} layers,"
        f" network considered too deep=<{network_too_deep}>"
    )

    return network_too_deep


def is_layer_too_wide(layer: gl.Layer, max_filters: int) -> bool:
    assert max_filters > 0

    if not isinstance(layer, gl.Conv2D | gl.Conv2DTranspose):
        return False

    return layer.filter_count > max_filters


def is_network_too_wide(
    ind: Individual,
    max_wide_layers: int,
    max_layer_width: int,
) -> bool:
    assert max_wide_layers >= 0
    assert max_layer_width > 0

    wide_layers = [
        layer
        for layer in ind.phenotype.backbone.layers
        if is_layer_too_wide(layer, max_layer_width)
    ]
    n_wide_layers = len(wide_layers)
    network_too_wide = n_wide_layers > max_wide_layers

    logger.info(
        f"genotype {ind.genotype} describes a network with {n_wide_layers}"
        f" layers with more than {max_layer_width} filters,"
        f" network considered too wide=<{network_too_wide}>"
    )

    return network_too_wide


def is_network_overparameterized(ind: Individual, max_params: int) -> bool:

    with tf.device("/device:CPU:0"):
        input_tensor, output_tensor = pheno.make_input_output_tensors(
            ind.phenotype,
            input_layer=gl.make_input(width=1, height=1, depth=1),
        )

        model = tf.keras.Model(
            inputs=input_tensor,
            outputs=output_tensor,
        )

        params: int = model.count_params()
        overparameterized = params > max_params

        logger.info(
            f"genotype {ind.genotype=} describes a network with {params} params,"
            f" discarded as overparameterize=<{overparameterized}>"
        )

        return overparameterized


def should_consider_for_population(
    ind: Individual,
    filter_params: IndividualFilter,
) -> bool:
    return not any(
        (
            is_network_too_deep(
                ind,
                max_depth=filter_params.max_network_depth,
            ),
            is_network_too_wide(
                ind,
                max_wide_layers=filter_params.max_wide_layers,
                max_layer_width=filter_params.max_layer_width,
            ),
            is_network_overparameterized(
                ind,
                max_params=filter_params.max_network_params,
            ),
        )
    )


def should_add_to_population(
    ind: Individual,
    tracker: novel.NoveltyTracker,
) -> bool:

    if not tracker.is_genotype_novel(ind.genotype):
        return False

    if not tracker.is_phenotype_novel(ind.phenotype):
        tracker.register_genotype(ind.genotype)
        return False

    tracker.register_genotype(ind.genotype)
    tracker.register_phenotype(ind.phenotype)
    return True


def create_individuals(
    queue: "mp.Queue[Individual]",
    grammar: gr.Grammar,
    filter: IndividualFilter,
    rng_seed: int,
) -> typing.NoReturn:
    rng = rand.create_rng(rng_seed)
    while True:
        genotype = cg.create_genotype(grammar, rng)
        phenotype = pheno.translate(genotype, grammar)
        ind = Individual(genotype, phenotype)

        if should_consider_for_population(ind, filter):
            logger.info(
                f"genotype {ind.genotype} passed the individual filter"
                " and was added to the initial population candidates"
            )
            queue.put(ind)
        else:
            logger.info(
                f"genotype {ind.genotype} dit not pass pass the individual filter"
                " and was not added to the initial population candidates"
            )


def create_producers(
    queue: "mp.Queue[Individual]",
    grammar: gr.Grammar,
    filter: IndividualFilter,
    rng_seed: int,
) -> list[mp.Process]:
    rng = rand.create_rng(rng_seed)
    first_worker_seed = int(rng.integers(low=0, high=2**30, size=1))
    worker_seeds = range(
        first_worker_seed,
        first_worker_seed + mp.cpu_count(),
    )

    return [
        mp.Process(
            target=create_individuals,
            kwargs={
                "queue": queue,
                "grammar": grammar,
                "filter": filter,
                "rng_seed": seed,
            },
            daemon=True,
        )
        for seed in worker_seeds
    ]


def collect_results(
    result_count: int,
    queue: "mp.Queue[Individual]",
) -> list[Individual]:
    assert result_count > 0

    results: list[Individual] = []
    tracker = novel.NoveltyTracker()

    while len(results) < result_count:
        ind = queue.get()

        if should_add_to_population(ind, tracker):
            logger.info(
                f"genotype {ind.genotype} passed the population filter"
                " and was added to the initial population"
            )
            results.append(ind)

        else:
            logger.info(
                f"genotype {ind.genotype} dit not pass the population filter"
                " and was discarded"
            )

    return results


def create_initial_population(
    pop_size: int,
    grammar: gr.Grammar,
    filter: IndividualFilter,
    rng_seed: int,
) -> list[Individual]:
    assert pop_size > 0

    # the value for maxsize was arbitrarily chosen
    queue: mp.Queue[Individual] = mp.Queue(maxsize=8 * mp.cpu_count())
    producers = create_producers(queue, grammar, filter, rng_seed)

    for p in producers:
        p.start()

    initial_population = collect_results(pop_size, queue)

    for p in producers:
        p.kill()

    return initial_population
