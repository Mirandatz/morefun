import multiprocessing as mp
import pathlib
import pickle
import sys
import typing

import attrs
import tensorflow as tf
import typer
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


def configure_logger(log_level: str) -> None:
    logger.remove()
    logger.add(
        sink=sys.stderr,
        level=log_level,
        enqueue=True,
    )


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
) -> list[mp.Process]:
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
        for seed in range(mp.cpu_count())
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
) -> list[Individual]:
    assert pop_size > 0

    # the value for maxsize was arbitrarily chosen
    queue: mp.Queue[Individual] = mp.Queue(maxsize=8 * mp.cpu_count())
    producers = create_producers(queue, grammar, filter)

    for p in producers:
        p.start()

    initial_population = collect_results(pop_size, queue)

    for p in producers:
        p.kill()

    return initial_population


def save_population(population: list[Individual], output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    genotypes = (ind.genotype for ind in population)
    for geno in genotypes:
        path = output_dir / f"{geno.unique_id.hex}.genotype"
        path.write_bytes(pickle.dumps(obj=geno, protocol=pickle.HIGHEST_PROTOCOL))


def main(
    grammar_path: pathlib.Path = typer.Option(
        ...,
        "-g",
        "--grammar-path",
        file_okay=True,
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "-o",
        "--output-dir",
        dir_okay=True,
        readable=True,
        writable=True,
        file_okay=False,
    ),
    pop_size: int = typer.Option(..., "-s", "--population-size", min=1),
    max_network_depth: int = typer.Option(..., "--max-depth", min=1),
    max_wide_layers: int = typer.Option(..., "--max-wide-layers", min=1),
    max_layer_width: int = typer.Option(..., "--max-layer-width", min=1),
    max_network_params: int = typer.Option(..., "--max-network-params", min=1),
    log_level: str = typer.Option("INFO", "--log_level"),
) -> None:
    configure_logger(log_level)
    grammar = gr.Grammar(grammar_path.read_text())

    filter = IndividualFilter(
        max_network_depth=max_network_depth,
        max_wide_layers=max_wide_layers,
        max_layer_width=max_layer_width,
        max_network_params=max_network_params,
    )

    population = create_initial_population(pop_size, grammar, filter)

    save_population(population, output_dir)


if __name__ == "__main__":
    experiment_dir = pathlib.Path(__file__).parent / "cifar10"
    grammar_path = experiment_dir / "grammar.lark"
    output_dir = experiment_dir / "initial_population"
    main(
        grammar_path=grammar_path,
        pop_size=4200,
        max_network_depth=5,
        max_wide_layers=0,
        max_layer_width=512,
        max_network_params=int(1.5 * (10**7)),
        output_dir=pathlib.Path(output_dir),
        log_level="INFO",
    )

# if we want to use this as a script:
# if __name__ == "__main__":
#     typer.run(main)
