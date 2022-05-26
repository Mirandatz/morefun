import functools
import pathlib
import pickle

import attrs
import tensorflow as tf
import typer
from loguru import logger

import gge.composite_genotypes as cg
import gge.fallible as fallible
import gge.grammars as gr
import gge.layers as gl
import gge.novelty as novel
import gge.phenotypes as pheno
import gge.phenotypes as gph
import gge.randomness as rand


@attrs.frozen
class PhenotypeFilter:
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
    import sys

    logger.remove()
    logger.add(
        sink=sys.stderr,
        level=log_level,
    )


def is_network_too_deep(
    phenotype: pheno.Phenotype,
    max_depth: int,
) -> bool:
    assert max_depth > 0

    # TODO: add 'has_parameters' method to layers
    layers_with_parameters = [
        layer
        for layer in phenotype.backbone.layers
        if isinstance(layer, gl.Conv2D | gl.Conv2DTranspose)
    ]
    num_real_layers = len(layers_with_parameters)
    network_too_deep = num_real_layers > max_depth

    logger.debug(
        f"phenotype<{phenotype}> describes a network with {num_real_layers} layers."
        f" considered too deep=<{network_too_deep}>"
    )

    return network_too_deep


def is_layer_too_wide(layer: gl.Layer, max_filters: int) -> bool:
    assert max_filters > 0

    if not isinstance(layer, gl.Conv2D | gl.Conv2DTranspose):
        return False

    return layer.filter_count > max_filters


def is_network_too_wide(
    phenotype: pheno.Phenotype,
    max_wide_layers: int,
    max_layer_width: int,
) -> bool:
    assert max_wide_layers >= 0
    assert max_layer_width > 0

    wide_layers = [
        layer
        for layer in phenotype.backbone.layers
        if is_layer_too_wide(layer, max_layer_width)
    ]
    n_wide_layers = len(wide_layers)
    network_too_wide = n_wide_layers > max_wide_layers

    logger.debug(
        f"phenotype<{phenotype}> describes a network with {n_wide_layers}"
        f" layers with more than {max_layer_width} filters."
        f" considered too wide=<{network_too_wide}>"
    )

    return network_too_wide


def is_network_overparameterized(phenotype: pheno.Phenotype, max_params: int) -> bool:
    assert max_params > 0

    with tf.device("/device:CPU:0"):
        input_tensor, output_tensor = pheno.make_input_output_tensors(
            phenotype,
            input_layer=gl.make_input(width=1, height=1, depth=1),
        )

        model = tf.keras.Model(
            inputs=input_tensor,
            outputs=output_tensor,
        )

        params: int = model.count_params()
        overparameterized = params > max_params

        logger.debug(
            f"phenotype<{phenotype}> describes a network with {params} params."
            f" considered overparameterize=<{overparameterized}>"
        )

        return overparameterized


def passes_phenotypical_filter(
    phenotype: pheno.Phenotype,
    filter_params: PhenotypeFilter,
) -> bool:
    return not any(
        (
            is_network_too_deep(
                phenotype,
                max_depth=filter_params.max_network_depth,
            ),
            is_network_too_wide(
                phenotype,
                max_wide_layers=filter_params.max_wide_layers,
                max_layer_width=filter_params.max_layer_width,
            ),
            is_network_overparameterized(
                phenotype,
                max_params=filter_params.max_network_params,
            ),
        )
    )


def try_create_individual(
    grammar: gr.Grammar,
    filter: PhenotypeFilter,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
) -> cg.CompositeGenotype | None:
    """
    Attempts to create a novel individual, that is,
    a genotype that is not yet tracked by `novelty_tracker`,
    whose phenotype is also not yet tracked by `novelty_tracker`,
    and whose phenotype passes all criteria of `filter`.

    If the created individual passes all tests, its genotype is returned.
    Returns `None` if it fails otherwise.
    """

    logger.trace("try_create_individual")

    genotype = cg.create_genotype(grammar, rng)
    genotype_is_novel = novelty_tracker.is_genotype_novel(genotype)
    if not genotype_is_novel:
        logger.info(f"discarding genotype=<{genotype}>")
        return None

    novelty_tracker.register_genotype(genotype)

    phenotype = gph.translate(genotype, grammar)
    phenotype_is_novel = novelty_tracker.is_phenotype_novel(phenotype)
    if not phenotype_is_novel:
        logger.info(f"discarding genotype=<{genotype}>")
        return None

    novelty_tracker.register_phenotype(phenotype)

    if passes_phenotypical_filter(phenotype, filter):
        logger.success(f"adding genotype=<{genotype}>")
        return genotype
    else:
        logger.info(f"discarding genotype=<{genotype}>")
        return None


def create_initial_population(
    pop_size: int,
    grammar: gr.Grammar,
    max_failures: int,
    filter: PhenotypeFilter,
    rng_seed: int,
) -> list[cg.CompositeGenotype]:
    assert pop_size > 0
    assert max_failures >= 0

    novelty_tracker = novel.NoveltyTracker()
    rng = rand.create_rng(rng_seed)

    generator = functools.partial(
        try_create_individual,
        grammar=grammar,
        novelty_tracker=novelty_tracker,
        filter=filter,
        rng=rng,
    )

    initial_pop = fallible.collect_results_from_fallible_function(
        generator,
        num_results=pop_size,
        max_failures=max_failures,
    )

    if initial_pop is None:
        logger.error(
            f"unable to generate initial population with size=<{pop_size}>."
            f" reason: maximum failures=<{max_failures}> reached"
        )
        exit(-1)

    return initial_pop


def save_population(
    genotypes: list[cg.CompositeGenotype],
    output_dir: pathlib.Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for geno in genotypes:
        path = output_dir / f"{geno.unique_id.hex}.genotype"
        path.write_bytes(pickle.dumps(obj=geno, protocol=pickle.HIGHEST_PROTOCOL))


def validate_output_dir(path: pathlib.Path) -> None:
    logger.debug(f"validating output dir, path=<{path}>")

    path.mkdir(parents=True, exist_ok=True)

    for _ in path.iterdir():
        logger.error("output dir is not empty")
        exit(-1)


def main(
    grammar_path: pathlib.Path = typer.Option(
        ...,
        "--grammar-path",
        file_okay=True,
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        dir_okay=True,
        readable=True,
        writable=True,
        file_okay=False,
    ),
    pop_size: int = typer.Option(..., "--population-size", min=1),
    max_failures: int = typer.Option(..., "--max-failures", min=0),
    max_network_depth: int = typer.Option(..., "--max-network-depth", min=1),
    max_wide_layers: int = typer.Option(..., "--max-wide-layers", min=0),
    max_layer_width: int = typer.Option(..., "--max-layer-width", min=1),
    max_network_params: int = typer.Option(..., "--max-network-params", min=1),
    rng_seed: int = typer.Option(42, "--rng-seed"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    configure_logger(log_level)
    validate_output_dir(output_dir)

    grammar = gr.Grammar(grammar_path.read_text())

    filter = PhenotypeFilter(
        max_network_depth=max_network_depth,
        max_wide_layers=max_wide_layers,
        max_layer_width=max_layer_width,
        max_network_params=max_network_params,
    )

    population = create_initial_population(
        pop_size=pop_size,
        grammar=grammar,
        max_failures=max_failures,
        filter=filter,
        rng_seed=rng_seed,
    )

    save_population(population, output_dir)


if __name__ == "__main__":
    typer.run(main)
