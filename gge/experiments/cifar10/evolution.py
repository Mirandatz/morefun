import pathlib

import typer
from loguru import logger

import gge.evolution as evo
import gge.fitnesses as gfit
import gge.grammars as gr
import gge.layers as gl
import gge.novelty as novel
import gge.population_generation as pop_gen
import gge.population_mutation as gmut
import gge.randomness as rand

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
CLASS_COUNT = 10

POPULATION_SIZE = 5
MAX_GENERATIONS = 100
MUTANTS_PER_GENERATION = 2
MAX_FAILURES = 500

BATCH_SIZE = 32
EPOCHS = 10


def get_grammar() -> gr.Grammar:
    raw_grammar = """
    start : first_block middle_block~2 optimizer

    first_block  : conv_block "fork"
    middle_block : "merge" conv_block "fork"

    conv_block : conv_layer batchnorm activation
               | conv_layer batchnorm activation pooling

    conv_layer : "conv" "filter_count" (32 | 64) "kernel_size" (1 | 3 | 5 | 7) "stride" (1 | 2)

    batchnorm  : "batchnorm"

    activation : relu | swish
    relu : "relu"
    swish : "swish"

    pooling : maxpool | avgpool
    maxpool : "maxpool" "pool_size" (1 | 2) "stride" (1 | 2)
    avgpool : "avgpool" "pool_size" (1 | 2) "stride" (1 | 2)

    optimizer : "adam" "learning_rate" (0.001 | 0.003 | 0.005) "beta1" 0.9 "beta2" 0.999 "epsilon" 1e-07 "amsgrad" false
    """
    return gr.Grammar(raw_grammar)


def get_input_layer() -> gl.Input:
    return gl.make_input(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)


def validate_output_dir(path: pathlib.Path) -> None:
    logger.debug(f"validating output dir, path=<{path}>")

    path.mkdir(parents=True, exist_ok=True)

    for _ in path.iterdir():
        logger.error("output dir is not empty")
        exit(-1)

    logger.success("output dir okay")


@logger.catch(reraise=True)
def main(
    train_dir: pathlib.Path = typer.Option(
        ...,
        "--train",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
    ),
    validation_dir: pathlib.Path = typer.Option(
        ...,
        "--validation",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output",
        file_okay=False,
        dir_okay=True,
    ),
    rng_seed: int = typer.Option(rand.get_rng_seed(), "--seed"),
    clobber: bool = typer.Option(
        False,
        "--clobber",
    ),
) -> None:
    if not clobber:
        validate_output_dir(output_dir)

    logger.add(output_dir / "log.txt")
    logger.info(
        f"Running with seed=<{rng_seed}>, saving results to output_dir=<{output_dir}>"
    )

    grammar = get_grammar()
    input_layer = get_input_layer()

    fit_params = gfit.FitnessEvaluationParameters(
        gfit.ValidationAccuracy(
            train_directory=train_dir,
            validation_directory=validation_dir,
            input_shape=input_layer.shape,
            batch_size=BATCH_SIZE,
            max_epochs=EPOCHS,
            class_count=CLASS_COUNT,
            shuffle_seed=rand.get_rng_seed(),
        ),
        grammar,
        input_layer,
    )

    mut_params = gmut.PopulationMutationParameters(
        mutants_to_generate=MUTANTS_PER_GENERATION,
        max_failures=MAX_FAILURES,
        grammar=grammar,
    )

    novelty_tracker = novel.NoveltyTracker()
    rng = rand.create_rng(rng_seed)

    initial_genotypes = pop_gen.create_initial_population(
        grammar,
        pop_size=POPULATION_SIZE,
        max_failures=MAX_FAILURES,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    evaluated_population = {g: gfit.evaluate(g, fit_params) for g in initial_genotypes}

    checkpoint = evo.Checkpoint(output_dir)
    print_stats = evo.PrintStatistics()

    evo.run_evolutionary_loop(
        evaluated_population,
        max_generations=MAX_GENERATIONS,
        mut_params=mut_params,
        fit_params=fit_params,
        callbacks=[checkpoint, print_stats],
        rng=rng,
        novelty_tracker=novelty_tracker,
    )


if __name__ == "__main__":
    typer.run(main)
