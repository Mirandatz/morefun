import pathlib

import keras
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
VALIDATION_RATIO = 0.15

POPULATION_SIZE = 5
MAX_GENERATIONS = 100
MUTANTS_PER_GENERATION = 2
MAX_FAILURES = 500

BATCH_SIZE = 32
EPOCHS = 10


def get_grammar() -> gr.Grammar:
    raw_grammar = """
    start      : conv_block~1..20
    conv_block : "merge" conv norm act "fork"
               | "merge" conv norm act pool "fork"
    conv : "conv2d" "filter_count" (32 | 64 | 128 | 256 | 512 | 1024) "kernel_size" (1 | 3 | 5 | 7) "stride" (1 | 2)
    norm : "batchnorm"
    act  : "relu"
    pool : "pool2d" ("max" | "avg") "stride" 2
    """
    return gr.Grammar(raw_grammar)


def get_cifar10_train_and_val(
    dataset_dir: pathlib.Path,
) -> tuple[
    keras.preprocessing.image.DirectoryIterator,
    keras.preprocessing.image.DirectoryIterator,
]:
    data_gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.15)

    train = data_gen.flow_from_directory(
        dataset_dir,
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        subset="training",
    )

    val = data_gen.flow_from_directory(
        dataset_dir,
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        subset="validation",
    )

    return train, val


def get_input_layer() -> gl.Input:
    return gl.make_input(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)


def validate_output_dir(path: pathlib.Path) -> None:
    logger.debug(f"validating output dir, path=<{path}>")

    if not path.exists():
        logger.info("output dir does not exist and will be created")
        path.mkdir()
        return

    else:
        logger.info("output dir already exists, checking if empty")
        for _ in path.iterdir():
            logger.error("output dir is not empty")
            exit(-1)


def main(
    dataset_dir: pathlib.Path = typer.Option(
        ...,
        "-d",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "-o",
        file_okay=False,
        dir_okay=True,
    ),
    rng_seed: int = typer.Option(rand.get_rng_seed(), "--seed"),
) -> None:
    validate_output_dir(output_dir)

    logger.add(output_dir / "log.txt")
    logger.info(
        f"Running with seed=<{rng_seed}>, saving results to output_dir=<{output_dir}>"
    )

    grammar = get_grammar()
    input_layer = get_input_layer()

    fit_params = gfit.FitnessEvaluationParameters(
        gfit.ValidationAccuracy(
            dataset_dir=dataset_dir,
            input_shape=input_layer.shape,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            class_count=CLASS_COUNT,
            shuffle_seed=rand.get_rng_seed(),
            validation_ratio=VALIDATION_RATIO,
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
