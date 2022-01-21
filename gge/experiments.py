import functools
import pathlib

import keras
from loguru import logger

import gge.composite_genotypes as cg
import gge.evolution as evo
import gge.fallible as fallible
import gge.fitnesses as gfit
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn
import gge.novelty as novel
import gge.population_mutation as gmut
import gge.randomness as rand

DATASET_DIR = pathlib.Path().home() / "source" / "datasets" / "cifar10" / "train"

OUTPUT_DIR = pathlib.Path().home() / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
CLASS_COUNT = 10
VALIDATION_RATIO = 0.15

POPULATION_SIZE = 5
MAX_GENERATIONS = 150
MUTANTS_PER_GENERATION = 2
MAX_FAILURES = 500

BATCH_SIZE = 32
EPOCHS = 10


def try_create_individual(
    grammar: gr.Grammar,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
) -> cg.CompositeGenotype | None:
    logger.trace("try_create_individual")

    genotype = cg.create_genotype(grammar, rng)

    if novelty_tracker.is_genotype_novel(genotype):
        logger.debug(f"Created novel genotype for initial population=<{genotype}>")
        novelty_tracker.register_genotype(genotype)
    else:
        logger.debug("Failed to produce a novel individual")
        return None

    dummy_input = gl.make_input(1, 1, 1)
    phenotype = gnn.make_network(genotype, grammar, dummy_input)
    if novelty_tracker.is_phenotype_novel(phenotype):
        logger.debug(f"Created novel phenotype for initial population=<{phenotype}>")
        novelty_tracker.register_phenotype(phenotype)
    else:
        logger.debug("Failed to produce a novel phenotype")
        return None

    return genotype


def make_initial_population(
    grammar: gr.Grammar,
    pop_size: int,
    max_failures: int,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
) -> list[cg.CompositeGenotype]:

    generator = functools.partial(
        try_create_individual,
        grammar=grammar,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    initial_pop = fallible.collect_results_from_fallible_function(
        generator,
        num_results=pop_size,
        max_failures=max_failures,
    )

    assert initial_pop is not None
    return initial_pop


def get_grammar() -> gr.Grammar:
    raw_grammar = """
    start      : conv_block~1..10
    conv_block : "merge" conv norm act "fork"
    conv : "conv2d" "filter_count" (32 | 64 | 128 | 256 | 512 | 1024) "kernel_size" (1 | 3 | 5 | 7) "stride" (1 | 2)
    norm : "batchnorm"
    act  : "relu"
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


def main() -> None:
    # logger.remove()
    # logger.add(sink=sys.stderr, level="WARNING")

    grammar = get_grammar()
    input_layer = get_input_layer()

    fit_params = gfit.FitnessEvaluationParameters(
        gfit.ValidationAccuracy(
            dataset_dir=DATASET_DIR,
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
    rng = rand.create_rng()

    initial_genotypes = make_initial_population(
        grammar,
        pop_size=POPULATION_SIZE,
        max_failures=MAX_FAILURES,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    evaluated_population = {g: gfit.evaluate(g, fit_params) for g in initial_genotypes}

    checkpoint = evo.Checkpoint(OUTPUT_DIR)
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
    print(evaluated_population.values())


if __name__ == "__main__":
    main()
