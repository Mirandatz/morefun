import pathlib
import pickle

import typer
from loguru import logger

import gge.composite_genotypes as cg
import gge.environment_variables
import gge.evolution as evo
import gge.fitnesses as gfit
import gge.grammars as gr
import gge.layers as gl
import gge.novelty as novel
import gge.population_mutation as gmut
import gge.randomness as rand

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
CLASS_COUNT = 10
INPUT_SHAPE = gl.Shape(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, depth=IMAGE_DEPTH)

MAX_GENERATIONS = 100
MUTANTS_PER_GENERATION = 2
MAX_FAILURES = 500

BATCH_SIZE = 32
EPOCHS = 10


def load_initial_population(path: pathlib.Path) -> list[cg.CompositeGenotype]:
    genotypes = [pickle.loads(p.read_bytes()) for p in path.iterdir()]
    assert len(genotypes) > 0
    assert all(isinstance(g, cg.CompositeGenotype) for g in genotypes)
    return genotypes


@logger.catch(reraise=True)
def main() -> None:
    settings = gge.environment_variables.get_paths()
    rng_seed = gge.environment_variables.get_rng_seed()

    logger.add(settings.logging_dir / "log.txt")

    startup_msg = f"""{'-'*80}
        Running with settings:
        {rng_seed=}
        {INPUT_SHAPE=}
        {CLASS_COUNT=}
        {MAX_GENERATIONS=}
        {MUTANTS_PER_GENERATION=}
        {MAX_FAILURES=}
        {BATCH_SIZE=}
        {EPOCHS=}
        {settings.grammar_path=}
        {settings.initial_population_dir=}
        {settings.train_dataset_dir=}
        {settings.validation_dataset_dir=}
        {settings.test_dataset_dir=}
        {settings.initial_population_dir=}
        {settings.output_dir=}
        {settings.logging_dir=}
        {'-'*80}
    """
    logger.info(startup_msg)

    grammar = gr.Grammar(settings.grammar_path.read_text())

    fit_params = gfit.FitnessEvaluationParameters(
        gfit.ValidationAccuracy(
            train_directory=settings.train_dataset_dir,
            validation_directory=settings.validation_dataset_dir,
            input_shape=INPUT_SHAPE,
            batch_size=BATCH_SIZE,
            max_epochs=EPOCHS,
            class_count=CLASS_COUNT,
        ),
        grammar,
    )

    mut_params = gmut.PopulationMutationParameters(
        mutants_to_generate=MUTANTS_PER_GENERATION,
        max_failures=MAX_FAILURES,
        grammar=grammar,
    )

    novelty_tracker = novel.NoveltyTracker()
    rng = rand.create_rng(rng_seed)

    initial_genotypes = load_initial_population(settings.initial_population_dir)

    evaluated_population = {g: gfit.evaluate(g, fit_params) for g in initial_genotypes}

    checkpoint = evo.Checkpoint(settings.output_dir)
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
