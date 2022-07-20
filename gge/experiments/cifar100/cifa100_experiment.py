import pathlib
import pickle

import typer
from loguru import logger

import gge.composite_genotypes as cg
import gge.evolution as evo
import gge.fitnesses as fit
import gge.grammars as gr
import gge.layers as gl
import gge.novelty as novel
import gge.population_mutation as gmut
import gge.randomness as rand
import gge.startup_settings as gge_settings

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
CLASS_COUNT = 100
INPUT_SHAPE = gl.Shape(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, depth=IMAGE_DEPTH)


def load_initial_population(path: pathlib.Path) -> list[cg.CompositeGenotype]:
    genotypes = [pickle.loads(p.read_bytes()) for p in path.iterdir()]
    assert len(genotypes) > 0
    assert all(isinstance(g, cg.CompositeGenotype) for g in genotypes)
    return genotypes


@logger.catch(reraise=True)
def main(
    grammar_path: pathlib.Path = gge_settings.GRAMMAR_PATH,
    initial_population_dir: pathlib.Path = gge_settings.INITIAL_POPULATION_DIR,
    batch_size: int = gge_settings.BATCH_SIZE,
    epochs: int = gge_settings.EPOCHS,
    max_generations: int = gge_settings.MAX_GENERATIONS,
    mutants_per_generation: int = gge_settings.MUTANTS_PER_GENERATION,
    max_failures: int = gge_settings.MAX_FAILURES,
    train_dataset_dir: pathlib.Path = gge_settings.TRAIN_DATASET_DIR,
    validation_dataset_dir: pathlib.Path = gge_settings.VALIDATION_DATASET_DIR,
    output_dir: pathlib.Path = gge_settings.OUTPUT_DIR,
    log_dir: pathlib.Path = gge_settings.LOG_DIR,
    log_level: str = gge_settings.LOG_LEVEL,
    rng_seed: int = gge_settings.RNG_SEED,
) -> None:

    gge_settings.configure_logger(log_dir, log_level)

    grammar = gr.Grammar(grammar_path.read_text())

    fit_params = fit.FitnessEvaluationParameters(
        fit.ValidationAccuracy(
            train_directory=train_dataset_dir,
            validation_directory=validation_dataset_dir,
            input_shape=INPUT_SHAPE,
            batch_size=batch_size,
            max_epochs=epochs,
            class_count=CLASS_COUNT,
        ),
        grammar,
    )

    mut_params = gmut.PopulationMutationParameters(
        mutants_to_generate=mutants_per_generation,
        max_failures=max_failures,
        grammar=grammar,
    )

    novelty_tracker = novel.NoveltyTracker()
    rng = rand.create_rng(rng_seed)

    initial_genotypes = load_initial_population(initial_population_dir)

    evaluated_population = [fit.evaluate(g, fit_params) for g in initial_genotypes]

    evo.run_evolutionary_loop(
        evaluated_population,
        max_generations=max_generations,
        mut_params=mut_params,
        fit_params=fit_params,
        rng=rng,
        novelty_tracker=novelty_tracker,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    typer.run(main)
