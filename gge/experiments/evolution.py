import pathlib
import pickle
import typing

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


def load_initial_population(path: pathlib.Path) -> list[cg.CompositeGenotype]:
    genotypes = [pickle.loads(p.read_bytes()) for p in path.iterdir()]
    assert len(genotypes) > 0
    assert all(isinstance(g, cg.CompositeGenotype) for g in genotypes)
    return genotypes


@logger.catch(reraise=True)
def main(
    # evolution
    max_generations: int = gge_settings.MAX_GENERATIONS,
    mutants_per_generation: int = gge_settings.MUTANTS_PER_GENERATION,
    max_failures: int = gge_settings.MAX_FAILURES,
    batch_size: int = gge_settings.BATCH_SIZE,
    epochs: int = gge_settings.EPOCHS,
    # dataset
    img_width: int = gge_settings.IMAGE_WIDTH,
    img_height: int = gge_settings.IMAGE_HEIGHT,
    img_depth: int = gge_settings.IMAGE_DEPTH,
    class_count: int = gge_settings.CLASS_COUNT,
    # paths
    grammar_path: pathlib.Path = gge_settings.GRAMMAR_PATH,
    initial_population_dir: pathlib.Path = gge_settings.INITIAL_POPULATION_DIR,
    train_dataset_dir: pathlib.Path = gge_settings.TRAIN_DATASET_DIR,
    validation_dataset_dir: pathlib.Path = gge_settings.VALIDATION_DATASET_DIR,
    output_dir: pathlib.Path = gge_settings.OUTPUT_DIR,
    # log
    log_dir: pathlib.Path = gge_settings.LOG_DIR,
    log_level: str = gge_settings.LOG_LEVEL,
    # the ticket
    rng_seed: int = gge_settings.RNG_SEED,
    # sanity check
    expected_train_instances: typing.Optional[
        int
    ] = gge_settings.EXPECTED_NUMBER_OF_TRAIN_INSTANCES,
    expected_validation_instances: typing.Optional[
        int
    ] = gge_settings.EXPECTED_NUMBER_OF_VALIDATION_INSTANCES,
) -> None:
    gge_settings.configure_logger(log_dir, log_level)

    if expected_train_instances is not None:
        logger.info("inspecting train dataset")
        gge_settings.validate_dataset_dir(
            path=train_dataset_dir,
            img_height=img_height,
            img_width=img_width,
            expected_num_instances=expected_train_instances,
            expected_class_count=class_count,
        )
        logger.info("train dataset is ok")
    else:
        logger.info(
            f"skipping train dataset inspection because {gge_settings.EXPECTED_NUMBER_OF_TRAIN_INSTANCES.envvar} is not defined"
        )

    if expected_validation_instances is not None:
        logger.info("inspecting validation dataset")
        gge_settings.validate_dataset_dir(
            path=validation_dataset_dir,
            img_height=img_height,
            img_width=img_width,
            expected_num_instances=expected_validation_instances,
            expected_class_count=class_count,
        )
        logger.info("validation dataset is ok")
    else:
        logger.info(
            f"skipping validation dataset inspection because {gge_settings.EXPECTED_NUMBER_OF_TRAIN_INSTANCES.envvar} is not defined"
        )

    grammar = gr.Grammar(grammar_path.read_text())

    input_shape = gl.Shape(
        height=img_height,
        width=img_width,
        depth=img_depth,
    )

    fit_params = fit.FitnessEvaluationParameters(
        fit.ValidationAccuracy(
            train_directory=train_dataset_dir,
            validation_directory=validation_dataset_dir,
            input_shape=input_shape,
            batch_size=batch_size,
            max_epochs=epochs,
            class_count=class_count,
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
