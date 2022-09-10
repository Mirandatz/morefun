import heapq
import pathlib

import pandas as pd
import tensorflow as tf
import typer

import gge.composite_genotypes as cg
import gge.evolution
import gge.experiments.create_initial_population_genotypes as gge_init
import gge.experiments.settings as gset
import gge.fitnesses as gf
import gge.novelty
import gge.persistence
import gge.phenotypes
import gge.randomness

SETTINGS_OPTION = typer.Option(
    pathlib.Path("/gge/settings.toml"),
    "--settings-path",
    file_okay=True,
    exists=True,
    readable=True,
    dir_okay=False,
)


app = typer.Typer()


@app.command(name="initialize")
def create_and_evaluate_initial_population(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)

    rng_seed = settings.experiment.rng_seed

    individuals = gge_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    genotypes = [ind.genotype for ind in individuals]
    phenotypes = [ind.phenotype for ind in individuals]

    fitness_evaluation_params = gset.make_fitness_evaluation_params(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        grammar=settings.grammar,
    )

    fitness_evaluation_results = [
        gf.evaluate(g, fitness_evaluation_params) for g in genotypes
    ]

    known_genotypes = set(genotypes)
    known_phenotypes = set(phenotypes)
    novelty_tracker = gge.novelty.NoveltyTracker(
        known_genotypes=known_genotypes,
        known_phenotypes=known_phenotypes,
    )

    # generations are 0-indexed, so first gen == 0
    gge.persistence.save_generational_artifacts(
        generation_number=0,
        fittest=fitness_evaluation_results,
        novelty_tracker=novelty_tracker,
        rng=gge.randomness.create_rng(rng_seed),
        output_dir=settings.output.directory,
    )


@app.command(name="evolve")
def evolutionary_loop(
    settings_path: pathlib.Path = SETTINGS_OPTION,
    generations: int = typer.Option(..., "--generations", min=1),
) -> None:
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)

    latest_gen_output = gge.persistence.load_latest_generational_artifacts(
        settings.output.directory
    )

    current_generation_number = latest_gen_output.generation_number + 1

    mutation_params = gset.make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    fitness_params = gset.make_fitness_evaluation_params(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        grammar=settings.grammar,
    )

    gge.evolution.run_evolutionary_loop(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_gen_output.fittest,
        mutation_params=mutation_params,
        fitness_params=fitness_params,
        novelty_tracker=latest_gen_output.novelty_tracker,
        rng=latest_gen_output.rng,
        output_dir=settings.output.directory,
    )


def _final_train_single_genotype(
    genotype: cg.CompositeGenotype,
    settings: gset.GgeSettings,
) -> None:
    phenotype = gge.phenotypes.translate(genotype, settings.grammar)
    model = gf.make_classification_model(
        phenotype,
        input_shape=settings.dataset.input_shape,
        class_count=settings.dataset.class_count,
    )

    training_history = gf.train_model(
        model,
        input_shape=settings.dataset.input_shape,
        batch_size=settings.final_train.batch_size,
        max_epochs=settings.final_train.max_epochs,
        early_stop_patience=settings.final_train.early_stop_patience,
        train_dir=settings.dataset.get_and_check_train_dir(),
        validation_dir=settings.dataset.get_and_check_validation_dir(),
    )

    gge.persistence.save_genotype(
        genotype,
        gge.persistence.get_genotype_path(
            genotype,
            settings.output.directory,
        ),
    )

    genotype_uuid_hex = genotype.unique_id.hex

    gge.persistence.save_and_zip_tf_model(
        model,
        gge.persistence.get_model_path(
            genotype_uuid_hex,
            settings.output.directory,
        ),
    )

    gge.persistence.save_training_history(
        training_history,
        gge.persistence.get_training_history_path(
            genotype_uuid_hex,
            settings.output.directory,
        ),
    )


@app.command(name="final-train")
def final_train(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)

    latest_gen_output = gge.persistence.load_latest_generational_artifacts(
        settings.output.directory
    )

    fittest = heapq.nlargest(
        n=settings.final_train.train_k_fittest,
        iterable=latest_gen_output.fittest,
        key=lambda fer: gf.effective_fitness(fer),
    )

    assert all(isinstance(fer, gf.SuccessfulEvaluationResult) for fer in fittest)

    genotypes = [fer.genotype for fer in fittest]

    for genotype in genotypes:
        _final_train_single_genotype(genotype, settings)


def _evaluate_model(
    genotype_uuid_hex: str,
    model: tf.keras.Model,
    settings: gset.GgeSettings,
) -> dict[str, float | int | str]:
    train_accuracy = model.evaluate(
        gf.load_non_train_partition(
            settings.dataset.input_shape,
            settings.final_train.batch_size,
            settings.dataset.get_and_check_train_dir(),
        )
    )

    val_accuracy = model.evaluate(
        gf.load_non_train_partition(
            settings.dataset.input_shape,
            settings.final_train.batch_size,
            settings.dataset.get_and_check_validation_dir(),
        )
    )

    test_accuracy = model.evaluate(
        gf.load_non_train_partition(
            settings.dataset.input_shape,
            settings.final_train.batch_size,
            settings.dataset.get_and_check_test_dir(),
        )
    )

    num_params: int = model.count_params()

    return {
        "rng_seed": settings.experiment.rng_seed,
        "genotype_uuid_hex": genotype_uuid_hex,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "num_params": num_params,
    }


@app.command(name="evaluate-models")
def evaluate_models(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)

    evaluations = []

    for genotype_uuid_hex, model in gge.persistence.load_models(
        settings.output.directory
    ):
        model_eval = _evaluate_model(genotype_uuid_hex, model, settings)
        evaluations.append(model_eval)

    df = pd.DataFrame(evaluations)
    save_path = gge.persistence.get_models_evaluations_path(settings.output.directory)
    df.to_csv(save_path)


if __name__ == "__main__":
    app()
