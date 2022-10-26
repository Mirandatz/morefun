import pathlib

from loguru import logger

import gge.experiments.settings as gset
import gge.fitnesses as gf
import gge.persistence
import gge.phenotypes

NUMBER_OF_RUNS_PER_DATASET = 5

RESULTS_DIR = pathlib.Path(__file__).parent / "gitignored"
CIFAR10_RESULTS = RESULTS_DIR / "cifar10_results"
CIFAR100_RESULTS = RESULTS_DIR / "cifar100_results"


def process_run_directory(run_directory: pathlib.Path) -> None:
    settings_path = run_directory / "settings.yaml"
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)
    gset.configure_tensorflow(settings.tensorflow)

    logger.info(f"processing: {run_directory}")

    output_path = run_directory / "output"
    last_generation_output = gge.persistence.load_latest_generational_artifacts(
        output_path
    )

    fittest = last_generation_output.fittest

    assert all(isinstance(fer, gf.SuccessfulEvaluationResult) for fer in fittest)

    best = max(fittest, key=lambda fer: fer.fitness.to_dict()["validation_accuracy"])
    genotype = best.genotype

    phenotype = gge.phenotypes.translate(genotype, settings.grammar)
    model = gf.make_classification_model(
        phenotype,
        input_shape=settings.dataset.input_shape,
        class_count=settings.dataset.class_count,
    )
    json = model.to_json()

    filename = (output_path / genotype.unique_id.hex).with_suffix(".json")
    filename.write_text(json)


def main() -> None:
    cifar10_dirs = list(CIFAR10_RESULTS.iterdir())
    assert len(cifar10_dirs) == NUMBER_OF_RUNS_PER_DATASET

    cifar100_dirs = list(CIFAR100_RESULTS.iterdir())
    assert len(cifar100_dirs) == NUMBER_OF_RUNS_PER_DATASET

    for run_directory in cifar10_dirs:
        process_run_directory(run_directory)

    for run_directory in cifar100_dirs:
        process_run_directory(run_directory)


if __name__ == "__main__":
    main()
