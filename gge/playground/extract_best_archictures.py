import pathlib

from loguru import logger

import gge.experiments.settings as gset
import gge.fitnesses as gf
import gge.persistence
import gge.phenotypes

NUMBER_OF_RUNS_PER_DATASET = 5

RESULTS_DIR = pathlib.Path(__file__).parent / "gitignored"

CIFAR10_RESULTS = RESULTS_DIR / "cifar10_results"
LAST_DITCH_CIFAR10_RESULTS = RESULTS_DIR / "last_ditch_cifar10"

CIFAR100_RESULTS = RESULTS_DIR / "cifar100_results"
LAST_DITCH_CIFAR100_RESULTS = RESULTS_DIR / "last_ditch_cifar100"


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

    fittest = [
        fer
        for fer in last_generation_output.fittest
        if isinstance(fer, gf.SuccessfulEvaluationResult)
    ]
    assert len(fittest) >= 1

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
    for run_directory in CIFAR10_RESULTS.iterdir():
        process_run_directory(run_directory)

    for run_directory in LAST_DITCH_CIFAR10_RESULTS.iterdir():
        process_run_directory(run_directory)

    for run_directory in CIFAR100_RESULTS.iterdir():
        process_run_directory(run_directory)

    for run_directory in LAST_DITCH_CIFAR100_RESULTS.iterdir():
        process_run_directory(run_directory)


if __name__ == "__main__":
    main()
