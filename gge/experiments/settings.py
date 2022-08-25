import pathlib
import sys
import typing

import tomli
from loguru import logger

import gge.experiments.create_initial_population_genotypes as gge_init
import gge.fitnesses as cf
import gge.grammars as gr
import gge.layers as gl
import gge.population_mutation
import gge.redirection

Settings = dict[str, typing.Any]


def load_settings(path: pathlib.Path) -> Settings:
    with path.open("rb") as file:
        return tomli.load(file)


def load_settings_and_configure_logger(path: pathlib.Path) -> Settings:
    settings = load_settings(path)
    configure_logger(settings)
    return settings


def get_base_output_dir(settings: Settings) -> pathlib.Path:
    directory = pathlib.Path(settings["output"]["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_logging_dir(settings: Settings) -> pathlib.Path:
    directory = get_base_output_dir(settings) / "logging"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logger(settings: Settings) -> None:
    log_settings = settings["logging"]
    log_level = log_settings["log_level"]

    if log_level not in ["DEBUG", "INFO", "WARNING"]:
        raise ValueError(f"unknown log_level=<{log_level}>")

    logging_dir = get_logging_dir(settings)

    logger.remove()
    logger.add(sink=sys.stderr, level=log_level)
    logger.add(sink=logging_dir / "log.txt")

    logger.info("started logger")


def get_grammar(settings: Settings) -> gr.Grammar:
    return gr.Grammar(settings["grammar"]["raw"])


def get_train_dataset_dir(settings: Settings) -> pathlib.Path:
    dataset_settings = settings["datasets"]

    shape = get_dataset_input_shape(settings)
    img_height = shape.height
    img_width = shape.width

    expected_class_count = get_dataset_class_count(settings)

    partitions_dir = pathlib.Path(dataset_settings["partitions_dir"])
    train_dir = partitions_dir / "train"

    expected_num_instances = dataset_settings["train_instances"]

    _validate_dataset_dir(
        train_dir,
        img_height=img_height,
        img_width=img_width,
        expected_num_instances=expected_num_instances,
        expected_class_count=expected_class_count,
    )

    return train_dir


def get_validation_dataset_dir(settings: Settings) -> pathlib.Path:
    dataset_settings = settings["datasets"]

    shape = get_dataset_input_shape(settings)
    img_height = shape.height
    img_width = shape.width

    expected_class_count = get_dataset_class_count(settings)

    partitions_dir = pathlib.Path(dataset_settings["partitions_dir"])
    validation_dir = partitions_dir / "validation"

    expected_num_instances = dataset_settings["validation_instances"]

    _validate_dataset_dir(
        validation_dir,
        img_height=img_height,
        img_width=img_width,
        expected_num_instances=expected_num_instances,
        expected_class_count=expected_class_count,
    )

    return validation_dir


def get_dataset_input_shape(settings: Settings) -> gl.Shape:
    return gl.Shape(
        height=settings["datasets"]["image_height"],
        width=settings["datasets"]["image_width"],
        depth=settings["datasets"]["image_depth"],
    )


def get_batch_size(settings: Settings) -> int:
    value = settings["fitness"]["batch_size"]
    assert isinstance(value, int)
    return value


def get_max_epochs(settings: Settings) -> int:
    value = settings["fitness"]["epochs"]
    assert isinstance(value, int)
    return value


def get_dataset_class_count(settings: Settings) -> int:
    value = settings["datasets"]["class_count"]
    assert isinstance(value, int)
    return value


def get_fitness_metric(settings: Settings) -> cf.ValidationAccuracy:
    return cf.ValidationAccuracy(
        train_directory=get_train_dataset_dir(settings),
        validation_directory=get_validation_dataset_dir(settings),
        input_shape=get_dataset_input_shape(settings),
        batch_size=get_batch_size(settings),
        max_epochs=get_max_epochs(settings),
        class_count=get_dataset_class_count(settings),
    )


def get_evolutionary_fitness_evaluation_params(
    settings: Settings,
) -> cf.FitnessEvaluationParameters:
    final_train_settings = settings["final_train"]
    batch_size = int(final_train_settings["batch_size"])
    max_epochs = int(final_train_settings["epochs"])
    test_dir = pathlib.Path(final_train_settings["test_dir"])

    metric = cf.TestAccuracy(
        train_directory=get_train_dataset_dir(settings),
        validation_directory=get_validation_dataset_dir(settings),
        test_dir=test_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        input_shape=get_dataset_input_shape(settings),
        class_count=get_dataset_class_count(settings),
    )

    return cf.FitnessEvaluationParameters(
        metric=metric,
        grammar=get_grammar(settings),
    )


def get_final_performance_evaluation_params(
    settings: Settings,
) -> cf.FitnessEvaluationParameters:
    return cf.FitnessEvaluationParameters(
        get_fitness_metric(settings),
        get_grammar(settings),
    )


def get_nr_of_mutants_to_create_per_generation(settings: Settings) -> int:
    value = settings["mutation"]["mutants_per_generation"]
    assert isinstance(value, int)
    return value


def get_max_mutation_failures_per_generation(settings: Settings) -> int:
    value = settings["mutation"]["max_failures_per_generation"]
    assert isinstance(value, int)
    return value


def get_mutation_params(
    settings: Settings,
) -> gge.population_mutation.PopulationMutationParameters:
    return gge.population_mutation.PopulationMutationParameters(
        mutants_to_generate=get_nr_of_mutants_to_create_per_generation(settings),
        max_failures=get_max_mutation_failures_per_generation(settings),
        grammar=get_grammar(settings),
    )


def get_rng_seed(settings: Settings) -> int:
    value = settings["experiment"]["rng_seed"]
    assert isinstance(value, int)
    return value


def get_initial_population_size(settings: Settings) -> int:
    value = settings["initialization"]["population_size"]
    assert isinstance(value, int)
    return value


def get_initial_population_max_network_depth(settings: Settings) -> int:
    value = settings["initialization"]["max_network_depth"]
    assert isinstance(value, int)
    return value


def get_initial_population_max_wide_layers(settings: Settings) -> int:
    value = settings["initialization"]["max_wide_layers"]
    assert isinstance(value, int)
    return value


def get_initial_population_wide_layer_threshold(settings: Settings) -> int:
    value = settings["initialization"]["wide_layer_threshold"]
    assert isinstance(value, int)
    return value


def get_initial_population_max_network_params(settings: Settings) -> int:
    value = settings["initialization"]["max_network_params"]
    assert isinstance(value, int)
    return value


def get_initialization_individual_filter(
    settings: Settings,
) -> gge_init.IndividualFilter:
    return gge_init.IndividualFilter(
        max_network_depth=get_initial_population_max_network_depth(settings),
        max_wide_layers=get_initial_population_max_wide_layers(settings),
        max_layer_width=get_initial_population_wide_layer_threshold(settings),
        max_network_params=get_initial_population_max_network_params(settings),
    )


def _validate_dataset_dir(
    path: pathlib.Path,
    img_height: int,
    img_width: int,
    expected_num_instances: int,
    expected_class_count: int,
) -> None:
    import tensorflow as tf

    with gge.redirection.discard_stderr_and_stdout():
        with tf.device("cpu"):
            ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
                directory=path,
                batch_size=None,
                image_size=(img_height, img_width),
                label_mode="categorical",
                shuffle=False,
                color_mode="rgb",
            )

            num_instances = ds.cardinality().numpy()
            if num_instances != expected_num_instances:
                raise ValueError(
                    f"unexpected number of instances. found=<{num_instances}>, expected=<{expected_num_instances}>"
                )

            num_classes = len(ds.class_names)
            if num_classes != expected_class_count:
                raise ValueError(
                    f"unexpected number of classes. found=<{num_classes}>, expected=<{expected_class_count}>"
                )
