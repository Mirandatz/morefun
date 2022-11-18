import functools
import pathlib
import sys
import typing

import attrs
import tensorflow as tf
import typeguard
import yaml
from loguru import logger

import gge.evolutionary.fitnesses as gf
import gge.evolutionary.mutations as gm
import gge.experiments.create_initial_population_genotypes as gge_init
import gge.grammars.upper_grammars as ugr
import gge.layers as gl
import gge.randomness as rand
import gge.redirection

YamlDict = dict[str, typing.Any]


@typeguard.typechecked()
@attrs.frozen
class ExperimentSettings:
    description: str
    rng_seed: int

    def __attrs_post_init__(self) -> None:
        # check if rng_seed is valid
        _ = rand.create_rng(self.rng_seed)

    @staticmethod
    def from_yaml(values: YamlDict) -> "ExperimentSettings":
        return ExperimentSettings(**values)


@typeguard.typechecked()
@attrs.frozen
class OutputSettings:
    log_level: str
    directory: pathlib.Path

    def __attrs_post_init__(self) -> None:
        assert self.log_level in [
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]

        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_yaml(values: YamlDict) -> "OutputSettings":
        copy_of_values = dict(values)
        directory = pathlib.Path(copy_of_values.pop("directory"))
        return OutputSettings(directory=directory, **copy_of_values)


@typeguard.typechecked()
@attrs.frozen
class InitializationSettings:
    population_size: int
    max_network_depth: int
    wide_layer_threshold: int
    max_wide_layers: int
    max_network_params: int

    def __attrs_post_init__(self) -> None:
        assert self.population_size >= 1
        assert self.max_network_depth >= 1
        assert self.wide_layer_threshold >= 1
        assert self.max_wide_layers >= 0
        assert self.max_network_params >= 1

    @staticmethod
    def from_yaml(values: YamlDict) -> "InitializationSettings":
        return InitializationSettings(**values)

    @property
    def individual_filter(self) -> gge_init.IndividualFilter:
        return gge_init.IndividualFilter(
            max_network_depth=self.max_network_depth,
            max_wide_layers=self.max_wide_layers,
            wide_layer_threshold=self.wide_layer_threshold,
            max_network_params=self.max_network_params,
        )


@typeguard.typechecked()
@attrs.frozen
class MutationSettings:
    mutants_per_generation: int
    max_failures_per_generation: int

    def __attrs_post_init__(self) -> None:
        assert self.mutants_per_generation >= 1
        assert self.max_failures_per_generation >= 0

    @staticmethod
    def from_yaml(values: YamlDict) -> "MutationSettings":
        return MutationSettings(**values)


@typeguard.typechecked()
@attrs.frozen
class FitnessSettings:
    batch_size: int
    max_epochs: int
    metrics: tuple[typing.Literal["train_loss", "number_of_parameters"], ...]
    early_stop_patience: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size >= 1
        assert self.max_epochs >= 1
        assert self.early_stop_patience >= 0
        assert len(set(self.metrics)) == len(self.metrics)

    @staticmethod
    def from_yaml(values: YamlDict) -> "FitnessSettings":
        return FitnessSettings(
            batch_size=values["batch_size"],
            max_epochs=values["max_epochs"],
            metrics=tuple(values["metrics"]),
            early_stop_patience=values["early_stop_patience"],
        )


@typeguard.typechecked()
@attrs.frozen
class DatasetSettings:
    partitions_dir: pathlib.Path
    image_height: int
    image_width: int
    image_depth: int
    class_count: int
    train_instances: int
    validation_instances: int
    test_instances: int

    def __attrs_post_init__(self) -> None:
        assert self.partitions_dir.is_dir()
        assert self.image_height >= 1
        assert self.image_width >= 1
        assert self.image_depth >= 1
        assert self.class_count >= 2
        assert self.train_instances >= 2
        assert self.validation_instances >= 0
        assert self.test_instances >= 2

    @staticmethod
    def from_yaml(values: YamlDict) -> "DatasetSettings":
        copy_of_values = dict(values)
        partitions_dir = pathlib.Path(copy_of_values.pop("partitions_dir"))

        return DatasetSettings(
            partitions_dir=partitions_dir,
            **copy_of_values,
        )

    @property
    def input_shape(self) -> gl.Shape:
        return gl.Shape(
            height=self.image_height,
            width=self.image_width,
            depth=self.image_depth,
        )

    @functools.cache
    def get_and_check_train_dir(self) -> pathlib.Path:
        train_dir = self.partitions_dir / "train"

        validate_dataset_dir(
            train_dir,
            img_height=self.image_height,
            img_width=self.image_width,
            expected_num_instances=self.train_instances,
            expected_class_count=self.class_count,
        )

        return train_dir

    @functools.cache
    def get_and_check_validation_dir(self) -> pathlib.Path:
        validation_dir = self.partitions_dir / "validation"

        validate_dataset_dir(
            validation_dir,
            img_height=self.image_height,
            img_width=self.image_width,
            expected_num_instances=self.validation_instances,
            expected_class_count=self.class_count,
        )

        return validation_dir

    @functools.cache
    def get_and_check_test_dir(self) -> pathlib.Path:
        test_dir = self.partitions_dir / "test"

        validate_dataset_dir(
            test_dir,
            img_height=self.image_height,
            img_width=self.image_width,
            expected_num_instances=self.test_instances,
            expected_class_count=self.class_count,
        )

        return test_dir


@typeguard.typechecked()
@attrs.frozen
class EvolutionSettings:
    mutation_settings: MutationSettings
    fitness_settings: FitnessSettings

    @staticmethod
    def from_yaml(values: YamlDict) -> "EvolutionSettings":
        assert values.keys() == {"mutation", "fitness_estimation"}
        return EvolutionSettings(
            MutationSettings.from_yaml(values["mutation"]),
            FitnessSettings.from_yaml(values["fitness_estimation"]),
        )


@attrs.frozen
class FinalTrainSettings:
    train_k_fittest: int
    batch_size: int
    max_epochs: int
    early_stop_patience: int

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.batch_size, int)
        assert isinstance(self.max_epochs, int)
        assert isinstance(self.early_stop_patience, int)

        assert self.batch_size >= 1
        assert self.max_epochs >= 1
        assert self.early_stop_patience >= 0

    @staticmethod
    def from_yaml(values: YamlDict) -> "FinalTrainSettings":

        return FinalTrainSettings(**values)


@typeguard.typechecked()
@attrs.frozen
class TensorflowSettings:
    xla: bool
    mixed_precision: bool

    @staticmethod
    def from_yaml(values: YamlDict) -> "TensorflowSettings":
        return TensorflowSettings(
            xla=values["xla"],
            mixed_precision=values["mixed_precision"],
        )


@typeguard.typechecked()
@attrs.frozen
class GgeSettings:
    experiment: ExperimentSettings
    dataset: DatasetSettings
    output: OutputSettings
    initialization: InitializationSettings
    evolution: EvolutionSettings
    final_train: FinalTrainSettings
    grammar: ugr.Grammar
    tensorflow: TensorflowSettings

    @staticmethod
    def from_yaml(values: YamlDict) -> "GgeSettings":
        assert values.keys() == {
            "experiment",
            "dataset",
            "output",
            "population_initialization",
            "evolution",
            "final_train",
            "tensorflow",
            "grammar",
        }

        return GgeSettings(
            experiment=ExperimentSettings.from_yaml(values["experiment"]),
            dataset=DatasetSettings.from_yaml(values["dataset"]),
            output=OutputSettings.from_yaml(values["output"]),
            initialization=InitializationSettings.from_yaml(
                values["population_initialization"]
            ),
            evolution=EvolutionSettings.from_yaml(values["evolution"]),
            final_train=FinalTrainSettings.from_yaml(values["final_train"]),
            tensorflow=TensorflowSettings.from_yaml(values["tensorflow"]),
            grammar=ugr.Grammar(values["grammar"]),
        )


def make_mutation_params(
    mutation: MutationSettings,
    grammar: ugr.Grammar,
) -> gm.PopulationMutationParameters:
    return gm.PopulationMutationParameters(
        mutants_to_generate=mutation.mutants_per_generation,
        max_failures=mutation.max_failures_per_generation,
        grammar=grammar,
    )


def make_metric(
    name: str,
    fitness: FitnessSettings,
    dataset: DatasetSettings,
) -> gf.Metric:
    match name:
        case "train_loss":
            return gf.TrainLoss(
                train_directory=dataset.get_and_check_train_dir(),
                input_shape=dataset.input_shape,
                batch_size=fitness.batch_size,
                max_epochs=fitness.max_epochs,
                class_count=dataset.class_count,
                early_stop_patience=fitness.early_stop_patience,
            )

        case "number_of_parameters":
            return gf.NumberOfParameters(dataset.input_shape, dataset.class_count)

        case _:
            raise ValueError(f"unknown metric name=<{name}>")


def make_metrics(
    dataset: DatasetSettings,
    fitness: FitnessSettings,
) -> tuple[gf.Metric, ...]:
    return tuple(make_metric(name, fitness, dataset) for name in fitness.metrics)


def load_gge_settings(path: pathlib.Path) -> GgeSettings:
    yaml_dict = yaml.safe_load(path.read_text())
    return GgeSettings.from_yaml(yaml_dict)


def configure_logger(settings: OutputSettings) -> None:
    logger.remove()
    logger.add(sink=sys.stderr, level=settings.log_level)
    logger.add(sink=settings.directory / "log.txt")

    logger.info("started logger")


def configure_tensorflow(settings: TensorflowSettings) -> None:
    if settings.xla:
        tf.config.optimizer.set_jit("autoclustering")
    else:
        logger.info("disabling xla")

    if settings.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    logger.info(
        f"tensorflow settings: xla=<{settings.xla}>, mixed_precision=<{settings.mixed_precision}>"
    )


def validate_dataset_dir(
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
