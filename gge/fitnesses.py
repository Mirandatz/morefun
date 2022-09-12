"""
This module assumes that fitnesses must be maximized.
"""

import abc
import datetime as dt
import pathlib
import traceback
import typing

import attrs
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import typeguard
from loguru import logger
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

import gge.composite_genotypes as cg
import gge.grammars as gr
import gge.layers as gl
import gge.phenotypes as pheno
import gge.randomness as rand
import gge.redirection as redirection


@typeguard.typechecked
@attrs.frozen
class Fitness:
    names: tuple[str, ...]
    values: tuple[float, ...]

    def __attrs_post_init__(self) -> None:
        assert len(self.names) == len(self.values)
        assert len(self.names) >= 1

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))

    def __str__(self) -> str:
        name_value_pairs = [
            f"{name}={value}" for name, value in zip(self.names, self.values)
        ]
        return ", ".join(name_value_pairs)


def make_classification_head(class_count: int, input_tensor: tf.Tensor) -> tf.Tensor:
    _, width, height, _ = input_tensor.shape

    conv = tf.keras.layers.Conv2D(
        filters=class_count,
        kernel_size=(width, height),
    )(input_tensor)

    global_pool = tf.keras.layers.GlobalMaxPooling2D()(conv)

    return tf.keras.layers.Activation(tf.nn.softmax)(global_pool)


def make_classification_model(
    phenotype: pheno.Phenotype,
    input_shape: gl.Shape,
    class_count: int,
) -> tf.keras.Model:
    input_tensor, output_tensor = pheno.make_input_output_tensors(
        phenotype, gl.Input(input_shape)
    )
    classification_head = make_classification_head(
        class_count,
        output_tensor,
    )
    model = tf.keras.Model(
        inputs=input_tensor,
        outputs=classification_head,
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=phenotype.optimizer.to_tensorflow(),
        metrics=["accuracy"],
    )
    return model


def load_dataset_and_rescale(
    directory: pathlib.Path,
    input_shape: gl.Shape,
) -> tf.data.Dataset:
    """
    Loads an image classification dataset and returns a `tf.data.Dataset` instance.

    Transformations:
    - resized to match `input_shape`
    - color values normalized to [0.0, 0.1] interval
    """

    match input_shape.depth:
        case 1:
            color_mode = "grayscale"
        case 3:
            color_mode = "rgb"
        case _:
            raise ValueError(
                f"unable to infer color_mode from input_shape=<{input_shape}>"
            )

    ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        batch_size=None,
        image_size=(input_shape.height, input_shape.width),
        label_mode="categorical",
        shuffle=False,
        color_mode=color_mode,
    )

    rescaling_layer = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255)])
    return ds.map(lambda d, t: (rescaling_layer(d, training=True), t))


def load_train_partition(
    input_shape: gl.Shape,
    batch_size: int,
    directory: pathlib.Path,
) -> tf.data.Dataset:
    """
    Loads an image classification dataset and returns a `tf.data.Dataset` instance.

    Transformations:
    - resized to match `input_shape`
    - color values normalized to [0.0, 0.1] interval
    - batched in `batch_size` batches, with remainder dropped
    - instances shuffled after each epoch
    """

    assert batch_size >= 1
    assert directory.is_dir()

    with redirection.discard_stderr_and_stdout():
        train = load_dataset_and_rescale(
            directory=directory,
            input_shape=input_shape,
        )

        return (
            train.cache()
            .shuffle(
                buffer_size=train.cardinality().numpy(),
                seed=rand.get_fixed_seed(),
                reshuffle_each_iteration=True,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )


def load_non_train_partition(
    input_shape: gl.Shape,
    batch_size: int,
    directory: pathlib.Path,
) -> tf.data.Dataset:
    """
    Loads an image classification dataset and returns a `tf.data.Dataset` instance.

    Transformations:
    - resized to match `input_shape`
    - color values normalized to [0.0, 0.1] interval
    - batched in `batch_size` batches, with remainder kept
    """

    with redirection.discard_stderr_and_stdout():
        return (
            load_dataset_and_rescale(
                directory=directory,
                input_shape=input_shape,
            )
            .cache()
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )


class FitnessMetric(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError("this is an abstract method")

    @abc.abstractmethod
    def should_be_maximized(self) -> bool:
        raise NotImplementedError("this is an abstract method")

    @abc.abstractmethod
    def evaluate(self, phenotype: pheno.Phenotype) -> float:
        """
        Fitnesses must be maximized, so metrics that must be minimized,
        such as 'number of parameters', may return (1 / value) or (-1 * value).
        """

        raise NotImplementedError("this is an abstract method")


@attrs.frozen
class ValidationAccuracy(FitnessMetric):
    train_directory: pathlib.Path
    validation_directory: pathlib.Path
    input_shape: gl.Shape
    class_count: int
    batch_size: int
    max_epochs: int
    early_stop_patience: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size >= 1, self.batch_size
        assert self.max_epochs >= 1, self.max_epochs
        assert self.class_count >= 2, self.class_count
        assert self.early_stop_patience >= 1
        assert self.train_directory.is_dir()
        assert self.validation_directory.is_dir()

        if self.train_directory == self.validation_directory:
            logger.warning("train_directory == validation_directory")

    def name(self) -> str:
        return "validation_accuracy"

    def evaluate(self, phenotype: pheno.Phenotype) -> float:
        train = load_train_partition(
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            directory=self.train_directory,
        )

        validation = load_non_train_partition(
            self.input_shape, self.batch_size, self.validation_directory
        )

        model = make_classification_model(
            phenotype,
            self.input_shape,
            self.class_count,
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=self.early_stop_patience,
            restore_best_weights=True,
        )

        with redirection.discard_stderr_and_stdout():
            fitting_result = model.fit(
                train,
                epochs=self.max_epochs,
                validation_data=validation,
                verbose=0,
                callbacks=[early_stop],
            )

        val_acc = max(fitting_result.history["val_accuracy"])
        assert isinstance(val_acc, float)

        return val_acc


@attrs.frozen
class NumberOfParameters(FitnessMetric):
    input_shape: gl.Shape
    class_count: int

    def __attrs_post_init__(self) -> None:
        assert self.class_count >= 2

    def name(self) -> str:
        return "number_of_parameters"

    def evaluate(self, phenotype: pheno.Phenotype) -> float:
        model = make_classification_model(phenotype, self.input_shape, self.class_count)
        param_count = model.count_params()
        assert isinstance(int, param_count)

        # fitnesses must be maximized and we want the smallest models
        return float(-1 * param_count)


@typeguard.typechecked
@attrs.frozen
class FitnessEvaluationParameters:
    metrics: tuple[FitnessMetric, ...]
    grammar: gr.Grammar

    def __attrs_post_init__(self) -> None:
        assert len(self.metrics) >= 1


@typeguard.typechecked
@attrs.frozen
class SuccessfulEvaluationResult:
    genotype: cg.CompositeGenotype
    fitness: Fitness
    start_time: dt.datetime
    end_time: dt.datetime

    def __attrs_post_init__(self) -> None:
        assert self.end_time >= self.start_time


@attrs.frozen
class FailedEvaluationResult:
    genotype: cg.CompositeGenotype
    description: str
    stacktrace: str
    start_time: dt.datetime
    end_time: dt.datetime


FitnessEvaluationResult = SuccessfulEvaluationResult | FailedEvaluationResult


def evaluate(
    genotype: cg.CompositeGenotype,
    params: FitnessEvaluationParameters,
) -> FitnessEvaluationResult:
    logger.info(f"starting fitness evaluation of genotype=<{genotype}>")

    start_time = dt.datetime.now()

    phenotype = pheno.translate(genotype, params.grammar)

    try:
        metrics_names = [m.name() for m in params.metrics]
        metrics_values = [m.evaluate(phenotype) for m in params.metrics]
        fitness = Fitness(
            names=tuple(metrics_names),
            values=tuple(metrics_values),
        )

        logger.info(
            f"finished fitness evaluation of genotype=<{genotype}>, fitness=<{fitness}>"
        )

        return SuccessfulEvaluationResult(
            genotype=genotype,
            fitness=fitness,
            start_time=start_time,
            end_time=dt.datetime.now(),
        )

    except tf.errors.ResourceExhaustedError:
        logger.warning(
            f"unable to evalute genotype due to resource exhaustion; genotype=<{genotype}>"
        )
        return FailedEvaluationResult(
            genotype=genotype,
            description="failed due to resource exhaustion",
            stacktrace=traceback.format_exc(),
            start_time=start_time,
            end_time=dt.datetime.now(),
        )

    except Exception as ex:
        logger.error(
            f"unable to evaluate genotype, a stacktrace was generated. error message=<{ex}>"
        )
        return FailedEvaluationResult(
            genotype=genotype,
            description="unknown error occured during fitness evaluation",
            stacktrace=traceback.format_exc(),
            start_time=start_time,
            end_time=dt.datetime.now(),
        )


T = typing.TypeVar("T")


def effective_fitness(
    evaluation_result: FitnessEvaluationResult,
    num_metrics: int,
) -> tuple[float, ...]:
    """
    If `evaluation_result` is a `SuccessfulEvaluationResult`, returns its `metrics_values`;
    if it is as `FailedEvaluationResult`, returns negative infinity.

    This function assumes that fitness must be maximized.
    """

    assert num_metrics >= 1

    match evaluation_result:
        case SuccessfulEvaluationResult():
            return evaluation_result.fitness.values

        case FailedEvaluationResult():
            failure_value = float("-inf")
            return tuple([failure_value] * num_metrics)

        case _:
            raise ValueError(f"unknown fitness evaluation type={evaluation_result}")


def fitness_evaluations_to_ndarray(
    fitness_evaluations: typing.Iterable[FitnessEvaluationResult],
) -> npt.NDArray[np.float64]:
    fers_copy = list(fitness_evaluations)

    successes = [
        fer for fer in fers_copy if isinstance(fer, SuccessfulEvaluationResult)
    ]

    if len(successes) == 0:
        raise ValueError(
            "fitness_evaluations must contain at least one `SuccessfulEvaluationResult`"
        )

    num_objectives = len(successes[0].fitness.names)
    effective_fitnesses = [effective_fitness(fer, num_objectives) for fer in fers_copy]
    return np.asarray(effective_fitnesses, dtype=np.float64)


def nsga2(
    fitnesses: npt.NDArray[np.float64],
    fittest_count: int,
) -> list[int]:
    """
    Returns a list of indices that would select the `fittest_count` elements
    of `fitnesses` using NSGA-II's fittest criteria.

    This function assumes that fitnesses must be maximized.
    """

    num_points, _ = fitnesses.shape

    assert num_points >= 1
    assert fittest_count >= 1
    assert fittest_count <= num_points

    fittest: list[int] = []

    # the functions `fast_non_dominated_sort` and `calc_crowding_distance`
    # assume that objective functions (our fitness) must be minimized,
    # so we must negate the values before using them
    negated_fitnesses = fitnesses * -1

    fronts = fast_non_dominated_sort(negated_fitnesses)

    for current_front in fronts:
        remaining_slots = fittest_count - len(fittest)

        if remaining_slots > len(current_front):
            fittest.extend(current_front)

        elif remaining_slots == len(current_front):
            fittest.extend(current_front)
            break

        else:
            front_fitnesses = negated_fitnesses[current_front]
            distances = calc_crowding_distance(
                front_fitnesses, filter_out_duplicates=False
            )
            indices_of_least_crowded = np.argsort(distances)
            sorted_by_crowding = [
                current_front[index] for index in indices_of_least_crowded
            ]
            least_crowded = sorted_by_crowding[:remaining_slots]
            fittest.extend(least_crowded)
            break

    # sanity check
    assert len(fittest) == fittest_count

    return fittest


@attrs.frozen(cache_hash=True, slots=True)
class ModelTrainingParameters:
    input_shape: gl.Shape
    batch_size: int
    max_epochs: int
    train_dir: pathlib.Path
    validation_dir: pathlib.Path
    early_stop_patience: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size >= 1, self.batch_size
        assert self.max_epochs >= 1, self.max_epochs
        assert self.early_stop_patience >= 0
        assert self.train_dir.is_dir()
        assert self.validation_dir.is_dir()

        if self.train_dir == self.validation_dir:
            logger.warning("train_directory == validation_directory")


@typeguard.typechecked()
@attrs.frozen(cache_hash=True, slots=True)
class TrainingHistory:
    train_losses: tuple[float, ...]
    train_accuracies: tuple[float, ...]
    val_losses: tuple[float, ...]
    val_accuracies: tuple[float, ...]

    def __attrs_post_init__(self) -> None:
        num_entries = len(self.train_losses)

        assert num_entries > 0
        assert len(self.train_accuracies) == num_entries
        assert len(self.val_losses) == num_entries
        assert len(self.val_accuracies) == num_entries

    @staticmethod
    def from_keras_history(history: dict[str, list[float]]) -> "TrainingHistory":
        assert history.keys() == {"loss", "accuracy", "val_loss", "val_accuracy"}

        return TrainingHistory(
            train_losses=tuple(history["loss"]),
            train_accuracies=tuple(history["accuracy"]),
            val_losses=tuple(history["val_loss"]),
            val_accuracies=tuple(history["val_accuracy"]),
        )


def train_model(
    model: tf.keras.Model,
    input_shape: gl.Shape,
    batch_size: int,
    max_epochs: int,
    early_stop_patience: int,
    train_dir: pathlib.Path,
    validation_dir: pathlib.Path,
) -> TrainingHistory:
    logger.info("starting model training")

    train = load_train_partition(
        input_shape,
        batch_size,
        train_dir,
    )

    validation = load_non_train_partition(
        input_shape,
        batch_size,
        validation_dir,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=early_stop_patience,
        restore_best_weights=True,
    )

    with redirection.discard_stderr_and_stdout():
        fitting_result = model.fit(
            train,
            epochs=max_epochs,
            validation_data=validation,
            verbose=0,
            callbacks=[early_stop],
        )
        keras_history = fitting_result.history
        assert isinstance(keras_history, dict)

    logger.info("finished model training")

    return TrainingHistory.from_keras_history(keras_history)


def compute_accuracy(
    model: tf.keras.Model,
    input_shape: gl.Shape,
    batch_size: int,
    dataset_dir: pathlib.Path,
) -> float:
    assert batch_size >= 1

    dataset = load_non_train_partition(input_shape, batch_size, dataset_dir)

    loss, accuracy = model.evaluate(dataset)
    assert isinstance(accuracy, float)
    return accuracy
