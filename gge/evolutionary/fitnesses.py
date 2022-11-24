"""
This module assumes that fitnesses must be MINIMIZED.
"""

import abc
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

import gge.neural_networks.layers as gl
import gge.phenotypes as pheno
import gge.randomness as rand
import gge.redirection as redirection


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
    tf.keras.backend.clear_session()
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


@typeguard.typechecked
@attrs.frozen
class SuccessfulMetricEvaluation:
    metric_name: str
    raw: float
    effective: float

    def __attrs_post_init__(self) -> None:
        assert self.metric_name


@typeguard.typechecked
@attrs.frozen
class FailedMetricEvaluation:
    metric_name: str
    effective: float
    description: str
    stacktrace: str

    def __attrs_post_init__(self) -> None:
        assert self.metric_name
        assert self.description
        assert self.stacktrace


MetricEvaluation = SuccessfulMetricEvaluation | FailedMetricEvaluation


class Metric(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError("this is an abstract method")

    @abc.abstractmethod
    def evaluate(self, phenotype: pheno.Phenotype) -> MetricEvaluation:
        raise NotImplementedError("this is an abstract method")


@typeguard.typechecked
@attrs.frozen
class NumberOfParameters(Metric):
    input_shape: gl.Shape
    class_count: int

    def __attrs_post_init__(self) -> None:
        assert self.class_count >= 2

    def name(self) -> str:
        return "NumberOfParameters"

    def _evaluate(self, phenotype: pheno.Phenotype) -> float:
        model = make_classification_model(phenotype, self.input_shape, self.class_count)
        param_count = model.count_params()
        assert isinstance(param_count, int)

        return param_count

    def evaluate(self, phenotype: pheno.Phenotype) -> MetricEvaluation:
        try:
            param_count = self._evaluate(phenotype)
            return SuccessfulMetricEvaluation(
                metric_name=self.name(),
                raw=param_count,
                effective=param_count,
            )

        except tf.errors.ResourceExhaustedError:
            msg = f"unable to evalute metric=<{self.name()}> of genotype=<{phenotype.genotype_uuid.hex}> due to resource exhaustion"
            logger.warning(msg)
            return FailedMetricEvaluation(
                metric_name=self.name(),
                effective=float("+inf"),
                description=msg,
                stacktrace=traceback.format_exc(),
            )

    def __call__(self, phenotype: pheno.Phenotype) -> MetricEvaluation:
        return self.evaluate(phenotype)


@typeguard.typechecked
@attrs.frozen
class TrainLoss(Metric):
    train_directory: pathlib.Path
    input_shape: gl.Shape
    class_count: int
    batch_size: int
    max_epochs: int
    early_stop_patience: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size >= 1
        assert self.max_epochs >= 1
        assert self.class_count >= 2
        assert self.early_stop_patience >= 1
        assert self.train_directory.is_dir()

    def name(self) -> str:
        return "TrainLoss"

    def _evaluate(self, phenotype: pheno.Phenotype) -> float:
        train = load_train_partition(
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            directory=self.train_directory,
        )

        model = make_classification_model(
            phenotype,
            self.input_shape,
            self.class_count,
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=self.early_stop_patience,
            restore_best_weights=False,
            monitor="loss",
        )

        with redirection.discard_stderr_and_stdout():
            train_history = model.fit(
                train,
                epochs=self.max_epochs,
                verbose=0,
                callbacks=[early_stop],
            )

        train_loss = min(train_history.history["loss"])
        assert isinstance(train_loss, float)

        return train_loss

    def evaluate(self, phenotype: pheno.Phenotype) -> MetricEvaluation:
        try:
            train_loss = self._evaluate(phenotype)
            return SuccessfulMetricEvaluation(
                metric_name=self.name(),
                raw=train_loss,
                effective=train_loss,
            )

        except tf.errors.ResourceExhaustedError:
            msg = f"unable to evalute metric=<{self.name()}> of genotype=<{phenotype.genotype_uuid.hex}> due to resource exhaustion"
            logger.warning(msg)
            return FailedMetricEvaluation(
                metric_name=self.name(),
                effective=float("+inf"),
                description=msg,
                stacktrace=traceback.format_exc(),
            )

    def __call__(self, phenotype: pheno.Phenotype) -> MetricEvaluation:
        return self.evaluate(phenotype)


@attrs.frozen
class Fitness:
    metric_evaluations: tuple[MetricEvaluation, ...]

    @typeguard.typechecked
    def __init__(self, metric_evaluations: tuple[MetricEvaluation, ...]) -> None:
        assert len(metric_evaluations) >= 1
        object.__setattr__(self, "metric_evaluations", metric_evaluations)

    def metric_names(self) -> tuple[str, ...]:
        return tuple(m.metric_name for m in self.metric_evaluations)

    def to_effective_fitnesses_dict(self) -> dict[str, float]:
        return dict(
            zip(
                self.metric_names(),
                self.effective_values(),
            )
        )

    def effective_values(self) -> tuple[float, ...]:
        return tuple(m.effective for m in self.metric_evaluations)

    def effective_values_as_ndarray(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.effective_values())

    def successfully_evaluated(self) -> bool:
        """
        Returns true if all metrics were successfully evaluated, false otherwise.
        """
        return all(
            isinstance(me, SuccessfulMetricEvaluation) for me in self.metric_evaluations
        )


def evaluate(
    phenotype: pheno.Phenotype,
    metrics: typing.Iterable[Metric],
) -> Fitness:
    logger.info(f"starting fitness evaluation of genotype=<{phenotype}>")

    metric_evals = (metric.evaluate(phenotype) for metric in metrics)
    fitness = Fitness(tuple(metric_evals))

    logger.info(
        f"finished fitness evaluation of genotype=<{phenotype}>, fitness=<{fitness}>"
    )

    return fitness


def fitnesses_to_ndarray(
    fitnesses: typing.Iterable[Fitness],
) -> npt.NDArray[np.float64]:
    """
    Returns a vstacked ndarray of the effective values of `fitnesses`.
    Requires that all fitnesses have the same metrics.
    """

    as_list = list(fitnesses)
    if len(as_list) == 0:
        raise ValueError("fitnesses must can not be empty")

    metric_names = (f.metric_names() for f in as_list)
    if len(set(metric_names)) != 1:
        raise ValueError("fitnesses must have the same metrics")

    arrays = [f.effective_values_as_ndarray() for f in fitnesses]
    return np.vstack(arrays)


def argsort_nsga2(
    fitnesses: npt.NDArray[np.float64],
    fittest_count: int,
) -> list[int]:
    """
    Returns a list of indices that would select the `fittest_count` elements
    of `fitnesses` using NSGA-II's fittest criteria.

    This function assumes that fitnesses must be minimized.
    """

    num_points, _ = fitnesses.shape

    assert num_points >= 1
    assert fittest_count >= 1
    assert fittest_count <= num_points

    fittest: list[int] = []

    fronts = fast_non_dominated_sort(fitnesses)

    for current_front in fronts:
        remaining_slots = fittest_count - len(fittest)

        if remaining_slots > len(current_front):
            fittest.extend(current_front)

        elif remaining_slots == len(current_front):
            fittest.extend(current_front)
            break

        else:
            front_fitnesses = fitnesses[current_front]
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


T = typing.TypeVar("T")


def select_fittest_nsga2(
    evaluated_solutions: dict[T, Fitness],
    fittest_count: int,
) -> list[T]:
    if fittest_count > len(evaluated_solutions):
        raise ValueError("fittest_count must be <= len(fitnesses)")

    indexed = list(evaluated_solutions.items())
    solutions, fitnesses = zip(*indexed)

    fitnesses_array = fitnesses_to_ndarray(fitnesses)
    fittest_indices = argsort_nsga2(fitnesses_array, fittest_count)

    return [solutions[index] for index in fittest_indices]
