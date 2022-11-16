"""
This module assumes that fitnesses must be MINIMIZED.
"""

import abc  # noqa
import pathlib
import traceback  # noqa
import typing  # noqa

import attrs
import numpy as np  # noqa
import numpy.typing as npt  # noqa
import tensorflow as tf
import typeguard
from loguru import logger  # noqa
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance  # noqa
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort  # noqa

import gge.composite_genotypes as cg  # noqa
import gge.grammars as gr  # noqa
import gge.layers as gl
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
Metric = typing.Callable[[pheno.Phenotype], MetricEvaluation]


@typeguard.typechecked
@attrs.frozen
class NumberOfParameters:
    _NAME = "NumberOfParameters"

    input_shape: gl.Shape
    class_count: int

    def __attrs_post_init__(self) -> None:
        assert self.class_count >= 2

    def _evaluate(self, phenotype: pheno.Phenotype) -> float:
        model = make_classification_model(phenotype, self.input_shape, self.class_count)
        param_count = model.count_params()
        assert isinstance(param_count, int)

        return param_count

    def evaluate(self, phenotype: pheno.Phenotype) -> MetricEvaluation:
        try:
            param_count = self._evaluate(phenotype)
            return SuccessfulMetricEvaluation(
                metric_name=self._NAME,
                raw=param_count,
                effective=param_count,
            )

        except tf.errors.ResourceExhaustedError:
            msg = f"unable to evalute metric=<{self._NAME}> of genotype=<{phenotype.genotype_uuid.hex}> due to resource exhaustion"
            logger.warning(msg)
            return FailedMetricEvaluation(
                metric_name=self._NAME,
                effective=float("+inf"),
                description=msg,
                stacktrace=traceback.format_exc(),
            )


@typeguard.typechecked
@attrs.frozen
class TrainLoss:
    _NAME = "TrainLoss"

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
                metric_name=self._NAME,
                raw=train_loss,
                effective=train_loss,
            )

        except tf.errors.ResourceExhaustedError:
            msg = f"unable to evalute metric=<{self._NAME}> of genotype=<{phenotype.genotype_uuid.hex}> due to resource exhaustion"
            logger.warning(msg)
            return FailedMetricEvaluation(
                metric_name=self._NAME,
                effective=float("+inf"),
                description=msg,
                stacktrace=traceback.format_exc(),
            )


@typeguard.typechecked
@attrs.frozen
class Fitness:
    """
    Represents a collection of `MetricEvaluation`s sorted in asceding order by `metric_name`.
    """

    metric_evaluations: tuple[MetricEvaluation, ...]

    def __attrs_post_init__(self) -> None:
        assert len(self.metric_evaluations) >= 1

        names = [m.metric_name for m in self.metric_evaluations]
        assert sorted(names) == names

    @staticmethod
    def from_unsorted_metric_evaluations(
        metric_evaluations: typing.Iterable[MetricEvaluation],
    ) -> "Fitness":
        return Fitness(tuple(sorted(metric_evaluations, key=lambda m: m.metric_name)))

    def metric_names(self) -> tuple[str, ...]:
        return tuple(m.metric_name for m in self.metric_evaluations)

    def effective_values(self) -> tuple[float, ...]:
        return tuple(m.effective for m in self.metric_evaluations)

    def effective_values_as_ndarray(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.effective_values())


def evaluate(
    phenotype: pheno.Phenotype,
    metrics: typing.Iterable[Metric],
) -> Fitness:
    logger.info(f"starting fitness evaluation of genotype=<{phenotype}>")

    metric_evals = (metric(phenotype) for metric in metrics)
    fitness = Fitness(tuple(metric_evals))

    logger.info(
        f"finished fitness evaluation of genotype=<{phenotype}>, fitness=<{fitness}>"
    )

    return fitness


def fitnesses_to_ndarray(
    fitnesses: typing.Iterable[Fitness],
) -> npt.NDArray[np.float64]:
    as_list = list(fitnesses)
    assert len(as_list) >= 1

    metric_names = [f.metric_names() for f in as_list]
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

    This function assumes that fitnesses must be maximized.
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


def select_fittest_nsga2(
    fitnesses: typing.Iterable[Fitness],
    fittest_count: int,
) -> list[Fitness]:
    as_list = list(fitnesses)

    if fittest_count > len(as_list):
        raise ValueError("fittest_count must be <= len(fitnesses)")

    as_array = fitnesses_to_ndarray(as_list)
    fittest_indices = argsort_nsga2(as_array, fittest_count)
    return [as_list[index] for index in fittest_indices]
