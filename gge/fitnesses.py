"""
This module assumes that fitnesses must be maximized.
"""

import datetime as dt
import pathlib
import traceback
import typing

import attrs
import tensorflow as tf
import typeguard
from loguru import logger

import gge.composite_genotypes as cg
import gge.grammars as gr
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


def get_train_dataset(
    input_shape: gl.Shape,
    batch_size: int,
    directory: pathlib.Path,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` instance with images reshaped to match `input_shape`,
    organized in batches with size `batch_size`.
    The instances are reshuffled after every epoch.
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


def non_train_dataset(
    input_shape: gl.Shape,
    batch_size: int,
    directory: pathlib.Path,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` instance with images reshaped
    to match `input_shape` organized in batches with size `batch_size`.
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


@attrs.frozen
class ValidationAccuracy:
    train_directory: pathlib.Path
    validation_directory: pathlib.Path
    input_shape: gl.Shape
    class_count: int
    batch_size: int
    max_epochs: int
    early_stop_patience: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size > 0, self.batch_size
        assert self.max_epochs > 0, self.max_epochs
        assert self.class_count > 1, self.class_count
        assert self.early_stop_patience > 0
        assert self.train_directory.is_dir()
        assert self.validation_directory.is_dir()

        if self.train_directory == self.validation_directory:
            logger.warning("train_directory == validation_directory")

    def evaluate(self, phenotype: pheno.Phenotype) -> float:
        train = get_train_dataset(
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            directory=self.train_directory,
        )

        validation = non_train_dataset(
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
class FitnessEvaluationParameters:
    metric: ValidationAccuracy
    grammar: gr.Grammar


@attrs.frozen
class SuccessfulEvaluationResult:
    genotype: cg.CompositeGenotype
    fitness: float
    start_time: dt.datetime
    end_time: dt.datetime


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
        fitness = params.metric.evaluate(phenotype)
        logger.info(
            f"finished fitness evaluation of genotype=<{genotype}>, fitness={fitness}"
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


def effective_fitness(evaluation_result: FitnessEvaluationResult) -> float:
    """
    If `evaluation_result` is a `SuccessfulEvaluationResult`, returns its `fitness`;
    if it is as `FailedEvaluationResult`, returns negative infinity.

    This function assumes that fitness must be maximized.
    """

    match evaluation_result:
        case SuccessfulEvaluationResult():
            return evaluation_result.fitness

        case FailedEvaluationResult():
            return float("-inf")

        case _:
            raise ValueError(f"unknown fitness evaluation type={evaluation_result}")


def select_fittest(
    candidates: typing.Iterable[T],
    metric: typing.Callable[[T], float],
    fittest_count: int,
) -> list[T]:
    """
    This function assumes that fitness must be maximized.
    """
    assert fittest_count > 0

    candidates_list = list(candidates)
    assert len(candidates_list) >= fittest_count

    best_to_worst = sorted(
        candidates,
        key=lambda c: metric(c),
        reverse=True,
    )

    return best_to_worst[:fittest_count]


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

    train = get_train_dataset(
        input_shape,
        batch_size,
        train_dir,
    )

    validation = non_train_dataset(
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

    dataset = non_train_dataset(input_shape, batch_size, dataset_dir)

    loss, accuracy = model.evaluate(dataset)
    assert isinstance(accuracy, float)
    return accuracy
