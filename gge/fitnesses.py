"""
This module assumes that fitnesses must be maximized.
"""

import datetime as dt
import pathlib
import traceback
import typing

import attrs
import tensorflow as tf
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


@attrs.frozen
class ValidationAccuracy:
    train_directory: pathlib.Path
    validation_directory: pathlib.Path
    input_shape: gl.Shape
    batch_size: int
    max_epochs: int
    class_count: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size > 0, self.batch_size
        assert self.max_epochs > 0, self.max_epochs
        assert self.class_count > 1, self.class_count
        assert self.train_directory.is_dir()
        assert self.validation_directory.is_dir()

    def get_train_dataset(self) -> tf.data.Dataset:
        with redirection.discard_stderr_and_stdout():
            train = load_dataset_and_rescale(
                directory=self.train_directory,
                input_shape=self.input_shape,
            )

        return (
            train.cache()
            .shuffle(
                buffer_size=train.cardinality().numpy(),
                seed=rand.get_fixed_seed(),
                reshuffle_each_iteration=True,
            )
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    def get_validation_dataset(self) -> tf.data.Dataset:
        with redirection.discard_stderr_and_stdout():
            return (
                load_dataset_and_rescale(
                    directory=self.validation_directory,
                    input_shape=self.input_shape,
                )
                .cache()
                .batch(self.batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )

    def evaluate(self, phenotype: pheno.Phenotype) -> float:
        logger.debug(f"starting fitness evaluation, phenotype=<{phenotype}>")

        train = self.get_train_dataset()
        validation = self.get_validation_dataset()
        model = make_classification_model(
            phenotype,
            self.input_shape,
            self.class_count,
        )

        with redirection.discard_stderr_and_stdout():
            fitting_result = model.fit(
                train,
                epochs=self.max_epochs,
                validation_data=validation,
                verbose=0,
            )

        val_acc = max(fitting_result.history["val_accuracy"])
        assert isinstance(val_acc, float)

        logger.debug(
            f"finished fitness evaluation, phenotype=<{phenotype}>, accuracy=<{val_acc}>"
        )

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
        logger.info(f"finished fitness evaluation of genotype=<{genotype}>")
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


def get_effective_fitness(evaluation_result: FitnessEvaluationResult) -> float:
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
