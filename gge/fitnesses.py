import pathlib
import typing

import attrs
import tensorflow as tf
from loguru import logger

import gge.composite_genotypes as cg
import gge.data_augmentations as gda
import gge.debugging as debug
import gge.grammars as gr
import gge.layers as gl
import gge.phenotypes as phenos
import gge.redirection as redirection

DataGen: typing.TypeAlias = tf.keras.preprocessing.image.DirectoryIterator


def make_classification_head(class_count: int, input_tensor: tf.Tensor) -> tf.Tensor:
    _, width, height, _ = input_tensor.shape

    conv = tf.keras.layers.Conv2D(
        filters=class_count,
        kernel_size=(width, height),
    )(input_tensor)

    global_pool = tf.keras.layers.GlobalMaxPooling2D()(conv)

    return tf.keras.layers.Activation(tf.nn.softmax)(global_pool)


def make_classification_model(
    phenotype: phenos.Phenotype,
    input_shape: gl.Shape,
    class_count: int,
) -> tf.keras.Model:
    input_tensor, output_tensor = phenos.make_input_output_tensors(
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


@attrs.frozen
class ValidationAccuracy:
    train_directory: pathlib.Path
    validation_directory: pathlib.Path
    data_augmentation: gda.DataAugmentation
    input_shape: gl.Shape
    batch_size: int
    max_epochs: int
    class_count: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size > 0, self.batch_size
        assert self.max_epochs > 0, self.max_epochs
        assert self.class_count > 1, self.class_count
        assert self.train_directory.is_dir(), self.train_directory
        assert self.validation_directory.is_dir(), self.validation_directory
        assert self.train_directory != self.validation_directory

    def get_train_generator(self) -> DataGen:
        with redirection.discard_stderr_and_stdout():
            data_generator = self.data_augmentation.to_tensorflow_data_generator()
            return data_generator.flow_from_directory(
                directory=self.train_directory,
                batch_size=self.batch_size,
                target_size=(self.input_shape.width, self.input_shape.height),
                shuffle=True,
                seed=0,
            )

    def get_validation_generator(self) -> DataGen:
        with redirection.discard_stderr_and_stdout():
            return (
                tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
                    directory=self.validation_directory,
                    batch_size=self.batch_size,
                    target_size=(self.input_shape.width, self.input_shape.height),
                )
            )

    def evaluate(self, phenotype: phenos.Phenotype) -> float:
        logger.debug(f"starting fitness evaluation, phenotype=<{phenotype}>")

        train_data = self.get_train_generator()
        val_data = self.get_validation_generator()
        model = make_classification_model(
            phenotype,
            self.input_shape,
            self.class_count,
        )

        with redirection.discard_stderr_and_stdout():
            fitting_result = model.fit(
                train_data,
                epochs=self.max_epochs,
                steps_per_epoch=train_data.samples // self.batch_size,
                validation_data=val_data,
                validation_steps=val_data.samples // self.batch_size,
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


def evaluate(
    genotype: cg.CompositeGenotype,
    params: FitnessEvaluationParameters,
) -> float:
    logger.debug(f"starting fitness evaluation of genotype=<{genotype}>")

    phenotype = phenos.translate(genotype, params.grammar)

    try:
        fitness = params.metric.evaluate(phenotype)
        logger.debug(f"finished fitness evaluation of genotype=<{genotype}>")
        return fitness

    except tf.errors.ResourceExhaustedError:
        logger.warning(
            f"unable to evalute genotype due to resource exhaustion; genotype=<{genotype}>"
        )
        return float("-inf")

    except (ValueError, tf.errors.InvalidArgumentError):
        filename = debug.save_genotype(genotype)
        logger.error(
            f"unable to evaluate genotype because the phenotype is malformed; saved as=<{filename}>"
        )
        raise


T = typing.TypeVar("T")


def select_fittest(
    population: dict[T, float],
    fittest_count: int,
) -> dict[T, float]:
    """
    This function assumes that fitness must be maximized.
    """
    assert len(population) >= fittest_count
    assert fittest_count > 0

    logger.debug("starting fittest selection")

    best_to_worst = sorted(
        population.keys(),
        key=lambda g: population[g],
        reverse=True,
    )

    fittest = best_to_worst[:fittest_count]
    assert len(fittest) == fittest_count

    keyed_fittest = {g: population[g] for g in fittest}

    logger.debug("finished fittest selection")
    return keyed_fittest
