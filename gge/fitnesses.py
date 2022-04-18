import pathlib
import typing

import attrs
import keras
import keras.layers as kl
import tensorflow as tf
from loguru import logger

import gge.composite_genotypes as cg
import gge.debugging as debug
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn
import gge.redirection as redirection

DataGen: typing.TypeAlias = keras.preprocessing.image.DirectoryIterator


def make_classification_head(class_count: int, input_tensor: tf.Tensor) -> tf.Tensor:
    _, width, height, _ = input_tensor.shape

    conv = kl.Conv2D(
        filters=class_count,
        kernel_size=(width, height),
    )(input_tensor)

    global_pool = kl.GlobalMaxPooling2D()(conv)

    return kl.Activation(tf.nn.softmax)(global_pool)


def make_tf_model(
    network_skeleton: gnn.NeuralNetwork,
    class_count: int,
) -> keras.Model:
    input_tensor, output_tensor = network_skeleton.to_input_output_tensor()
    classification_head = make_classification_head(class_count, output_tensor)
    optimizer = network_skeleton.optimizer.to_keras()
    tf_model = keras.Model(inputs=input_tensor, outputs=classification_head)
    tf_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return tf_model


@attrs.frozen
class ValidationAccuracy:
    train_dir: pathlib.Path
    validation_dir: pathlib.Path
    input_shape: gl.Shape
    batch_size: int
    epochs: int
    class_count: int
    shuffle_seed: int

    def __attrs_post_init__(self) -> None:
        assert self.batch_size > 0, self.batch_size
        assert self.epochs > 0, self.epochs
        assert self.class_count > 1, self.class_count
        assert self.train_dir.is_dir(), self.train_dir
        assert self.validation_dir.is_dir(), self.validation_dir
        assert self.train_dir != self.validation_dir

    @property
    def should_be_maximized(self) -> bool:
        return True

    def get_train_generator(self) -> DataGen:
        with redirection.discard_stderr_and_stdout():
            return keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
                self.train_dir,
                batch_size=self.batch_size,
                target_size=(self.input_shape.width, self.input_shape.height),
                shuffle=True,
                seed=self.shuffle_seed,
            )

    def get_validation_generator(self) -> DataGen:
        with redirection.discard_stderr_and_stdout():
            return keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
                self.validation_dir,
                batch_size=self.batch_size,
                target_size=(self.input_shape.width, self.input_shape.height),
            )

    def evaluate(self, model: gnn.NeuralNetwork) -> float:
        logger.debug(f"starting fitness evaluation, model=<{model}>")

        train = self.get_train_generator()
        val = self.get_validation_generator()
        tf_model = make_tf_model(model, self.class_count)

        fitting_result = tf_model.fit(
            train,
            epochs=self.epochs,
            steps_per_epoch=train.samples // self.batch_size,
            validation_data=val,
            validation_steps=val.samples // self.batch_size,
            verbose=0,
        )

        val_acc = max(fitting_result.history["val_accuracy"])
        assert isinstance(val_acc, float)

        logger.debug(
            f"finished fitness evaluation, model=<{model}>, accuracy=<{val_acc}>"
        )

        return val_acc


@attrs.frozen
class FitnessEvaluationParameters:
    metric: ValidationAccuracy
    grammar: gr.Grammar
    input_layer: gl.Input


def evaluate(
    genotype: cg.CompositeGenotype,
    params: FitnessEvaluationParameters,
) -> float:
    logger.debug(f"starting fitness evaluation of genotype=<{genotype}>")

    phenotype = gnn.make_network(
        genotype,
        params.grammar,
        params.input_layer,
    )

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
