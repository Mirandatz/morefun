import dataclasses
import pathlib
import typing

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


class FitnessMetric(typing.Protocol):
    @property
    def should_be_maximized(self) -> bool:
        ...

    def evaluate(self, model: gnn.NeuralNetwork) -> float:
        ...


class LayerCount:
    """
    This is a dummy fitness metric used for debugging/testing purposes.
    """

    @property
    def should_be_maximized(self) -> bool:
        return False

    def evaluate(self, model: gnn.NeuralNetwork) -> float:
        graph = gnn.convert_to_digraph(model.output_layer)
        layers = graph.nodes
        cool_layers = [
            layer
            for layer in layers
            if isinstance(layer, gl.SingleInputLayer | gl.MultiInputLayer)
        ]
        return len(cool_layers)


class ValidationAccuracy:
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        input_shape: tuple[int, int, int, int],
        epochs: int,
        class_count: int,
        shuffle_seed: int,
        validation_ratio: float,
    ) -> None:
        for d in input_shape:
            assert d >= 1
        assert epochs > 0
        assert class_count > 1
        assert dataset_dir.is_dir()
        assert 0 < validation_ratio < 1

        self._dataset_dir = dataset_dir
        self._input_shape = input_shape
        self._epochs = epochs
        self._class_count = class_count
        self._shuffle_seed = shuffle_seed
        self._validation_ratio = validation_ratio

    @property
    def should_be_maximized(self) -> bool:
        return True

    @property
    def batch_size(self) -> int:
        return self._input_shape[0]

    @property
    def input_width(self) -> int:
        return self._input_shape[1]

    @property
    def input_height(self) -> int:
        return self._input_shape[2]

    @property
    def input_depth(self) -> int:
        return self._input_shape[3]

    def get_train_and_val(self) -> tuple[DataGen, DataGen]:
        with redirection.discard_stderr_and_stdout():
            data_gen = keras.preprocessing.image.ImageDataGenerator(
                validation_split=self._validation_ratio,
            )

            train = data_gen.flow_from_directory(
                self._dataset_dir,
                batch_size=self.batch_size,
                target_size=(self.input_width, self.input_height),
                shuffle=True,
                seed=self._shuffle_seed,
                subset="training",
            )

            val = data_gen.flow_from_directory(
                self._dataset_dir,
                batch_size=self.batch_size,
                target_size=(self.input_width, self.input_height),
                subset="validation",
            )

        return train, val

    def make_classification_head(self, input_tensor: tf.Tensor) -> tf.Tensor:
        _, width, height, _ = input_tensor.shape

        conv = kl.Conv2D(
            filters=self._class_count,
            kernel_size=(width, height),
        )(input_tensor)

        global_pool = kl.GlobalMaxPooling2D()(conv)

        return kl.Activation(tf.nn.softmax)(global_pool)

    def evaluate(self, model: gnn.NeuralNetwork) -> float:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        input_tensor, output_tensor = model.to_input_output_tensores()
        classification_head = self.make_classification_head(output_tensor)
        tf_model = keras.Model(inputs=input_tensor, outputs=classification_head)
        tf_model.compile(
            loss="categorical_crossentropy",
            optimizer="nadam",
            metrics=["accuracy"],
        )

        train, val = self.get_train_and_val()

        fitting_result = tf_model.fit(
            train,
            epochs=self._epochs,
            steps_per_epoch=train.samples // self.batch_size,
            validation_data=val,
            validation_steps=val.samples // self.batch_size,
            verbose=0,
        )

        val_acc = max(fitting_result.history["val_accuracy"])
        assert isinstance(val_acc, float)
        return val_acc


@dataclasses.dataclass(frozen=True)
class FitnessEvaluationParameters:
    metric: FitnessMetric
    grammar: gr.Grammar
    input_layer: gl.Input


def evaluate(
    genotype: cg.CompositeGenotype,
    params: FitnessEvaluationParameters,
) -> float:
    logger.trace("evaluate")

    phenotype = gnn.make_network(
        genotype,
        params.grammar,
        params.input_layer,
    )

    try:
        return params.metric.evaluate(phenotype)

    except tf.errors.ResourceExhaustedError:
        logger.warning(
            f"Unable to evalute genotype due to resource exhaustion; genotype=<{genotype}>"
        )
        return float("-inf")

    except tf.errors.InvalidArgumentError:
        filename = debug.save_genotype(genotype)
        logger.error(
            f"Unable to evaluate genotype because the phenotype is malformed; saved as=<{filename}>"
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

    best_to_worst = sorted(
        population.keys(),
        key=lambda g: population[g],
        reverse=True,
    )

    fittest = best_to_worst[:fittest_count]
    return {g: population[g] for g in fittest}
