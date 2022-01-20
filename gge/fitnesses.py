import dataclasses
import pathlib
import typing

import keras
import tensorflow as tf

import gge.composite_genotypes as cg
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn

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
        graph = gnn.convert_to_digraph(model)
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
        data_gen = keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.15,
        )

        batch_size, width, height, depth = self._input_shape
        train = data_gen.flow_from_directory(
            self._dataset_dir,
            batch_size=self.batch_size,
            target_size=(self.input_width, self.input_height),
            subset="training",
        )

        val = data_gen.flow_from_directory(
            self._dataset_dir,
            batch_size=self.batch_size,
            target_size=(self.input_width, self.input_height),
            subset="validation",
        )

        return train, val

    def _try_evaluate(self, model: gnn.NeuralNetwork) -> float:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        tf_model = model.to_tensorflow()
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
        )

        val_acc = max(fitting_result.history["val_accuracy"])
        assert isinstance(val_acc, float)
        return val_acc

    def evaluate(self, model: gnn.NeuralNetwork) -> float:
        try:
            return self._try_evaluate(model)
        except tf.errors.ResourceExhaustedError:
            return float("-inf")


@dataclasses.dataclass(frozen=True)
class FitnessEvaluationParameters:
    metric: FitnessMetric
    grammar: gr.Grammar
    input_layer: gl.Input


def evaluate(
    genotype: cg.CompositeGenotype,
    params: FitnessEvaluationParameters,
) -> float:
    phenotype = gnn.make_network(
        genotype,
        params.grammar,
        params.input_layer,
    )

    fitness = params.metric.evaluate(phenotype)
    return fitness


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
