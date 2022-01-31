import pathlib
import pickle

import tensorflow as tf  # noqa
import typer
from loguru import logger

import gge.composite_genotypes as cg
import gge.experiments.gecco.run_evolution as run_exp
import gge.fitnesses as gfit
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn


def plot_model(
    genotype: cg.CompositeGenotype,
    grammar: gr.Grammar,
    class_count: int,
    input_layer: gl.Input,
    path: pathlib.Path,
) -> None:
    network = gnn.make_network(genotype, grammar, input_layer)
    model = gfit.make_tf_model(network, class_count)
    tf.keras.utils.plot_model(
        model=model,
        to_file=path,
        show_layer_names=False,
        show_shapes=True,
    )


def main(
    genotype_path: pathlib.Path = typer.Option(
        ...,
        "-g",
        "--genotype",
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "-o",
        "--output",
        file_okay=False,
        exists=True,
        dir_okay=True,
        writable=True,
    ),
) -> None:
    logger.remove()
    output_path = output_dir / genotype_path.with_suffix(".png").name
    genotype: cg.CompositeGenotype = pickle.loads(genotype_path.read_bytes())
    grammar = run_exp.get_grammar()
    plot_model(
        genotype,
        grammar,
        run_exp.CLASS_COUNT,
        run_exp.get_input_layer(),
        output_path,
    )


if __name__ == "__main__":
    typer.run(main)
