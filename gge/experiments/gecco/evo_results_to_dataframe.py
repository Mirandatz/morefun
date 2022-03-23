import functools
import multiprocessing
import pathlib
import pickle
import re

import keras
import networkx as nx
import pandas as pd
import tensorflow as tf
import tqdm
from loguru import logger

import gge.composite_genotypes as cg
import gge.experiments.gecco.run_evolution as exp
import gge.fitnesses as gfit
import gge.layers as gl
import gge.neural_network as gnn

GENOTYPES_DIR = pathlib.Path().home() / "experiments"


@functools.cache
def get_network(genotype_path: pathlib.Path) -> gnn.NeuralNetwork:
    genotype: cg.CompositeGenotype = pickle.loads(genotype_path.read_bytes())
    return gnn.make_network(genotype, exp.get_grammar(), exp.get_input_layer())


@functools.cache
def get_graph(genotype_path: pathlib.Path) -> nx.DiGraph:
    model = get_network(genotype_path)
    return gnn.convert_to_digraph(model.output_layer)


def get_tf_model(genotype_path: pathlib.Path) -> keras.Model:
    network = get_network(genotype_path)
    tf_model = gfit.make_tf_model(network, exp.CLASS_COUNT)
    return tf_model


def extract_gen_nr(genotype_path: pathlib.Path) -> int:
    match = re.search(
        pattern=r"generation_(\d+)",
        string=genotype_path.name,
    )
    assert isinstance(match, re.Match)
    return int(match.group(1))


def extract_id(genotype_path: pathlib.Path) -> int:
    match = re.search(
        pattern=r"genotype_(\d+)",
        string=genotype_path.name,
    )
    assert isinstance(match, re.Match)
    return int(match.group(1))


def extract_fitness(genotype_path: pathlib.Path) -> float:
    match = re.search(
        pattern=r"fitness_(\d+\.\d+)",
        string=genotype_path.name,
    )
    assert isinstance(match, re.Match)
    return float(match.group(1))


def extract_num_layers(genotype_path: pathlib.Path) -> int:
    graph = get_graph(genotype_path)
    conv_layers = [layer for layer in graph.nodes]
    return len(conv_layers)


def extract_num_skips(genotype_path: pathlib.Path) -> int:
    graph = get_graph(genotype_path)
    skip_connections = [
        layer
        for layer in graph.nodes
        if isinstance(layer, gl.ConnectedAdd | gl.ConnectedConcatenate)
    ]
    return len(skip_connections)


@functools.cache
def extract_num_params(genotype_path: pathlib.Path) -> int:
    with tf.device("/cpu:0"):
        tf_model = get_tf_model(genotype_path)
        num_params = tf_model.count_params()
        assert isinstance(num_params, int)
        return num_params


def extract_data(genotype_path: pathlib.Path) -> dict[str, int | float]:
    return {
        "gen": extract_gen_nr(genotype_path),
        "id": extract_id(genotype_path),
        "fit": extract_fitness(genotype_path),
        "layers": extract_num_layers(genotype_path),
        "skips": extract_num_skips(genotype_path),
        "params": extract_num_params(genotype_path),
    }


def compute_run_data(run_dir: pathlib.Path) -> pd.DataFrame:
    pickles = [file for file in run_dir.iterdir() if file.suffix == ".pickle"]
    data = [extract_data(fn) for fn in pickles]
    return pd.DataFrame(data)


def main() -> None:
    logger.remove()
    for rundir in [
        GENOTYPES_DIR / "seed0",
        GENOTYPES_DIR / "seed1",
        GENOTYPES_DIR / "seed2",
    ]:
        pickles = list(rundir.glob("*.pickle"))

        data = []
        with (
            multiprocessing.Pool(processes=2) as pool,
            tqdm.tqdm(total=len(pickles)) as pbar,
        ):
            for datum in pool.imap_unordered(extract_data, pickles):
                data.append(datum)
                pbar.update()

        df = pd.DataFrame(data)
        df.to_parquet(f"~/Desktop/{rundir.name}.parquet")


if __name__ == "__main__":
    main()
