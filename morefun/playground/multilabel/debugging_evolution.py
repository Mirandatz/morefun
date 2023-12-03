import os
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from scipy.io.arff import loadarff

import morefun.evolutionary.fitnesses as gf
import morefun.evolutionary.generations
import morefun.evolutionary.novelty
import morefun.experiments.create_initial_population_genotypes as mf_init
import morefun.experiments.settings as mf_cfg
import morefun.randomness as rand


def config_env(
    unmute_tensorflow: bool,
    use_xla: bool,
    use_mixed_precision: bool,
) -> None:
    if unmute_tensorflow and "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]

    if use_xla:
        tf.config.optimizer.set_jit("autoclustering")
    else:
        os.environ.pop("TF_XLA_FLAGS", None)
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def load_yeast(
    partition: Literal["all", "train", "test"],
    batch_size: int,
) -> tf.data.Dataset:
    # https://mulan.sourceforge.net/datasets-mlc.html
    match partition:
        case "all":
            path = "morefun/playground/input/yeast.arff"

        case "train":
            path = "morefun/playground/input/yeast-train.arff"

        case "test":
            path = "morefun/playground/input/yeast-test.arff"
        case _:
            raise ValueError(f"unknown partition={partition}")

    dataset, metadata = loadarff(path)
    df = pd.DataFrame(dataset)

    column_names = df.columns
    features_names = [c for c in column_names if "Att" in c]
    targets_names = [c for c in column_names if "Class" in c]
    assert set(features_names).isdisjoint(targets_names)
    assert set(features_names) | set(targets_names) == set(column_names)

    features = df[features_names].astype(np.float32)
    targets = df[targets_names].replace({b"0": 0, b"1": 1})

    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    return (
        ds.cache()
        .shuffle(
            ds.cardinality(),
            seed=rand.get_fixed_seed(),
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_settings() -> mf_cfg.MorefunSettings:
    raw = r"""
experiment:
    rng_seed: 0
    description: >
        debug evolution multilabel classification

dataset:
    partitions_dir: "/app/datasets/cifar10_train_test"
    image_height: 32
    image_width: 32
    image_depth: 3
    class_count: 10
    train_instances: 50000
    validation_instances: 0
    test_instances: 10000

output:
    directory: "/app/playground/output"
    log_level: "INFO"

population_initialization:
    population_size: 2
    max_network_depth: 9999
    wide_layer_threshold: 9999
    max_wide_layers: 9999
    max_network_params: 999999999

evolution:
    mutation:
        mutants_per_generation: 1
        max_failures_per_generation: 500

    fitness_estimation:
        batch_size: 256
        max_epochs: 2
        early_stop_patience: 6
        metrics: ["train_loss", "number_of_parameters"]

final_train:
    train_k_fittest: 20
    batch_size: 256
    max_epochs: 2000
    early_stop_patience: 10

tensorflow:
    xla: true
    mixed_precision: true

grammar: |
    start : learning optimizer

    learning : intro_a mid 
             | intro_b mid

    intro_a : dense_small act norm
    intro_b : dense_large act norm

    dense_small : "dense" ("8" | "16" | "24" | "32")
    dense_large : "dense" ("64" | "96" | "128" | "160")
    
    mid : block~1..6

    block : dense_small act norm
          | dense_large act norm
          
    norm : "batchnorm"

    act   : relu | prelu
    relu  : "relu"
    prelu : "prelu"

    optimizer : adam | ranger
    adam      : "adam" "learning_rate" "0.001" "beta1" "0.9" "beta2" "0.999" "epsilon" "1e-07" "amsgrad" "false"
    ranger    : "ranger" "learning_rate" "0.001" "beta1" "0.9" "beta2" "0.999" "epsilon" "1e-07" "amsgrad" "false" "sync_period" "6" "slow_step_size" "0.5"
"""
    return mf_cfg.MorefunSettings.from_yaml(yaml.safe_load(raw))


def initialize() -> None:
    settings = get_settings()

    mf_cfg.configure_logger(settings.output)
    mf_cfg.configure_tensorflow(settings.tensorflow)

    rng_seed = settings.experiment.rng_seed

    individuals = mf_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    metrics = mf_cfg.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        output=settings.output,
    )

    genotypes = [ind.genotype for ind in individuals]
    phenotypes = [ind.phenotype for ind in individuals]
    fitnesses = [gf.evaluate(ind.phenotype, metrics) for ind in individuals]

    known_genotypes = set(genotypes)
    known_phenotypes = set(phenotypes)
    novelty_tracker = morefun.evolutionary.novelty.NoveltyTracker(
        known_genotypes=known_genotypes,
        known_phenotypes=known_phenotypes,
    )

    initial_population = [
        morefun.evolutionary.generations.EvaluatedGenotype(g, p, f)
        for g, p, f in zip(
            genotypes,
            phenotypes,
            fitnesses,
        )
    ]

    # generations are 0-indexed, so first gen == 0
    generation_number = 0

    checkpoint = morefun.evolutionary.generations.GenerationCheckpoint(
        generation_number=generation_number,
        population=tuple(initial_population),
        rng=morefun.randomness.create_rng(rng_seed),
        novelty_tracker=novelty_tracker,
    )

    save_path = morefun.paths.get_generation_checkpoint_path(
        settings.output.directory, generation_number
    )

    checkpoint.save(save_path)


def evolve(generations: int) -> None:
    settings = get_settings()

    mf_cfg.configure_logger(settings.output)
    mf_cfg.configure_tensorflow(settings.tensorflow)

    latest_checkpoint = morefun.evolutionary.generations.GenerationCheckpoint.load(
        morefun.paths.get_latest_generation_checkpoint_path(settings.output.directory)
    )

    current_generation_number = latest_checkpoint.get_generation_number() + 1

    mutation_params = mf_cfg.make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    metrics = mf_cfg.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        output=settings.output,
    )

    morefun.evolutionary.generations.run_multiple_generations(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_checkpoint.get_population(),
        grammar=settings.grammar,
        mutation_params=mutation_params,
        metrics=metrics,
        novelty_tracker=latest_checkpoint.get_novelty_tracker(),
        rng=latest_checkpoint.get_rng(),
        output_dir=settings.output.directory,
    )


if __name__ == "__main__":
    initialize()
    evolve(generations=2)
