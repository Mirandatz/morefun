import yaml

import gge.evolutionary.fitnesses as gf
import gge.evolutionary.generations
import gge.evolutionary.novelty
import gge.experiments.create_initial_population_genotypes as gge_init
import gge.experiments.settings as gset
import gge.persistence


def get_settings() -> gset.GgeSettings:
    raw = r"""
experiment:
    rng_seed: 0
    description: >
        new grammar, merge train + validation

dataset:
    partitions_dir: "/gge/datasets/cifar10_train_test"
    image_height: 32
    image_width: 32
    image_depth: 3
    class_count: 10
    train_instances: 50000
    validation_instances: 0
    test_instances: 10000

output:
    directory: "/gge/output"
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
    start : data_aug learning optimizer

    data_aug  : flip rotate translate
    flip      : "random_flip" "horizontal"
    rotate    : "random_rotation" ("15" | "30" | "45")
    translate : "random_translation" ("0.1" | "0.15" | "0.20")

    learning : intro_a mid | intro_b mid

    intro_a : conv_128_3 act norm conv_128_3 act norm "fork"
    intro_b : conv_128_5 act norm conv_128_5 act norm "fork"

    mid : pool block_1~1..2 pool block_2~1..2 pool block_3~1..2 pool block_3~1..2

    block_1 : conv_128_3 act norm
            | conv_128_5 act norm

    block_2 : conv_256_3 act norm
            | conv_256_5 act norm

    block_3 : conv_512_3 act norm
            | conv_512_5 act norm

    conv_128_3 : "conv" "filter_count" "128" "kernel_size" "3" "stride" "1"
    conv_128_5 : "conv" "filter_count" "128" "kernel_size" "5" "stride" "1"
    conv_256_3 : "conv" "filter_count" "256" "kernel_size" "3" "stride" "1"
    conv_256_5 : "conv" "filter_count" "256" "kernel_size" "5" "stride" "1"
    conv_512_3 : "conv" "filter_count" "512" "kernel_size" "3" "stride" "1"
    conv_512_5 : "conv" "filter_count" "512" "kernel_size" "5" "stride" "1"

    pool    : maxpool | avgpool
    maxpool : "maxpool" "pool_size" "2" "stride" "2"
    avgpool : "avgpool" "pool_size" "2" "stride" "2"

    norm : "batchnorm"

    act   : relu | prelu
    relu  : "relu"
    prelu : "prelu"

    optimizer : adam | ranger
    adam      : "adam" "learning_rate" "0.001" "beta1" "0.9" "beta2" "0.999" "epsilon" "1e-07" "amsgrad" "false"
    ranger    : "ranger" "learning_rate" "0.001" "beta1" "0.9" "beta2" "0.999" "epsilon" "1e-07" "amsgrad" "false" "sync_period" "6" "slow_step_size" "0.5"
"""
    return gset.GgeSettings.from_yaml(yaml.safe_load(raw))


def initialize() -> None:
    settings = get_settings()
    gset.configure_logger(settings.output)
    gset.configure_tensorflow(settings.tensorflow)

    rng_seed = settings.experiment.rng_seed

    individuals = gge_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    metrics = gset.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
    )

    genotypes = [ind.genotype for ind in individuals]
    phenotypes = [ind.phenotype for ind in individuals]
    fitnesses = {
        ind.genotype: gf.evaluate(ind.phenotype, metrics) for ind in individuals
    }

    known_genotypes = set(genotypes)
    known_phenotypes = set(phenotypes)
    novelty_tracker = gge.evolutionary.novelty.NoveltyTracker(
        known_genotypes=known_genotypes,
        known_phenotypes=known_phenotypes,
    )

    # generations are 0-indexed, so first gen == 0
    gge.persistence.save_generational_artifacts(
        generation_number=0,
        fittest=fitnesses,
        novelty_tracker=novelty_tracker,
        rng=gge.randomness.create_rng(rng_seed),
        output_dir=settings.output.directory,
    )


def evolve(generations: int) -> None:
    settings = get_settings()
    gset.configure_logger(settings.output)
    gset.configure_tensorflow(settings.tensorflow)

    latest_gen_output = gge.persistence.load_latest_generational_artifacts(
        settings.output.directory
    )

    current_generation_number = latest_gen_output.get_generation_number() + 1

    mutation_params = gset.make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    metrics = gset.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
    )

    gge.evolutionary.generations.run_multiple_generations(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_gen_output.get_fittest(),
        grammar=settings.grammar,
        mutation_params=mutation_params,
        metrics=metrics,
        novelty_tracker=latest_gen_output.get_novelty_tracker(),
        rng=latest_gen_output.get_rng(),
        output_dir=settings.output.directory,
    )


if __name__ == "__main__":
    initialize()
    evolve(generations=20)
