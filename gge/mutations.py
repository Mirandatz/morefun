import dataclasses as dt
import typing

import gge.connections as conn
import gge.genotypes as geno
import gge.randomness as rand

_T = typing.TypeVar("_T")


def _mutate_tuple(tup: tuple[_T, ...], index: int, new_value: _T) -> tuple[_T, ...]:
    assert 0 <= index < len(tup)
    temporary_container = list(tup)
    temporary_container[index] = new_value
    return tuple(temporary_container)


def mutate(genotype: geno.Genotype, rng: rand.RNG) -> geno.Genotype:
    if rng.random() < 0.5:
        raise NotImplementedError()

    else:
        new_connections = mutate_connections_schema(
            genotype.connections_genotype,
            rng,
        )
        return dt.replace(
            genotype,
            connections_genotype=new_connections,
        )


def mutate_connections_schema(
    schema: conn.ConnectionsSchema,
    rng: rand.RNG,
) -> conn.ConnectionsSchema:
    old_params = schema.merge_params

    index = rng.integers(low=0, high=len(old_params))
    mutation_target = old_params[index]
    mutation_result = mutate_merge_parameters(mutation_target, rng)

    new_params = _mutate_tuple(old_params, index, mutation_result)

    return dt.replace(schema, merge_params=new_params)


def mutate_merge_parameters(
    params: conn.MergeParameters,
    rng: rand.RNG,
) -> conn.MergeParameters:
    options = ["forks", "reshape", "merge"]

    if len(params.forks_mask) == 0:
        options.remove("forks")

    mutation_type = rng.choice(options)

    if mutation_type == "forks":
        new_forks_mask = mutate_forks_mask(params.forks_mask, rng)
        return dt.replace(params, forks_mask=new_forks_mask)

    elif mutation_type == "reshape":
        new_reshape_strategy = mutate_reshape_strategy(params.reshape_strategy)
        return dt.replace(params, reshape_strategy=new_reshape_strategy)

    else:
        new_merge_strategy = mutate_merge_strategy(params.merge_strategy)
        return dt.replace(params, merge_strategy=new_merge_strategy)


def mutate_forks_mask(mask: tuple[bool, ...], rng: rand.RNG) -> tuple[bool, ...]:
    index = rng.integers(low=0, high=len(mask))
    old_value = mask[index]
    new_value = not old_value
    return _mutate_tuple(mask, index, new_value)


def mutate_reshape_strategy(strat: conn.ReshapeStrategy) -> conn.ReshapeStrategy:
    if strat == conn.ReshapeStrategy.DOWNSAMPLE:
        return conn.ReshapeStrategy.UPSAMPLE

    elif strat == conn.ReshapeStrategy.UPSAMPLE:
        return conn.ReshapeStrategy.DOWNSAMPLE

    else:
        raise ValueError(f"unknown reshape strategy: {strat}")


def mutate_merge_strategy(strat: conn.MergeStrategy) -> conn.MergeStrategy:
    if strat == conn.MergeStrategy.ADD:
        return conn.MergeStrategy.CONCAT

    elif strat == conn.MergeStrategy.CONCAT:
        return conn.MergeStrategy.ADD

    else:
        raise ValueError(f"unknown merge strategy: {strat}")
