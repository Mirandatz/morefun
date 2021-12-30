import typing

import numpy as np

RNG = typing.NewType("RNG", np.random.Generator)


def get_rng_seed() -> int:
    return 42


def create_rng(seed: typing.Optional[int] = None) -> RNG:
    if seed is None:
        seed = get_rng_seed()

    bit_gen = np.random.SFC64(seed=seed)
    return RNG(np.random.Generator(bit_gen))
