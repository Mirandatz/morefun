import numpy as np

RNG = np.random.Generator


def create_rng(seed: int) -> RNG:
    bit_gen = np.random.SFC64(seed=seed)
    return np.random.Generator(bit_gen)
