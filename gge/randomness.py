from typing import NewType, Optional

import numpy as np

RNG = NewType("RNG", np.random.Generator)


def get_rng_seed() -> int:
    return 42


def create_rng(seed: Optional[int] = None) -> RNG:
    if seed is None:
        seed = get_rng_seed()

    bit_gen = np.random.SFC64(seed=seed)
    return RNG(np.random.Generator(bit_gen))


def main() -> None:
    rng = create_rng()
    print(rng.integers(low=0, high=10, size=10))
    print(rng.integers(low=0, high=10, size=10))

    rng = create_rng()
    print(rng.integers(low=0, high=10, size=10))
    print(rng.integers(low=0, high=10, size=10))


if __name__ == "__main__":
    main()
