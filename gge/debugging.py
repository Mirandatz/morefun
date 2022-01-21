import pathlib
import pickle
import tempfile
import typing

import gge.composite_genotypes as cg

DEBUG_DATA_DIR = pathlib.Path(__file__).parent.parent / "debug_data"
DEBUG_DATA_DIR.mkdir(exist_ok=True)


def save_genotype(genotype: cg.CompositeGenotype) -> str:
    serialized = pickle.dumps(genotype, protocol=pickle.HIGHEST_PROTOCOL)
    with tempfile.NamedTemporaryFile(dir=DEBUG_DATA_DIR, delete=False) as file:
        file.write(serialized)
        return file.name


def load_genotype(filename: str) -> cg.CompositeGenotype:
    serialized = (DEBUG_DATA_DIR / filename).read_bytes()
    deserialized = pickle.loads(serialized)
    return typing.cast(cg.CompositeGenotype, deserialized)
