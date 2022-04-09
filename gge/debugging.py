import pickle
import tempfile

import gge.composite_genotypes as cg


def save_genotype(genotype: cg.CompositeGenotype) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="gge_debug_",
        suffix=".genotype",
        delete=False,
    ) as file:
        pickle.dump(obj=genotype, file=file, protocol=pickle.HIGHEST_PROTOCOL)
        return file.name
