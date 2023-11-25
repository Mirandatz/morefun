import tempfile
from pathlib import Path

from morefun.composite_genotypes import CompositeGenotype


def atomic_write(path: Path, content: bytes) -> None:
    tmp_file = tempfile.TemporaryFile(mode="wb", dir=path.parent)
    tmp_file.write(content)
    tmp_file.close()
    Path(tmp_file.name).rename(path)


def save_genotype(dir: Path, genotype: CompositeGenotype) -> None:
    raise NotImplementedError()
