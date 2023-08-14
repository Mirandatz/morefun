import pickle
import tempfile
from pathlib import Path

import torch

import morefun.torching.data as mtd
from morefun.redirection import discard_stderr_and_stdout


def main() -> None:
    dataset_map = {
        "cifar10_train": mtd.load_cifar10_train,
        "cifar10_test": mtd.load_cifar10_test,
    }

    for name, constructor in dataset_map.items():
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp_dir:
            dir_path = Path(tmp_dir)
            file_path = dir_path / f"{name}.pickle"

            with discard_stderr_and_stdout():
                dataset = constructor(dataset_dir=dir_path)

            with file_path.open("wb") as f:
                torch.save(dataset, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

            print(name, mtd.calculate_md5(file_path))


if __name__ == "__main__":
    main()
