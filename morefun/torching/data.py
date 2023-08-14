import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

DATASETS_DIR = Path("/dev/shm/datasets")


class MemoryCachedDataset(VisionDataset):  # type: ignore
    def __init__(
        self,
        dataset: torchvision.datasets.VisionDataset,
    ) -> None:
        self._data = list(dataset)

    def __getitem__(self, index: int) -> Any:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def load_cifar10_train(
    dataset_dir: Path = Path("/dev/shm/datasets"),
) -> MemoryCachedDataset:
    path = dataset_dir / "cifar10_train.pickle"
    expected_md5 = "223f9648865daf4a336eea017bf8ce93"

    if check_integrity(path, expected_md5):
        with path.open("rb") as f:
            dataset = torch.load(f)
        assert isinstance(dataset, MemoryCachedDataset)
        return dataset

    dataset = MemoryCachedDataset(
        torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            download=True,
            transform=get_transforms(),
        )
    )

    with path.open("wb") as f:
        torch.save(dataset, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    assert isinstance(dataset, MemoryCachedDataset)
    return dataset


def load_cifar10_test(
    dataset_dir: Path = Path("/dev/shm/datasets"),
) -> MemoryCachedDataset:
    path = dataset_dir / "cifar10_test.pickle"
    expected_md5 = "f4aafedf9f4cbe5de04e877f97e1e8a5"

    if check_integrity(path, expected_md5):
        with path.open("rb") as f:
            dataset = torch.load(f)
            assert isinstance(dataset, MemoryCachedDataset)
            return dataset

    dataset = MemoryCachedDataset(
        torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            download=True,
            transform=get_transforms(),
        )
    )

    with path.open("wb") as f:
        torch.save(dataset, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    assert isinstance(dataset, MemoryCachedDataset)
    return dataset


def make_train_loader(
    dataset: VisionDataset, batch_size: int, prefetch_factor: int, num_workers: int
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )


def make_test_loader(
    dataset: VisionDataset, batch_size: int, prefetch_factor: int, num_workers: int
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )


def calculate_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(path: Path, md5: str) -> bool:
    if not path.is_file():
        return False

    return calculate_md5(path) == md5
