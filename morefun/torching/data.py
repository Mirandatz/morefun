from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

DATASETS_DIR = Path("/dev/shm/datasets")


class CachedDataset(VisionDataset):  # type: ignore
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
    batch_size: int,
    prefetch_factor: int,
    num_workers: int,
    dataset_dir: Path = Path("/dev/shm/datasets"),
) -> DataLoader[Tensor]:
    cached = CachedDataset(
        torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            download=True,
            transform=get_transforms(),
        )
    )
    return torch.utils.data.DataLoader(
        cached,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )


def load_cifar10_test(
    batch_size: int,
    prefetch_factor: int,
    num_workers: int,
    dataset_dir: Path = Path("/dev/shm/datasets"),
) -> DataLoader[Tensor]:
    cached = CachedDataset(
        torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            download=True,
            transform=get_transforms(),
        )
    )
    return torch.utils.data.DataLoader(
        cached,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
