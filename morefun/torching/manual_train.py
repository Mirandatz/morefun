from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def get_train(
    batch_size: int,
    num_workers: int,
    dataset_dir: Path = Path("/dev/shm/datasets"),
) -> DataLoader[Tensor]:
    train_set = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        download=True,
        transform=get_transforms(),
    )

    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=64,
        persistent_workers=True,
    )


def get_test(
    batch_size: int,
    num_workers: int,
    dataset_dir: Path = Path("/dev/shm/datasets"),
) -> DataLoader[Tensor]:
    test_set = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False, download=True, transform=get_transforms()
    )
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


class Modelberrg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: Tensor) -> Tensor:
        # training_step defines the train loop.
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_model(
    model: Modelberrg,
    data_loader: DataLoader[torch.Tensor],
    epochs: int,
    optimizer: optim.Optimizer,
    device: str,
) -> None:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        running_loss = 0.0
        for _, batch in tqdm.tqdm(
            enumerate(data_loader), unit="batch", total=len(data_loader)
        ):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()


def main() -> None:
    torch.set_float32_matmul_precision("medium")  # type: ignore
    batch_size = 512
    model = Modelberrg()

    train_model(
        model,
        get_train(batch_size, num_workers=16),
        epochs=10,
        optimizer=optim.Adam(
            model.parameters(),
        ),
        device="cuda",
    )

    print("Finished Training")


if __name__ == "__main__":
    main()
