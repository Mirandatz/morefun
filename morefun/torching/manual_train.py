import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import morefun.torching.data as mtd


class Modelberrg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(
    model: Modelberrg,
    data_loader: DataLoader[Tensor],
    epochs: int,
    optimizer: optim.Optimizer,
    device: str,
) -> None:
    model = model.train().to(device)
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


def check_accuracy(
    model: nn.Module, loader: DataLoader[Tensor], device: str
) -> NDArray[np.float32]:
    metric = MulticlassAccuracy(num_classes=10).to(device)
    model = model.eval().to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            predicted = model(x)
            metric.update(preds=predicted, target=y)

    acc = metric.compute().cpu().numpy()
    assert isinstance(acc, np.ndarray)
    return acc


def main() -> None:
    torch.set_float32_matmul_precision("medium")  # type: ignore
    batch_size = 512
    prefetch_factor = 32
    num_workers = 4
    epochs = 10
    model = Modelberrg()
    device = "cuda"

    train_dataset = mtd.load_cifar10_train()
    train_loader = mtd.make_train_loader(
        train_dataset,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    train_model(
        model,
        train_loader,
        epochs=epochs,
        optimizer=optim.Adam(
            model.parameters(),
        ),
        device=device,
    )

    test_dataset = mtd.load_cifar10_test()
    test_loader = mtd.make_test_loader(
        test_dataset,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )
    acc = check_accuracy(
        model,
        test_loader,
        device,
    )

    print(acc)


if __name__ == "__main__":
    main()
