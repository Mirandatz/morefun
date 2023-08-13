import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from torch import Tensor
from torch.utils.data import DataLoader


class Modelberrg(pl.LightningModule):
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
    net: Modelberrg,
    dataset: DataLoader[torch.Tensor],
    epochs: int,
    optimizer: optim.Optimizer,
) -> None:
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataset):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                running_loss = 0.0
                print(running_loss)
        print(".", end="")


def main() -> None:
    torch.set_float32_matmul_precision("medium")  # type: ignore
    batch_size = 512

    train_set = torchvision.datasets.CIFAR10(
        root="/dev/shm/datasets",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # testset = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, download=True, transform=transform
    # )
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=2
    # )

    model = Modelberrg()
    train_model(model, train_loader, epochs=999, optimizer=optim.Adam(model.para))

    print("Finished Training")


if __name__ == "__main__":
    main()
