# import os
# from pathlib import Path
# from typing import Any, Callable, List, Optional, Tuple

# import lightning.pytorch as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch import Tensor
# from torch.utils.data import DataLoader
# from torchmetrics import Accuracy


# class CachedDataset(torchvision.datasets.VisionDataset):  # type: ignore
#     def __init__(
#         self,
#         dataset: torchvision.datasets.VisionDataset,
#     ) -> None:
#         self._data = []
#         for index in range(len(dataset)):
#             x, y = dataset[index]
#             self._data.append((x.to("cuda"), y))

#     def __getitem__(self, index: int) -> Any:
#         return self._data[index]

#     def __len__(self) -> int:
#         return len(self._data)

#     def __repr__(self) -> str:
#         raise NotImplementedError()

#     def _format_transform_repr(self, transform: Any, head: Any) -> None:
#         raise NotImplementedError()

#     def extra_repr(self) -> str:
#         raise NotImplementedError()


# def get_train(
#     batch_size: int,
#     num_workers: int,
#     prefetch_factor: int,
#     dataset_dir: Path = Path("/dev/shm/datasets"),
# ) -> DataLoader[Tensor]:
#     cached = CachedDataset(
#         torchvision.datasets.CIFAR10(
#             root=dataset_dir,
#             train=True,
#             download=True,
#             transform=get_transforms(),
#         )
#     )

#     return torch.utils.data.DataLoader(
#         cached,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#         prefetch_factor=prefetch_factor,
#         persistent_workers=True,
#     )


# def get_transforms() -> transforms.Compose:
#     return transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#         ]
#     )


# def get_test(
#     batch_size: int,
#     num_workers: int,
#     dataset_dir: Path = Path("/dev/shm/datasets"),
# ) -> DataLoader[Tensor]:
#     test_set = torchvision.datasets.CIFAR10(
#         root=dataset_dir, train=False, download=True, transform=get_transforms()
#     )
#     return torch.utils.data.DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )


# class Modelberrg(pl.LightningModule):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def training_step(self, batch: Tensor, batch_idx: Tensor) -> Tensor:
#         # training_step defines the train loop.
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         preds = self.forward(x)
#         loss = self.criterion(preds, y)
#         accuracy = Accuracy(task="multiclass")
#         acc = accuracy(preds, y)
#         self.log("accuracy", acc, on_epoch=True)
#         return loss

#     def configure_optimizers(self) -> optim.Optimizer:
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer


# def main() -> None:
#     torch.set_float32_matmul_precision("medium")  # type: ignore
#     batch_size = 512
#     epochs = 20
#     num_workers = 16
#     model = Modelberrg()
#     device = "gpu"

#     trainer = pl.Trainer(
#         accelerator=device,
#         devices=-1,
#         logger=False,
#         callbacks=[],
#         max_epochs=epochs,
#         enable_checkpointing=False,
#         enable_model_summary=False,
#         enable_progress_bar=True,
#     )

#     trainer.fit(
#         model,
#         get_train(
#             batch_size,
#             num_workers=num_workers,
#             prefetch_factor=64,
#         ),
#     )

#     print("Finished Training")


# if __name__ == "__main__":
#     main()
