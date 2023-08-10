import dataclasses
import os
from glob import glob
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import MetricCollection
from tqdm import tqdm

from ._constants import MODELS_PATH
from ._metrics import SequenceCharacterAccuracy
from ._metrics import SequenceExactAccuracy
from .dataset import collate_fn
from .dataset import Dataset
from .model import MODELS
from .vocab import decode
from .vocab import PAD_TOKEN_IDX


@dataclasses.dataclass
class Trainer:
    model_name: str
    load_path: Path | None
    train_path: Path
    val_path: Path
    batch_size: int
    device: torch.device
    epochs: int
    lr: float
    num_samples: int
    num_workers: int
    weight_decay: float

    def __post_init__(self):
        if len(glob(os.path.join("runs", f"*{str(self)}"))) != 0:
            raise RuntimeError(
                f"A run for this configuration ({str(self)}) already exists"
            )

        self.model = torch.nn.DataParallel(MODELS[self.model_name]())
        if self.load_path is not None:
            with open(self.load_path, "rb") as f:
                self.model.load_state_dict(torch.load(f))

        self.train_loader = _init_dataloader(self.train_path, self)
        self.val_loader = _init_dataloader(self.val_path, self)
        self.h_params = {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if type(v) in [int, float, str, bool, torch.Tensor]
        }

        self.best_val_loss = float("inf")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
        self.metrics = MetricCollection(
            [
                SequenceExactAccuracy(pad_index=PAD_TOKEN_IDX),
                SequenceCharacterAccuracy(pad_index=PAD_TOKEN_IDX),
            ]
        )

        self.writer = SummaryWriter(comment=str(self))
        os.makedirs(MODELS_PATH, exist_ok=True)

        self.model.to(self.device)
        self.metrics.to(self.device)

    def train(self) -> None:
        for epoch in tqdm(range(self.epochs)):
            self._train_iter(epoch)
            self._val_iter(epoch)

        self.writer.add_hparams(
            hparam_dict=self.h_params,
            metric_dict=self.metrics.compute(),
        )

    def _train_iter(self, epoch: int) -> None:
        self.model.train()
        loss_sum = 0

        pbar_loader = tqdm(self.train_loader, leave=False)
        for item in pbar_loader:
            item.to(self.device)
            model_output = self.model(item)

            loss = self.loss_func(
                model_output.movedim(0, -1), item.tgt_tokens_out.movedim(0, -1)
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            pbar_loader.set_description(f"train batch loss: {loss.item()}")

        self.writer.add_scalar("loss/train", loss_sum / len(pbar_loader), epoch)

    @torch.no_grad()
    def _val_iter(self, epoch: int) -> None:
        self.model.eval()
        loss_sum = 0
        self.metrics.reset()

        pbar_loader = tqdm(self.val_loader, leave=False)
        for batch_num, item in enumerate(pbar_loader):
            item.to(self.device)
            model_output = self.model(item)
            output_norm = torch.softmax(model_output, dim=-1)

            loss = self.loss_func(
                model_output.movedim(0, -1), item.tgt_tokens_out.movedim(0, -1)
            )

            loss_sum += loss.item()
            self.metrics.update(output_norm, item.tgt_tokens_out)
            pbar_loader.set_description(f"val batch loss: {loss.item()}")

            model_pred = torch.argmax(output_norm, dim=-1)
            if batch_num == 0:
                sample_strings = [
                    f"Source: {decode(item.src_tokens[:, i].tolist())}"
                    + f"\nPredicted: {decode(model_pred[:, i].tolist())}"
                    + f"\nActual: {decode(item.tgt_tokens_out[:, i])}"
                    for i in range(min(self.num_samples, model_pred.size(1)))
                ]
                sample = "\n\n".join(sample_strings)
                sample = sample.encode(errors="replace").decode()
                self.writer.add_text("Samples", f"<pre>{sample}</pre>", epoch)

        loss = loss_sum / len(pbar_loader)
        self.writer.add_scalar("loss/val", loss, epoch)
        for metric, score in self.metrics.compute().items():
            self.writer.add_scalar(f"{metric}/val", score, epoch)

        if loss < self.best_val_loss:
            torch.save(
                self.model.state_dict(), os.path.join(MODELS_PATH, f"{str(self)}.pt")
            )

    def __str__(self) -> str:
        return " ".join(
            f"{k}={str(v).replace('/', '')}"
            for k, v in dataclasses.asdict(self).items()
        )


def _init_dataloader(path: Path, trainer: Trainer) -> DataLoader:
    dataset = Dataset(path)
    return DataLoader(
        dataset,
        batch_size=trainer.batch_size,
        shuffle=True,
        num_workers=trainer.num_workers,
        collate_fn=collate_fn,
    )
