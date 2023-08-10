import torch
from torchmetrics import Metric


class SequenceCharacterAccuracy(Metric):
    correct_sc: torch.Tensor
    total_sc: torch.Tensor

    def __init__(self, pad_index: int):
        super().__init__()
        self.add_state("correct_sc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_sc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pad_index = pad_index

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=-1)

        self.correct_sc += torch.sum(
            torch.logical_and(preds == target, target != self.pad_index)
        )
        self.total_sc += torch.sum(target != self.pad_index)

    def compute(self):
        return self.correct_sc.float() / self.total_sc


class SequenceExactAccuracy(Metric):
    correct_se: torch.Tensor
    total_se: torch.Tensor

    def __init__(self, pad_index: int):
        super().__init__()
        self.add_state("correct_se", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_se", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pad_index = pad_index

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=-1)

        seq = torch.logical_or(preds == target, target == self.pad_index).all(dim=0)
        self.correct_se += torch.sum(seq)
        self.total_se += seq.numel()

    def compute(self):
        return self.correct_se.float() / self.total_se
