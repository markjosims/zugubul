import os
from glob import glob
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from zugubul.models._constants import MODELS_PATH
from zugubul.models._metrics import SequenceCharacterAccuracy
from zugubul.models._metrics import SequenceExactAccuracy
from zugubul.models.dataset import collate_fn
from zugubul.models.dataset import Dataset
from zugubul.models.models import MODELS
from zugubul.models.vocab import decode
from zugubul.models.vocab import PAD_TOKEN_IDX


@torch.no_grad()
def infer(
    model_name: str,
    data: Path,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    output: Path,
):
    model = torch.nn.DataParallel(MODELS[model_name]())
    model.to(device)
    _load_model(model, model_name)
    model.eval()
    data_loader = DataLoader(
        Dataset(data),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)

    metrics = MetricCollection(
        [
            SequenceExactAccuracy(pad_index=PAD_TOKEN_IDX),
            SequenceCharacterAccuracy(pad_index=PAD_TOKEN_IDX),
        ]
    )
    metrics.to(device)
    preds = []
    for item in tqdm(data_loader):
        item.to(device)
        model_output = model(item)
        output_norm = torch.softmax(model_output, dim=-1)

        metrics.update(output_norm, item.tgt_tokens_out)

        model_pred = torch.argmax(output_norm, dim=-1)
        preds += [
            f"{decode(seq.tolist(), include_special_tokens=False)}\n"
            for seq in model_pred.T
        ]

    for metric, score in metrics.compute().items():
        print(f"{metric}: {score}")

    with open(output, "w") as f:
        f.writelines(preds)


def _load_model(model: nn.Module, model_name: str) -> None:
    models = glob(os.path.join(MODELS_PATH, f"*{model_name} *.pt"))
    num_models = len(models)
    if num_models == 1:
        model_file = models[0]
    else:
        print("\n".join(f"{i}: {model_file}" for i, model_file in enumerate(models)))
        model_file = models[int(input(f"\nEnter a selection (0-{num_models - 1}): "))]
    with open(model_file, "rb") as f:
        model.load_state_dict(torch.load(f))
