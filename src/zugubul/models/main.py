from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List

import torch

from zugubul.models.infer import infer
from zugubul.models.models import MODELS
from zugubul.models.train import Trainer


def _get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "model_name", choices=MODELS.keys(), help="Which model to use"
    )
    parent_parser.add_argument(
        "--load", required=False, type=Path, help="Path of a pretrained model state"
    )
    parent_parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="number of examples per batch"
    )
    parent_parser.add_argument(
        "-w",
        "--num-workers",
        default=2,
        type=int,
        help="Number of workers for data loading",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="train the model",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
    )
    _add_train_parser_args(train_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Use a trained model for inference",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
    )
    _add_infer_parser_args(infer_parser)

    return parser


def _add_train_parser_args(train_parser: ArgumentParser) -> None:
    train_parser.add_argument(
        "train_path", type=Path, help="Path of the training dataset"
    )
    train_parser.add_argument(
        "val_path", type=Path, help="Path of the validation dataset"
    )
    train_parser.add_argument(
        "-e", "--epochs", default=10, type=int, help="number of training epochs"
    )
    train_parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        help="learning rate of the optimizer",
    )
    train_parser.add_argument(
        "-s",
        "--num-samples",
        default=5,
        type=int,
        help="Number of samples to log per epoch",
    )
    train_parser.add_argument(
        "-wd",
        "--weight-decay",
        default=1e-6,
        type=float,
        help="weight decay of the optimizer",
    )


def _add_infer_parser_args(infer_parser: ArgumentParser) -> None:
    infer_parser.add_argument(
        "data", type=Path, help="Path of the dataset to perform inference on"
    )
    infer_parser.add_argument("output", type=Path, help="Path to output predictions")


def main(argv: Optional(List[str]) = None) -> int:
    args = _get_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        trainer = Trainer(
            model_name=args.model_name,
            load_path=args.load,
            train_path=args.train_path,
            val_path=args.val_path,
            batch_size=args.batch_size,
            device=device,
            epochs=args.epochs,
            lr=args.learning_rate,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            weight_decay=args.weight_decay,
        )
        trainer.train()
        return 0

    if args.command == "infer":
        infer(
            model_name=args.model_name,
            data=args.data,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            output=args.output,
        )
        return 0

    return 1
