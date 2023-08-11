import csv
from pathlib import Path
from typing import NamedTuple
from typing import Sequence

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from zugubul.models._constants import CHARACTER_TYPE
from zugubul.models._constants import FEATURE_TYPE
from zugubul.models._constants import SPECIAL_TOKEN_TYPE
from zugubul.models.models import LangIdentifierInput
from zugubul.models.vocab import encode
from zugubul.models.vocab import END_TOKEN
from zugubul.models.vocab import END_TOKEN_IDX
from zugubul.models.vocab import PAD_TOKEN_IDX
from zugubul.models.vocab import START_TOKEN
from zugubul.models.vocab import START_TOKEN_IDX


class RawDatum(NamedTuple):
    """
    The unprocessed features of single training example

    Attributes
    ----------
    lemma: The input lemma to be inflected
    word: The correctly inflected word
    features: The grammatical features determining the inflection
    """

    lemma: str
    word: str
    features: str


class Dataset(data.Dataset):
    def __init__(self, path: Path) -> None:
        with open(path) as csvfile:
            r = csv.reader(csvfile, delimiter="\t")
            self.data = [RawDatum(line[0], line[1], line[2]) for line in r]

    def __getitem__(self, index: int) -> LangIdentifierInput:
        feature_tokens = torch.tensor(encode(self.data[index].features, "features"))
        lemma_tokens = torch.tensor(encode(self.data[index].lemma, "characters"))

        src_tokens = torch.cat(
            [
                torch.tensor([START_TOKEN_IDX]),
                feature_tokens,
                lemma_tokens,
                torch.tensor([END_TOKEN_IDX]),
            ]
        )
        src_types = torch.tensor(
            [SPECIAL_TOKEN_TYPE]
            + [FEATURE_TYPE for _ in range(len(feature_tokens))]
            + [CHARACTER_TYPE for _ in range(len(lemma_tokens))]
            + [SPECIAL_TOKEN_TYPE]
        )
        tgt_tokens = torch.tensor(
            encode(START_TOKEN + self.data[index].word + END_TOKEN, "characters")
        )
        tgt_tokens_in = tgt_tokens[:-1]
        tgt_tokens_out = tgt_tokens[1:]
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt_tokens_in.size(0)
        )
        return LangIdentifierInput(
            src_tokens, src_types, tgt_tokens_in, tgt_tokens_out, tgt_mask
        )

    def __len__(self) -> int:
        return len(self.data)


def collate_fn(batch: Sequence[LangIdentifierInput]) -> LangIdentifierInput:
    src_tokens = pad_sequence(
        [item.src_tokens for item in batch], padding_value=PAD_TOKEN_IDX
    )
    src_types = pad_sequence(
        [item.src_types for item in batch], padding_value=SPECIAL_TOKEN_TYPE
    )
    tgt_tokens_in = pad_sequence(
        [item.tgt_tokens_in for item in batch], padding_value=PAD_TOKEN_IDX
    )
    tgt_tokens_out = pad_sequence(
        [item.tgt_tokens_out for item in batch], padding_value=PAD_TOKEN_IDX
    )
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        tgt_tokens_in.size(0)
    )

    return LangIdentifierInput(
        src_tokens, src_types, tgt_tokens_in, tgt_tokens_out, tgt_mask
    )
