import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from zugubul.models._constants import FEATURE_TYPE
from zugubul.models._constants import TOKEN_TYPES
# from zugubul.models.vocab import PAD_TOKEN_IDX
# from zugubul.models.vocab import SIZE
PAD_TOKEN_IDX=-1 # implement
SIZE=-1 # implement


D_MODEL = 256
N_HEAD = 4
N_LAYERS = 4
DIM_FEEDFORWARD = 1024
POSITIONAL_ENCODING_N = 10000


@dataclass
class LangIdentifierInput:
    """
    The processed features that are input to the model

    Attributes
    ----------
    src_tokens: Input sequence of tokens
    src_types: The types (character or unimorph feature) of each token
    tgt_tokens_in: Target sequence of tokens
    tgt_tokens_out: Target sequence of tokens
    tgt_mask: Mask over target sequence
    """

    src_tokens: torch.Tensor
    src_types: torch.Tensor
    tgt_tokens_in: torch.Tensor
    tgt_tokens_out: torch.Tensor
    tgt_mask: torch.Tensor

    def to(self, device: torch.device) -> None:
        self.src_tokens = self.src_tokens.to(device)
        self.src_types = self.src_types.to(device)
        self.tgt_tokens_in = self.tgt_tokens_in.to(device)
        self.tgt_tokens_out = self.tgt_tokens_out.to(device)
        self.tgt_mask = self.tgt_mask.to(device)


class LangIdentifier(nn.Module):
    """
    Sequence to sequence transformer over characters for generating inflections

    From: https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.src_token_emb = nn.Embedding(SIZE, D_MODEL)
        self.src_types_emb = nn.Embedding(len(TOKEN_TYPES), D_MODEL)
        self.tgt_token_emb = nn.Embedding(SIZE, D_MODEL)
        self.pe = _PositionalEncoding(D_MODEL)
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_encoder_layers=N_LAYERS,
            num_decoder_layers=N_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
        )
        self.generator = nn.Linear(D_MODEL, SIZE)

    def forward(self, model_input: LangIdentifierInput) -> torch.Tensor:
        src_emb = (
            self.src_token_emb(model_input.src_tokens)
            + self.pe(model_input.src_tokens, model_input.src_types == FEATURE_TYPE)
            + self.src_types_emb(model_input.src_types)
        )
        tgt_emb = self.tgt_token_emb(model_input.tgt_tokens_in) + self.pe(
            model_input.tgt_tokens_in
        )
        outs = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=model_input.tgt_mask,
            src_key_padding_mask=(model_input.src_tokens == PAD_TOKEN_IDX).transpose(
                0, 1
            ),
            tgt_key_padding_mask=(model_input.tgt_tokens_in == PAD_TOKEN_IDX).transpose(
                0, 1
            ),
            memory_key_padding_mask=(model_input.src_tokens == PAD_TOKEN_IDX).transpose(
                0, 1
            ),
        )
        return self.generator(outs)


class _PositionalEncoding(nn.Module):
    """
    A positional encoding module used in the transformer.
    Additionally uses masking to set specific locations to 0 position

    From https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, d_model, 2) * math.log(POSITIONAL_ENCODING_N) / d_model
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pe = self.pe[: x.size(0)]
        if mask is not None:
            pe = pe.repeat(1, mask.size(1), 1)
            pe[mask] = self.pe[0]
        return pe


MODELS = {"LangIdentifier": LangIdentifier}
