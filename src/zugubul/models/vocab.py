from typing import Sequence

from ._unimorph import DELIMITER
from ._unimorph import FEATURES


NUM_UNICODE_SYMBOLS = 149186
PAD_TOKEN = "<PAD>"
START_TOKEN = "<S>"
END_TOKEN = "<E>"
SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN)
PAD_TOKEN_IDX = NUM_UNICODE_SYMBOLS + SPECIAL_TOKENS.index(PAD_TOKEN)
START_TOKEN_IDX = NUM_UNICODE_SYMBOLS + SPECIAL_TOKENS.index(START_TOKEN)
END_TOKEN_IDX = NUM_UNICODE_SYMBOLS + SPECIAL_TOKENS.index(END_TOKEN)
SIZE = NUM_UNICODE_SYMBOLS + len(SPECIAL_TOKENS) + len(FEATURES)


def _try_feature_index(f: str) -> int:
    try:
        return FEATURES.index(f)
    except ValueError:
        return FEATURES.index("UNK")


def encode(to_encode: str, encode_type: str) -> list[int]:
    if encode_type == "characters":
        for i, tok in enumerate(SPECIAL_TOKENS):
            to_encode = to_encode.replace(tok, chr(NUM_UNICODE_SYMBOLS + i))
        return [ord(c) for c in to_encode]
    elif encode_type == "features":
        return [
            _try_feature_index(f) + NUM_UNICODE_SYMBOLS + len(SPECIAL_TOKENS)
            for f in to_encode.split(DELIMITER)
        ]
    else:
        raise ValueError("encode_type must be 'character' or 'features'")


def decode(to_decode: Sequence[int], include_special_tokens: bool = True) -> str:
    decoded: list[str] = []
    for c in to_decode:
        if c < NUM_UNICODE_SYMBOLS:
            decoded.append(chr(c))
        elif c < NUM_UNICODE_SYMBOLS + len(SPECIAL_TOKENS):
            if include_special_tokens:
                decoded.append(SPECIAL_TOKENS[c - NUM_UNICODE_SYMBOLS])
            if c == END_TOKEN_IDX:
                break
        else:
            decoded.append(FEATURES[c - NUM_UNICODE_SYMBOLS - len(SPECIAL_TOKENS)])
    return "".join(decoded)
