from typing import Sequence, Union, Optional, Dict
import os
import csv
import json


def vocab_from_list(
        vocab: Sequence[str],
        vocab_dir: Union[str, os.PathLike],
        lid: bool = False,
    ) -> str:
    """
    vocab is a list of strings containing tokens to include in vocabulary.
    vocab_dir is folder to save vocab.json in.
    lid is a bool indicating whether the vocabulary is being made for language identification or not.
    If True, add each whole item in list to vocab.
    If False, add each character from each item in list to vocab.
    Returns path to vocab.json
    """
    if lid:
        tokens = set(vocab)
    else:
        tokens = set(c for v in vocab for c in v)
    if ' ' in tokens:
        tokens.remove(' ')
    tokens_dict = {k: v for v, k in enumerate(tokens)}

    if not lid:
        # add special tokens (ASR only)
        tokens_dict['<pad>'] = len(tokens_dict)
        tokens_dict['<unk>'] = len(tokens_dict)

    json_path = os.path.join(vocab_dir, 'vocab.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tokens_dict, f)
    return json_path

def vocab_from_csv(
        csv_path: Union[str, os.PathLike],
        vocab_dir: Union[str, os.PathLike],
        lid: bool = False,
    ) -> str:
    """
    vocab is a list of strings containing tokens to include in vocabulary.
    vocab_dir is folder to save vocab.json in.
    lid is a bool indicating whether tokenizer is being made for language identification or not.
    Returns path to vocab.json
    """
    vocab = set()
    label_col = 'text'
    if lid:
        label_col = 'lang'
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            vocab.add(row[label_col])
    return vocab_from_list(vocab=vocab, vocab_dir=vocab_dir, lid=lid)

def make_lm_vocab(text: str, initial_vocab: Optional[Dict[str]] = None) -> dict:
    """
    Returns a dictionary containing the vocab for a given LM dataset.
    If passed initial_vocab, only adds what chars are not already present.
    """
    unique_chars = set(text)
    vocab = {}
    if initial_vocab:
        vocab = initial_vocab
    for c in unique_chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab