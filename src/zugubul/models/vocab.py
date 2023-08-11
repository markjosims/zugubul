from typing import Sequence, Union
from transformers import Wav2Vec2CTCTokenizer
import os
import csv
import json

def vocab_from_list(vocab: Sequence[str], vocab_dir: Union[str, os.PathLike]) -> str:
    """
    vocab is a list of strings containing tokens to include in vocabulary.
    vocab_dir is folder to save vocab.json in.
    Returns path to vocab.json
    """
    vocab_dict = {k: v for v, k in enumerate(vocab)}
    json_path = os.path.join(vocab_dir, 'vocab.json')
    with open(json_path, 'w') as f:
        json.dump(vocab_dict, f)
    return json_path

def tokenizer_from_list(vocab: Sequence[str], vocab_dir: Union[str, os.PathLike]) -> Wav2Vec2CTCTokenizer:
    """
    vocab is a list of strings containing tokens to include in vocabulary.
    vocab_dir is folder to save vocab.json in.
    Returns Wav2Vec2CTCTokenizer object.
    """
    vocab_path = vocab_from_list(vocab=vocab, vocab_dir=vocab_dir)
    return Wav2Vec2CTCTokenizer(vocab_path)

def tokenizer_from_csv(
        csv_path: Union[str, os.PathLike],
        vocab_dir: Union[str, os.PathLike],
    ) -> Wav2Vec2CTCTokenizer:
    """
    vocab is a list of strings containing tokens to include in vocabulary.
    vocab_dir is folder to save vocab.json in.
    Returns Wav2Vec2CTCTokenizer object.
    """
    vocab = set()
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            vocab.add(row[0])
    vocab_path = vocab_from_list(vocab=vocab, vocab_dir=vocab_dir)
    return Wav2Vec2CTCTokenizer(vocab_path)