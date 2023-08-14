from typing import Sequence, Union, Optional
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from pathlib import Path
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
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            vocab.add(row['text'])
    vocab_path = vocab_from_list(vocab=vocab, vocab_dir=vocab_dir)
    return Wav2Vec2CTCTokenizer(vocab_path)

def init_processor(vocab: Union[str, os.PathLike, Sequence[str]], vocab_dir: Optional[Union[str, os.PathLike]] = None) -> Wav2Vec2Processor:
    """
    vocab may be path to a .csv file, vocab.json file or a list or set containing vocab items.
    vocab_dir is the directory for the vocab.json to be stored (if not already saved).
    """
    if type(vocab) is str and Path(vocab).suffix == '.json':
        tokenizer == Wav2Vec2CTCTokenizer(vocab)
    else:
        if not vocab_dir:
            raise ValueError('If vocab is not a path to a json file, vocab_dir must be passed.')
        if type(vocab) in (list, set):
            tokenizer = tokenizer_from_list(vocab, vocab_dir)
        elif type(vocab) is str and Path(vocab).suffix == '.csv':
            tokenizer = tokenizer_from_csv(vocab, vocab_dir)
        else:
            raise ValueError(
                'vocab argument of unrecognized type. Must be list or set of vocab items, path to vocab.json file, or path to csv file.'
            )
        
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)