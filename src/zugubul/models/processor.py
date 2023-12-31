import os
from pathlib import Path
from typing import Optional, Sequence, Union
import torch
from dataclasses import dataclass
from transformers import Wav2Vec2Processor
from typing import Sequence, Union, Dict, List

from zugubul.models.vocab import vocab_from_csv, vocab_from_list


def init_processor(
        vocab: Union[str, os.PathLike, Sequence[str]],
        vocab_dir: Optional[Union[str, os.PathLike]] = None,
        lid: bool = False,
    ) -> Wav2Vec2Processor:
    """
    vocab may be path to a .csv file, vocab.json file or a list or set containing vocab items.
    lid is a bool indicating whether processor is being made for language identification or not.
    vocab_dir is the directory for the vocab.json to be stored (if not already saved).
    """
    if type(vocab) in (list, set):
        vocab_path = vocab_from_list(vocab, vocab_dir, lid)
    elif os.path.isfile(vocab) and Path(vocab).suffix == '.json':
        vocab_path = vocab
    elif os.path.isfile(vocab) and Path(vocab).suffix == '.csv':
        vocab_path = vocab_from_csv(vocab, vocab_dir, lid)
    else:
        raise ValueError(
            'vocab argument of unrecognized type. Must be list or set of vocab items, path to vocab.json file, or path to csv file.'
        )

    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token='<unk>', pad_token='<pad>')

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# taken from https://huggingface.co/blog/mms_adapters on 08 August 2023.
@dataclass
class DataCollatorCTC:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
# taken from https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb#scrollTo=ZXVl9qW1Gw_-
# on 13 Sep 2023
@dataclass
class DataCollatorSeqClassification:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch