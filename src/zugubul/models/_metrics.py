from evaluate import load, EvaluationModule
from transformers import PreTrainedTokenizer
import numpy as np
from typing import Dict, Union, Mapping, Optional
import torch

wer_metric = load("wer")
cer_metric = load("cer")
accuracy = load("accuracy")

def get_pred_str(pred_logits, processor) -> str:
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    return pred_str    

def compute_str_acc(
        pred: Union[Mapping, str, None] = None,
        processor: Optional[PreTrainedTokenizer] = None,
        pred_logits: Optional[torch.tensor] = None,
        label_str: Optional[str] = None,
        metrics: Dict[str, EvaluationModule] = 'wer',
        return_labels: bool = False,
    ):
    if type(pred) is not str:
        if not pred_logits:
            pred_logits = pred.predictions
        pred_str = get_pred_str(pred_logits, processor)

    if label_str is None:
        if hasattr(processor, 'pad_token_id'):
            pad_token_id = processor.pad_token_id
        else:
            pad_token_id = processor.tokenizer.pad_token_id
        pred.label_ids[pred.label_ids == -100] = pad_token_id
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    metric_outs = dict()
    for metric_name, metric_funct in metrics.items():
        metric_outs[metric_name] = metric_funct.compute(
            predictions=pred_str,
            references=label_str
        )
    if return_labels:
        metric_outs['label'] = label_str
        metric_outs['pred'] = pred_str

    return metric_outs

def compute_wer(*args, **kwargs):
    return compute_str_acc(*args, **kwargs, metrics={'wer': wer_metric})

def compute_cer(*args, **kwargs):
    return compute_str_acc(*args, **kwargs, metrics={'cer': cer_metric})

def compute_cer_and_wer(*args, **kwargs):
    return compute_str_acc(*args, **kwargs, metrics={'wer': wer_metric, 'cer': cer_metric})

# taken from https://huggingface.co/docs/transformers/tasks/audio_classification on Sep 12 2023
def compute_acc(pred, return_labels: bool = False):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    out = accuracy.compute(predictions=pred_ids, references=pred.label_ids)
    if return_labels:
        out['label'] = pred.label_ids
        out['pred'] = pred_ids
    return out