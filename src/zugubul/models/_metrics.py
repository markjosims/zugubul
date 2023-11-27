from evaluate import load, EvaluationModule
import numpy as np
from typing import Dict

wer_metric = load("wer")
cer_metric = load("cer")
accuracy = load("accuracy")

def compute_str_acc(
        pred,
        processor,
        metrics: Dict[str, EvaluationModule],
    ):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    if hasattr(processor, 'pad_token_id'):
        pad_token_id = processor.pad_token_id
    else:
        pad_token_id = processor.tokenizer.pad_token_id
    pred.label_ids[pred.label_ids == -100] = pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    metric_outs = dict()
    for metric_name, metric_funct in metrics.items():
        metric_outs[metric_name] = metric_funct.compute(
            predictions=pred_str,
            references=label_str
        )

    return metric_outs

def compute_wer(pred, processor):
    return compute_str_acc(pred, processor, metrics={'wer': wer_metric})

def compute_cer(pred, processor):
    return compute_str_acc(pred, processor, metrics={'cer': cer_metric})

def compute_cer_and_wer(pred, processor):
    return compute_str_acc(pred, processor, metrics={'wer': wer_metric, 'cer': cer_metric})

# taken from https://huggingface.co/docs/transformers/tasks/audio_classification on Sep 12 2023
def compute_acc(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    return accuracy.compute(predictions=pred_ids, references=pred.label_ids)