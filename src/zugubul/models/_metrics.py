from evaluate import load
import numpy as np

wer_metric = load("wer")
accuracy = load("accuracy")

def compute_wer(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# from https://huggingface.co/docs/transformers/tasks/audio_classification
def compute_acc(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    return accuracy.compute(predictions=pred_ids, references=pred.label_ids)