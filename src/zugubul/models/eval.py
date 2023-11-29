from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import Dataset
from zugubul.models.dataset import load_dataset_safe
from zugubul.models._metrics import (
    compute_acc,
    compute_cer,
    compute_wer,
    compute_cer_and_wer
)
from zugubul.main import init_eval_parser, handle_eval
import torch

from typing import Callable, Union, List, Optional, Sequence
from collections import defaultdict
import argparse

METRICS = {
    'accuracy': compute_acc,
    'cer': compute_cer,
    'wer': compute_wer,
    'cer_and_wer': compute_cer_and_wer,
}

def eval(
        dataset: Union[str, Dataset],
        model_str: Optional[str],
        funct: Optional[Callable] = None,
        metric: Union[str, List[str]] = 'cer_and_wer',
        label_col: str = 'text',
        input_col: str = 'audio',
        split: str = 'test'
) -> None:
    if model_str:
        print('Loading model and processor...')
        model = Wav2Vec2ForCTC.from_pretrained(model_str)
        processor = AutoProcessor.from_pretrained(model_str)
        model.eval()

    if type(dataset) is str:
        print('Loading dataset...')
        dataset = load_dataset_safe(dataset, split=split)

    if type(metric) is str:
        metric = [metric,]
    
    outputs = defaultdict(lambda: defaultdict(list))
    def eval_row(row) -> None:
        label = row[label_col]
        input_cell = row[input_col]
        input_dict = processor(input_cell['array'], return_tensors="pt", padding=True, sampling_rate=16000)
        for m in metric:
            calc_m = METRICS[m]
            if funct:
                funct_label = funct(input_dict)
                funct_outs = calc_m(pred=funct_label, label_str=label, return_labels=True)
                outputs['funct'][m].extend(funct_outs)
            if model:
                with torch.no_grad():
                    pred = model(**input_dict)
                model_outs = calc_m(
                    pred_logits=pred.logits,
                    processor=processor,
                    label_str=label,
                    return_labels=True,
                )
                outputs['model'][m].extend(model_outs)        
    print('Evaluating...')
    dataset.map(eval_row, batched=True)
    return outputs

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Evaluate a HF model on test data.')
    init_eval_parser(parser)
    args = vars(parser.parse_args(argv))

    handle_eval(args)
    return 0

if __name__ == '__main__':
    main()