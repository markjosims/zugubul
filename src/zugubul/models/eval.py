from transformers import AutoModel, AutoProcessor
from datasets import Dataset
from evaluate import load
from zugubul.models.dataset import load_dataset_safe
from zugubul.models._metrics import (
    compute_acc,
    compute_cer,
    compute_wer,
    compute_cer_and_wer
)

from typing import Callable, Union, List, Optional

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
        metric: Union[str, List[str]] = 'accuracy',
        label_col: str = 'text',
        input_col: str = 'audio',
) -> None:
    if model_str:
        print('Loading model and processor...')
        model = AutoModel.from_pretrained(model_str)
        processor = AutoProcessor.from_pretrained(model_str)
        model.eval()

    if type(dataset) is str:
        print('Loading dataset...')
        dataset = load_dataset_safe(dataset)

    if type(metric) is str:
        metric = [metric,]
    
    outputs = {}
    def eval_row(row) -> None:
        label = row[label_col]
        input_cell = row[input_col]
        input_dict = processor(input_cell['array'], return_tensors="pt", padding=True, sampling_rate=16000)
        for m in metric:
            calc_m = METRICS[m]
            if funct:
                funct_label = funct(input_dict)
                funct_outs = calc_m(pred=funct_label, label=label)
                outputs\
                    .get('funct', dict())\
                    .get(m, list()).append(funct_outs)
            if model:
                pred = model(**input_dict)
                model_outs = calc_m(pred=pred, label=label)
                outputs\
                    .get('model', dict())\
                    .get(m, list()).append(model_outs)        
    print('Evaluating...')
    dataset.map(eval_row)
    return outputs
    
