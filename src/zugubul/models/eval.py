from transformers import AutoModel, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk, load_metric

import numpy as np
from typing import Callable, Union

def eval(
        model: Union[str, Callable],
        dataset: Union[str, Dataset],
        metric: str = 'accuracy'
) -> None:
    training_args = TrainingArguments("test_trainer")
    
    if type(model) is str:
        model = AutoModel.from_pretrained(model)
    if type(dataset) is str:
        dataset = load_from_disk(dataset)

    metric = load_metric(metric)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    
