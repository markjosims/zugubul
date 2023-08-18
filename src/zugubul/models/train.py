from transformers import Trainer, Wav2Vec2ForCTC, DataCollator, TrainingArguments, Wav2Vec2Processor
from datasets import Dataset
from typing import Callable, Optional, Union
import os

from zugubul.models.vocab import DataCollatorCTCWithPadding, init_processor
from zugubul.models._metrics import compute_wer

def train(
        model: Union[str, Wav2Vec2ForCTC],
        dataset: Dataset,
        data_collator: Optional[DataCollator] = None,
        training_args: Optional[TrainingArguments] = None,
        compute_metrics: Optional[Callable] = None,
        vocab: Union[str, os.PathLike, None] = None,
        processor: Optional[Wav2Vec2Processor] = None,
        **kwargs
    ) -> str:

    if not processor:
        if not vocab:
            raise ValueError('Either processor object or path to vocab.json must be provided.')
        processor = init_processor(vocab)

    if type(model) is not Wav2Vec2ForCTC:
        model = download_model(processor, model_name=model, **kwargs)

    if not training_args:
        training_args = get_training_args(kwargs)

    if not data_collator:
        data_collator = DataCollatorCTCWithPadding(processor, padding=True)

    if not compute_metrics:
        # use wer by default
        compute_metrics = lambda pred : compute_wer(pred, processor)

    # taken from https://huggingface.co/blog/mms_adapters
    # prepare model for finetuning
    model.init_adapter_layers()
    model.freeze_base_model()
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        test_dataset=dataset['test'],
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

def get_training_args(**kwargs) -> TrainingArguments:
    """
    Returns TrainingArguments object with any argument values specified in kwargs.
    kwarg 'out_dir' must be provided, all others are optional.
    Default values taken from https://huggingface.co/blog/mms_adapters
    """
    return TrainingArguments(
        output_dir=kwargs['out_dir'],
        group_by_length= kwargs.get('group_by_length',                          True),
        per_device_train_batch_size= kwargs.get('per_device_train_batch_size',  32),
        evaluation_strategy= kwargs.get('evaluation_strategy',                  "steps"),
        num_train_epochs= kwargs.get('num_train_epochs',                        4),
        gradient_checkpointing= kwargs.get('gradient_checkpointing',            True),
        fp16= kwargs.get('fp16',                                                True),
        save_steps= kwargs.get('save_steps',                                    200),
        eval_steps= kwargs.get('eval_steps',                                    100),
        logging_steps= kwargs.get('logging_steps',                              100),
        learning_rate= kwargs.get('learning_rate',                              1e-3),
        warmup_steps= kwargs.get('warmup_steps',                                100),
        save_total_limit= kwargs.get('save_total_limit',                        2),
        push_to_hub= kwargs.get('push_to_hub',                                  True),
    )

def download_model(
        processor: Wav2Vec2Processor,
        model_name: str = "facebook/mms-1b-all",
        **kwargs
    ) -> Wav2Vec2ForCTC:
    """
    Opens Wav2Vec2ForCTC model at given url or path.
    Default parameter values taken from https://huggingface.co/blog/mms_adapters
    """
    return Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=kwargs.get('attention_dropout',               0.0),
        hidden_dropout=kwargs.get('hidden_dropout',                     0.0),
        feat_proj_dropout=kwargs.get('feat_proj_dropout',               0.0),
        layerdrop=kwargs.get('layerdrop',                               0.0),
        ctc_loss_reduction=kwargs.get('ctc_loss_reduction',             "mean"),
        pad_token_id=kwargs.get('pad_token_id',                         processor.tokenizer.pad_token_id),
        vocab_size=kwargs.get('vocab_size',                             len(processor.tokenizer)),
        ignore_mismatched_sizes=kwargs.get('ignore_mismatched_sizes',   True),
    )