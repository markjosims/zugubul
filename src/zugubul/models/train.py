from transformers import Trainer, Wav2Vec2ForCTC, DataCollator, TrainingArguments, Wav2Vec2Processor
from datasets import Dataset, Audio, load_dataset
from huggingface_hub import login, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
#import bitsandbytes as bnb
import torch
from transformers.trainer_pt_utils import get_parameter_names

from typing import Callable, Optional, Union, Tuple
import os

from zugubul.models.vocab import DataCollatorCTCWithPadding, init_processor
from zugubul.models._metrics import compute_wer

def train(
        out_dir: Union[str, os.PathLike],
        model: Union[str, os.PathLike, Wav2Vec2ForCTC],
        dataset: Union[str, os.PathLike, Dataset],
        data_collator: Optional[DataCollator] = None,
        training_args: Optional[TrainingArguments] = None,
        optimizers = (None, None),
        compute_metrics: Optional[Callable] = None,
        vocab: Union[str, os.PathLike, None] = None,
        processor: Optional[Wav2Vec2Processor] = None,
        hf: bool = True,
        **kwargs
    ) -> str:

    if hf:
        login()

    if not processor:
        if hf:
            vocab = hf_hub_download(
                repo_id=dataset,
                repo_type='dataset',
                filename='vocab.json'
            )
        if (not hf) and (not vocab):
            raise ValueError('Either processor object or path to vocab.json must be provided.')

        print('Initializing processor...')
        processor = init_processor(vocab)

    if type(model) is not Wav2Vec2ForCTC:
        print('Downloading model...')
        model = download_model(processor, model_name=model, **kwargs)

    print('Preparing model for finetuning...')
    # taken from https://huggingface.co/blog/mms_adapters
    # prepare model for finetuning
    model.init_adapter_layers()
    model.freeze_base_model()
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    if not training_args:
        if ('optim' in kwargs) and (kwargs['optim']=='adam8bit'):
            training_args = get_training_args(
                output_dir=out_dir,
                push_to_hub=hf,
                # experimenting with hyperparameters to reduce memory footprint
                per_device_train_batch_size=1,
                save_steps=50,
                fp16=False,
                torch_compile=True,
                **kwargs
            )
            optimizers = (get_adam8bit_optimizer(training_args, model), None)
        else: training_args = get_training_args(output_dir=out_dir, push_to_hub=hf, **kwargs)

    if type(dataset) is not Dataset:
        print('Loading dataset...')
        dataset = load_dataset(dataset)
    print('Resampling audio to 16kHz...')
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    print('Reshaping data columns for training...')
    dataset = dataset.map(lambda x: prepare_dataset(x, processor, 'lang'))

    if not data_collator:
        print('Initializing data collator...')
        data_collator = DataCollatorCTCWithPadding(processor, padding=True)

    if not compute_metrics:
        # use wer by default
        compute_metrics = lambda pred : compute_wer(pred, processor)

    print('Starting training...')
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    print('Done training')

    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE#.format(target_lang)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

    if hf:
        trainer.push_to_hub(private=True)


def get_training_args(**kwargs) -> TrainingArguments:
    """
    Returns TrainingArguments object with any argument values specified in kwargs.
    kwarg 'out_dir' must be provided, all others are optional.
    Default values taken from https://huggingface.co/blog/mms_adapters
    """
    default_values = {
        'group_by_length': True,
        'per_device_train_batch_size': 32,
        'evaluation_strategy': "steps",
        'num_train_epochs': 4,
        'gradient_checkpointing': True,
        'fp16': True,
        'save_steps': 100,
        'eval_steps': 100,
        'logging_steps': 100,
        'learning_rate': 1e-3,
        'warmup_steps': 100,
        'save_total_limit': 2,
        'torch_compile': False,
        'push_to_hub': False,
    }
    for k, v in default_values.items():
        if k not in kwargs:
            kwargs[k] = v
    return TrainingArguments(**kwargs)

def download_model(
        processor: Wav2Vec2Processor,
        model_name: str = "facebook/mms-1b-all",
        **kwargs
    ) -> Wav2Vec2ForCTC:
    """
    Opens Wav2Vec2ForCTC model at given url or path.
    Default parameter values taken from https://huggingface.co/blog/mms_adapters
    """
    default_values = {
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'feat_proj_dropout': 0.0,
        'layerdrop': 0.0,
        'ctc_loss_reduction': "mean",
        'pad_token_id': processor.tokenizer.pad_token_id,
        'vocab_size': len(processor.tokenizer),
        'ignore_mismatched_sizes': True,
    }
    for k, v in default_values.items():
        if k not in kwargs:
            kwargs[k] = v
    return Wav2Vec2ForCTC.from_pretrained(
        model_name,
        **kwargs
    )

def prepare_dataset(batch: Dataset, processor: Wav2Vec2Processor, label_col: str) -> Dataset:
    """
    Puts dataset in format needed for training.
    Taken from https://huggingface.co/blog/mms_adapters
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch[label_col]).input_ids
    return batch

# def get_adam8bit_optimizer(training_args: TrainingArguments, model: Wav2Vec2ForCTC) -> bnb.optim.Adam8bit:
#     """
#     Returns adam 8bit optimizer for given training args and model.
#     Code from https://huggingface.co/docs/transformers/perf_train_gpu_one
#     """
#     decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
#     decay_parameters = [name for name in decay_parameters if "bias" not in name]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if n in decay_parameters],
#             "weight_decay": training_args.weight_decay,
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
#             "weight_decay": 0.0,
#         },
#     ]

#     optimizer_kwargs = {
#         "betas": (training_args.adam_beta1, training_args.adam_beta2),
#         "eps": training_args.adam_epsilon,
#     }
#     optimizer_kwargs["lr"] = training_args.learning_rate
#     adam_bnb_optim = bnb.optim.Adam8bit(
#         optimizer_grouped_parameters,
#         betas=(training_args.adam_beta1, training_args.adam_beta2),
#         eps=training_args.adam_epsilon,
#         lr=training_args.learning_rate,
#     )
#     return adam_bnb_optim