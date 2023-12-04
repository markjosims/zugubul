from transformers import Trainer, Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification,\
    Wav2Vec2Model, DataCollator, TrainingArguments, Wav2Vec2Processor, Wav2Vec2FeatureExtractor,\
    AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer, CanineTokenizer,\
    AutoConfig, GPT2LMHeadModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from zugubul.models.models import CanineForMaskedLM
# TODO: use Auto objects across the board
from datasets import Dataset, DatasetDict, Audio, load_dataset
from huggingface_hub import login, hf_hub_download, HfFolder
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
#import bitsandbytes as bnb

from typing import Callable, Optional, Union, Literal, Sequence
import os
import json
import argparse
from zugubul.models.processor import init_processor, DataCollatorCTC, DataCollatorSeqClassification
from zugubul.models.vocab import make_lm_vocab
from zugubul.models._metrics import compute_cer_and_wer, compute_acc
from zugubul.main import init_train_parser, handle_train, DEFAULT_HYPERPARAMS


def train(
        out_dir: Union[str, os.PathLike],
        model_str: Union[str, os.PathLike],
        dataset: Union[str, os.PathLike, Dataset],
        device_map: str = "auto",
        label_col: Optional[str] = None,
        data_collator: Optional[DataCollator] = None,
        training_args: Optional[TrainingArguments] = None,
        audio_cutoff: Optional[int] = None,
        optimizers = (None, None),
        task: Literal['LID', 'ASR', 'LM'] = 'ASR',
        compute_metrics: Optional[Callable] = None,
        vocab: Union[str, os.PathLike, None] = None,
        processor: Union[Wav2Vec2Processor, Wav2Vec2FeatureExtractor, None] = None,
        save_eval_preds: bool = False,
        hf: bool = True,
        **kwargs
    ) -> str:

    if task == 'LM':
        return train_lm(
            dataset=dataset,
            out_dir=out_dir,
            checkpoint=model_str,
            label_col=label_col or 'text',
            device_map=device_map,
            **kwargs,
        )

    if hf:
        token = HfFolder.get_token()
        while not token:
            login()
            token = HfFolder.get_token()

    if (not processor) and (task in ['ASR', 'LM']):
        vocab = _get_vocab_path(vocab, dataset, hf)
        print('Initializing processor...')
        processor = init_processor(vocab, vocab_dir=dataset, task=task)
    elif (not processor) and (task == 'LID'):
        print('Downloading feature extractor...')
        processor = Wav2Vec2Processor.from_pretrained(model_str)

    print('Downloading model...')
    if task == 'ASR':
        print('Instantiating model as Wav2Vec2ForCTC for ASR.')
        model_wrapper = Wav2Vec2ForCTC
        model = download_model(
            model_name=model_str,
            model_wrapper=model_wrapper,
            task=task,
            processor=processor,
        )
    else:
        print('Instantiating model as Wav2Vec2ForSequenceClassification for LID.')
        model_wrapper = Wav2Vec2ForSequenceClassification
        vocab = _get_vocab_path(vocab, dataset, hf)
        with open(vocab, 'r') as f:
            vocab = json.load(f)
        label2id = vocab
        id2label = {id: label for label, id in vocab.items()}
        model = download_model(
            model_name=model_str,
            model_wrapper=model_wrapper,
            task=task,
            num_labels=len(vocab),
            label2id=label2id,
            id2label=id2label,
        )

    print('Preparing model for finetuning...')
    if 'mms' in model_str:
    # taken from https://huggingface.co/blog/mms_adapters
    # freeze non adapter parameters
        if hasattr(model, 'init_adapter_layers'):
            model.init_adapter_layers()
            model.freeze_base_model()
            adapter_weights = model._get_adapters()
            for param in adapter_weights.values():
                param.requires_grad = True
    else:
        # other wav2vec2 models like XLSR do not have adapter layers
        # instead, we freeze the feature encoder
        model.freeze_feature_encoder()

    if not training_args:
        training_args = get_training_args(
            output_dir=out_dir,
            push_to_hub=hf,
            **kwargs
        )

    if type(dataset) is not Dataset:
        if os.path.exists(dataset) and os.path.exists(
            os.path.join(dataset, 'dataset_dict.json')
        ):
            print('Loading dataset from local file...')
            dataset = DatasetDict.load_from_disk(dataset)
        else:
            print('Loading dataset from Huggingface hub (or cache)...')
            dataset = load_dataset(dataset)

    print('Resampling audio to 16kHz...')
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    if audio_cutoff is not None:
        print(f'Removing audio records with greater than {audio_cutoff} frames...')
        dataset = dataset.filter(lambda r: r['audio']['array'].shape[0]<audio_cutoff)

    print(f'Reshaping data columns for training {task}...')
    if not label_col:
        label_col = 'lang' if task=='LID' else 'text'
    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor, label_col, task, vocab),
        remove_columns=dataset['train'].column_names
    )

    if not data_collator:
        print('Initializing data collator...')
        collator_obj = DataCollatorCTC if task=='ASR'\
            else DataCollatorForLanguageModeling if task=='LM'\
            else DataCollatorSeqClassification
        data_collator = collator_obj(processor)

    if not compute_metrics:
        eval_step = 0
        def compute_metrics(pred):
            if task == 'ASR':
                out = compute_cer_and_wer(pred, processor, return_labels=save_eval_preds)
            else:
                out = compute_acc(pred, return_labels=save_eval_preds)
            if save_eval_preds:
                label, pred = out.pop('label'), out.pop('pred')
                nonlocal eval_step
                eval_step+=training_args.eval_steps
                if not os.path.exists(training_args.logging_dir):
                    os.makedirs(training_args.logging_dir)
                eval_fpath = os.path.join(training_args.logging_dir, f"eval_{eval_step}.txt")
                with open(eval_fpath, 'a') as f:
                    json.dump({'label': label, 'pred': pred}, f)
            return out

    print('Starting training...')
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=processor.feature_extractor
    )
    trainer.train()
    print('Done training')
    trainer.evaluate(dataset['test'])

    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE#.format(target_lang)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

    if hf:
        trainer.push_to_hub()
        processor.push_to_hub()
    else:
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)

def train_lm(
       dataset: Union[str, Dataset],
       out_dir: str,
       checkpoint: str = 'gpt2',
       label_col: str = 'text',
       context_length: int = 128,
       device_map: str = 'auto',
       **kwargs
) -> None:
    """
    Uses character-based tokenization.
    """
    print('Loading dataset...')
    dataset = DatasetDict.load_from_disk(dataset)

    print('Identifying unique characters...')
    unique_chars = set()
    dataset.map(lambda r: unique_chars.update(r[label_col]))

    print('Creating tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    num_special_tokens = len(tokenizer.all_special_tokens)
    len_vocab = len(unique_chars) + num_special_tokens
    tokenizer = tokenizer.train_new_from_iterator(unique_chars, vocab_size=len_vocab)

    print('Generating config for model...')
    config = AutoConfig.from_pretrained(
        checkpoint,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        device_map=device_map,
    )
    print(f'Initializing GPT2LMHeadModel from checkpoint {checkpoint}...')
    model = GPT2LMHeadModel(config)

    print('Tokenizing dataset...')
    dataset = prepare_lm_dataset(dataset, tokenizer, label_col, context_length)

    print('Defining training arguments...')
    training_arguments = get_training_args(output_dir=out_dir, **kwargs)

    print('Initializing data collator')
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print('Beginning training...')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=lambda pred : compute_cer_and_wer(pred, tokenizer),
    )
    trainer.train()
    trainer.evaluate(dataset['test'])
    trainer.save_model(training_arguments.output_dir)
    tokenizer.save_pretrained(training_arguments.output_dir)


def _get_vocab_path(vocab: Union[str, os.PathLike, None], dataset: str, hf: bool) -> dict:
    if hf:
        vocab = hf_hub_download(
            repo_id=dataset,
            repo_type='dataset',
            filename=vocab or 'vocab.json'
        )
    elif os.path.isfile(vocab):
        return vocab
    elif os.path.isfile(os.path.join(dataset, vocab)):
        return os.path.join(dataset, vocab)
    raise ValueError('Either processor object or path to vocab.json must be provided if not loading from HuggingFace.')


def get_training_args(**kwargs) -> TrainingArguments:
    """
    Returns TrainingArguments object with any argument values specified in kwargs.
    kwarg 'out_dir' must be provided, all others are optional.
    Default values taken from https://huggingface.co/blog/mms_adapters
    """
    for k, v in DEFAULT_HYPERPARAMS.items():
        if k not in kwargs:
            kwargs[k] = v

    return TrainingArguments(**kwargs)

def download_model(
        model_name: str = "facebook/mms-1b-all",
        model_wrapper: Wav2Vec2Model = Wav2Vec2Model,
        task: Literal['LID', 'ASR', 'LM'] = 'ASR',
        processor: Optional[Wav2Vec2Processor] = None,
        **kwargs
    ) -> Union[Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification]:
    """
    Opens Wav2Vec2 model at given url or path.
    Default parameter values for ASR taken from https://huggingface.co/blog/mms_adapters
    """
    if task == 'ASR':
        default_values = {
            'attention_dropout': 0.0,
            'hidden_dropout': 0.0,
            'feat_proj_dropout': 0.0,
            'layerdrop': 0.0,
            'mask_time_prob': 0.05,
            'ctc_loss_reduction': "mean",
            'ignore_mismatched_sizes': True,
        }
        if processor:
            default_values['pad_token_id'] = processor.tokenizer.pad_token_id
            default_values['vocab_size'] = len(processor.tokenizer)
        for k, v in default_values.items():
            if k not in kwargs:
                kwargs[k] = v
    return model_wrapper.from_pretrained(
        model_name,
        **kwargs
    )

def prepare_dataset(
        batch: Dataset,
        processor: Union[Wav2Vec2Processor, AutoTokenizer],
        label_col: str,
        task: Literal['ASR', 'LID', 'LM'],
        label2id: Optional[dict] = None,
    ) -> Dataset:
    """
    Puts dataset in format needed for training.
    Code for asr taken from https://huggingface.co/blog/mms_adapters
    TODO: make separate methods for ASR, LID and LM
    """
    if task == 'LM':
        return processor(batch[label_col])

    audio=batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    if task == 'ASR':
        batch["labels"] = processor(text=batch[label_col].strip()).input_ids
    else:
        if not label2id:
            raise ValueError("label2id must be passed for LID task.")
        batch["labels"] = [label2id.get(row[label_col], -1) for row in batch]
    return batch

def prepare_lm_dataset(
        data: Union[Dataset, dict, str],
        tokenizer: AutoTokenizer,
        label_col: str = 'text',
        context_length: int = 128,
) -> dict:
    """
    Code based off tutorial in https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    """
    if type(data) is str:
        data = DatasetDict.load_from_disk(data)
    def map_tokenizer(batch):
        outputs = tokenizer(
            batch[label_col],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    return data.map(map_tokenizer, batched=True, remove_columns=data['train'].column_names)
    

# TODO: reimplement this
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

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Train Language IDentification (LID) or Automatic Speech Recognition (ASR) model on given dataset.')
    init_train_parser(parser)
    args = vars(parser.parse_args(argv))

    handle_train(args)
    return 0

    

if __name__ == '__main__':
    main()