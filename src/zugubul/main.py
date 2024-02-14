#!/usr/bin/env python

"""
Usage: main COMMAND ARGS...
If zugubul package installed, zugubul COMMAND ARGS...
Run main/zugubul -h for more information.
"""

docstr = """
Runs rVAD_fast (Copyright (c) 2022 Zheng-Hua Tan and Achintya Kumar Sarkar)
on WAV_FILEPATH then generates elan annotations of speech segments at EAF_FILEPATH
If both EAF_FILEPATH and EAF_OUT_FILEPATH are provided,
merges the generated Eaf object with a preexisting eaf at EAF_FILEPATH,
using the provided OVERLAP value (or default 200ms), see documentation at elan_tools.py.
If -r flag is provided, and WAV_FILEPATH is a directory, search for wavs recursively in the directory.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import argparse
from typing import Optional, Sequence, Callable, Dict, Any
from zugubul.utils import is_valid_file, file_in_valid_dir, is_valid_dir, batch_funct, eaf_to_file_safe
from tqdm import tqdm
from pympi import Elan
import importlib_resources
import importlib
import json

TORCH = importlib.util.find_spec('torch') is not None
GOOEY = importlib.util.find_spec('gooey_tools') is not None
PYSIMPLEGUI = importlib.util.find_spec('PySimpleGUIWx') is not None
GUI = os.environ.get('GUI', 0)

if not GOOEY:
    # hacky way of avoiding calling gooey_tools
    # TODO: clean this up
    def innocent_wrapper(f=None, **_):
        if not callable(f):
            return lambda f: innocent_wrapper(f)
        return f
    HybridGooey = innocent_wrapper
    HybridGooeyParser = argparse.ArgumentParser
    def add_arg_nogui(parser, *args, **kwargs):
        kwargs.pop('widget', None)
        if kwargs.get('action', None) in ('store_true', 'store_false'):
            kwargs.pop('metavar', None)
        return parser.add_argument(*args, **kwargs)
    add_hybrid_arg = add_arg_nogui
else:
    from gooey_tools import HybridGooey, HybridGooeyParser, add_hybrid_arg


DEFAULT_HYPERPARAMS = {
    'group_by_length': True,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 1,
    'evaluation_strategy': "steps",
    'num_train_epochs': 4,
    'gradient_checkpointing': True,
    'fp16': False,
    'save_steps': 5000,
    'eval_steps': 1000,
    'logging_steps': 500,
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'torch_compile': False,
    'report_to': 'tensorboard',
}

DEFAULT_VAD_PARAMS = {
    "threshold": 0.5,
    "sampling_rate": 16000,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float('inf'),
    "min_silence_duration_ms": 100,
    "window_size_samples": 512,
    "speech_pad_ms": 30,
    "visualize_probs": False,
}

def init_merge_parser(merge_parser: argparse.ArgumentParser):
    add_arg = lambda *args, **kwargs: add_hybrid_arg(merge_parser, *args, **kwargs)
    add_arg(
        'EAF_SOURCE',
        type=lambda x: is_valid_file(merge_parser, x),
        widget='FileChooser',
        help='Filepath for eaf file to take output from (or parent directory of eafs).'
    )
    add_arg(
        'EAF_MATRIX',
        type=lambda x: is_valid_file(merge_parser, x),
        widget='FileChooser',
        help='Filepath for eaf file to insert annotations into (or parent directory of eafs). '
    )
    add_arg(
        '-o',
        '--eaf_out',
        type=lambda x: file_in_valid_dir(merge_parser, x),
        widget='FileChooser',
        help='Filepath for eaf file to output to (or parent directory of eafs).'
    )
    add_arg(
        '-t',
        '--tier',
        required=False,
        nargs='+',
        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
            +'For multiple tiers write tier names separated by space.'
    ) 
    add_arg(
        '--overlap_behavior',
        choices=['keep_source', 'keep_matrix', 'keep_both'],
        default='keep_source', nargs='?',
        help='Behavior for treating segments that overlap between the two .eafs.\n'
            +'If keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE.\n'\
            +'If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX.\n'\
            +'If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not.\n'\
            +'Default behavior is keep_source'
    )
    add_batch_args(merge_parser)
    
def init_trim_parser(trim_parser: argparse.ArgumentParser):
    add_arg = lambda *args, **kwargs: add_hybrid_arg(trim_parser, *args, **kwargs)
    add_arg(
        'EAF_FILEPATH',
        type=lambda x: is_valid_file(trim_parser, x),
        widget='FileChooser',
        help='Filepath for eaf file to trim (or parent directory of eafs).'
    )
    add_arg(
        '-o',
        '--eaf_out',
        type=lambda x: file_in_valid_dir(trim_parser, x),
        widget='FileChooser',
        help='Filepath for eaf file to output to (or parent directory of eafs).'
    )
    add_arg(
        '-t',
        '--tier',
        required=False, nargs='+',
        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
            +'For multiple tiers write tier names separated by space.'
    ) 
    add_arg(
        '-s',
        '--stopword',
        type=str,
        default='',
        help='Remove all annotations with this value. By default removes all annotations with no text.'
    ) 
    add_batch_args(trim_parser)

def init_vad_parser(vad_parser: argparse.ArgumentParser):
    add_arg = lambda *args, **kwargs: add_hybrid_arg(vad_parser, *args, **kwargs)
    add_arg(
        'WAV_FILEPATH',
        type=lambda x: is_valid_file(vad_parser, x),
        widget='FileChooser',
        help='Filepath for wav file to be processed (or parent directory).'
    )
    add_arg(
        'EAF_FILEPATH',
        type=lambda x: file_in_valid_dir(vad_parser, x),
        widget='FileChooser',
        help='Filepath for eaf file to output to (or parent directory).'
    )
    add_arg(
        '-s',
        '--source',
        type=lambda x: is_valid_file(vad_parser, x),
        widget='FileChooser',
        help='Source .eaf file to merge output into (or parent directory).'
    )
    # add_arg('-m', '--merge_output', help='Filepath to output  to merge output into (or parent directory).')
    add_arg(
        '--template',
        type=lambda x: is_valid_file(vad_parser, x),
        widget='FileChooser',
        help='Template .etf file for generating output .eafs.'
    )
    add_arg('--overlap_behavior', choices=['keep_source', 'keep_matrix', 'keep_both'], default='keep_source', nargs='?',
        help='Behavior for treating segments that overlap between the two .eafs. '
            +'If keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE. '\
            +'If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
            +'If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not. '\
    )
    add_arg(
        '-v',
        '--vad',
        type=lambda x: is_valid_file(vad_parser, x),
        widget='FileChooser',
        help='Filepath for .vad file linked to the recording. '\
            +'If passed, reads VAD data from file instead of running rVAD_fast.'
    )
    add_arg(
        '-t',
        '--tier',
        nargs='+',
        help='Tier label(s) to add annotations to. If none specified uses `default-lt`',
        default='default-lt'
    )
    silero_args = vad_parser.add_argument_group("silero_params", "Parameters used by Silero VAD")
    add_args_from_dict(DEFAULT_VAD_PARAMS, silero_args)

    add_batch_args(vad_parser)

def init_eaf_data_parser(eaf_data_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(eaf_data_parser, *args, **kwargs)
    add_arg(
        'EAF_FILEPATH',
        type=lambda x: is_valid_file(eaf_data_parser, x),
        help='.eaf file to process metadata for.'
    )
    add_arg(
        '-o',
        '--output',
        type=lambda x: is_valid_file(eaf_data_parser, x),
        widget='FileChooser',
        help='.csv path to output to. Default behavior is to use same filename as EAF_FILEPATH, '\
            +'or if EAF_FILEPATH is a directory, create a file named eaf_data.csv in the given directory.'
    )
    add_arg(
        '-t',
        '--tier',
        nargs='+',
        help='Tier label(s) to read annotation data from. If none specified read from all tiers.'
    )
    add_arg(
        '-m',
        '--media',
        help='Path to media file, if different than media path specified in .eaf file.'
    )
    add_batch_args(eaf_data_parser)

def init_split_data_parser(split_data_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(split_data_parser, *args, **kwargs)
    add_arg(
        'EAF_DATA',
        type=lambda x: is_valid_file(split_data_parser, x),
        widget='FileChooser',
        help='Path to eaf_data.csv (output of eaf_data command) to generate data splits from.'
    )
    add_arg(
        'OUT_DIR',
        type=lambda x: is_valid_dir(split_data_parser, x),
        widget='FileChooser',
        help='.csv path to output to. Default behavior is to use same filename as EAF_FILEPATH, '\
            +'or if EAF_FILEPATH is a directory, create a file named split_data.csv in the given directory.'
    )
    add_arg(
        '-s',
        '--splitsize',
        type=float,
        nargs=3,
        help='Size of training, validation and test splits, given as floats where 0 < size < 1. Default is (0.8, 0.1, 0.1).'
    )
    add_arg(
        '--lid',
        action='store_true',
        help="Indicates data split is being made for LID model (split_data will use 'lang' columns instead of 'text')")

def init_snip_audio_parser(snip_audio_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(snip_audio_parser, *args, **kwargs)
    add_arg(
        'ANNOTATIONS',
        type=lambda x: is_valid_file(snip_audio_parser, x),
        widget='FileChooser',
        help='Path to eaf_data.csv or .eaf file.'
    )
    add_arg(
        'OUT_DIR', type=lambda x: is_valid_dir(snip_audio_parser, x),
        widget='DirChooser',
        help='Path to folder to save output in.'
    )
    add_arg(
        '-t',
        '--tier',
        nargs='+',
        help='Tier label(s) to read annotations from (default is to use all tiers).'
    )

def init_lid_labels_parser(lid_labels_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(lid_labels_parser, *args, **kwargs)
    add_arg(
        '--categories',
        help="Unique category labels.", 
        nargs='+'
    )
    add_arg(
        '--default_category',
        '-d',
        help='Unique identifier to map to all other strings.'
    )
    add_arg(
        '-e',
        '--empty',
        help='What category empty annotations should be mapped to. If not provided, exclude all empty rows.'
    )
    add_arg(
        '--no_balance',
        action='store_true',
        help='Default behavior is to downsample overrepresented categories so that an equal number of each language is represented in the dataset. '\
        +'Pass this argument to override this behavior and allow for an unequal number of labels for each language.'
    )

def init_asr_labels_parser(asr_labels_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(asr_labels_parser, *args, **kwargs)
    add_arg(
        '--lang_labels',
        help='ISO code or other unique identifiers used in LID, to be ignored in ASR.',
        nargs='+'
    )

def init_textnorm_parser(textnorm_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(textnorm_parser, *args, **kwargs)
    add_arg(
        'METADATA',
        type=lambda x: is_valid_file(textnorm_parser, x),
        widget='FileChooser',
        help="Path to metadata.csv, eaf_data.csv, or other .csv file containing a 'text' column to normalize.",
    )
    add_arg(
        '--skip_normalize_unicode',
        action='store_true',
        help='Pass this arg to skip unicode normalization '+\
            '(replacing composite diacritics with combining '+\
            'and replacing non-canonical character variants with their canonical equivalent). '+\
            'Default behavior is to perform normalization.',
    )
    add_arg(
        '--dont_lower',
        action='store_true',
        help='Pass this arg to skip casting all letters to lowercase.'
    )
    add_arg(
        '--outpath',
        '-o',
        help='Filepath to store csv with normalized output to. Default is to overwrite original file.'
    )
    add_arg(
        '--keep_unnormalized',
        '-k',
        help='Keep unnormalized text in a separate column `unnormalized`. Default is to discard.',
        action='store_true'
    )

def init_dataset_parser(lid_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(lid_parser, *args, **kwargs)
    add_arg(
        'EAF_DIR',
        type=lambda x: is_valid_dir(lid_parser, x),
        widget='DirChooser',
        help='Folder containing .eafs for processing into training datasets.'
    )
    add_arg(
        '--ac',
        type=lambda x: is_valid_dir(lid_parser, x),
        widget='DirChooser',
        help='Folder to initalize Audio Classification (AC) dataset in.'
    )
    add_arg(
        '--asr',
        type=lambda x: is_valid_dir(lid_parser, x),
        widget='DirChooser',
        help='Folder to initalize Automatic Speech Recognition (ASR) dataset in.'
    )
    add_arg(
        '-t',
        '--tier',
        nargs='+',
        help='Tier label(s) to read annotations from (default is to use all tiers).'
    )

    add_arg(
        '-s',
        '--splitsize',
        type=float,
        nargs=3,
        help='Size of training, validation and test splits, given as floats where 0 < size < 1. Default is (0.8, 0.1, 0.1).'
    )
    add_arg(
        '-r',
        '--recursive',
        action='store_true',
        help='Recurse over all subdirectories in EAF_DIR.'
    )
    add_arg(
        '--hf_user',
        help='User name for Huggingface Hub. Only pass if wanting to save dataset to Huggingface Hub, otherwise saves locally only.'
    )
    asr_args = lid_parser.add_argument_group('ASR', 'Arguments for ASR dataset')
    init_asr_labels_parser(asr_args)
    
    lid_args = lid_parser.add_argument_group('LID', 'Arguments for LID dataset')
    init_lid_labels_parser(lid_args)

def init_vocab_parser(vocab_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(vocab_parser, *args, **kwargs)
    add_arg(
        'CSV_PATH',
        type=lambda x: is_valid_file(vocab_parser, x),
        widget='FileChooser',
        help='Path to eaf_data.csv or metadata.csv containing data to create vocab with.'                              
    )
    add_arg(
        'TASK',
        choices=['LID', 'ASR'],
        help='Task to be trained, either Language IDentification (LID) or Automatic Speech Recognition (ASR).'
    )
    add_arg(
        '--out_dir',
        type=lambda x: is_valid_dir(vocab_parser, x),
        widget='FileChooser',
        help='Path to directory to save vocab.json in. Default is to save it to parent directory of CSV_PATH.'                          
    )

def init_train_parser(train_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(train_parser, *args, **kwargs)
    add_arg(
        'DATA_PATH',
        # type=lambda x: is_valid_dir(train_parser, x), TODO: create validation function for HF urls
        widget='DirChooser',
        help='Folder or HuggingFace URL containing dataset for language identification and/or automatic speech recognition.'                          
    )
    add_arg(
        'OUT_PATH',
        # type=lambda x: is_valid_dir(train_parser, x),
        widget='DirChooser',
        help='Folder or HuggingFace URL to save language identification and/or automatic speech recognition model to. '\
            + 'Recommended format is wav2vec2-large-mms-1b-LANGUAGENAME (if using default model mms-1b-all).'                          
    )
    add_arg(
        'TASK',
        choices=['LID', 'ASR', 'LM'],
        help='Task to be trained, either Language IDentification (LID), Automatic Speech Recognition (ASR) or Language Modeling (LM).'
    )
    add_arg(
        '--hf_user',
        help='User name for Huggingface Hub. Only pass if wanting to save model to Huggingface Hub, otherwise saves locally only.'
    )
    add_arg(
        '-m',
        '--model_url',
        help='url or filepath to pretrained model to finetune. '\
        +'By default uses Massive Multilingual Speech (facebook/mms-1b-all) for ASR and LID '\
        +'and gpt2 for LM.'
    )
    add_arg(
        '-v',
        '--vocab_path',
        default='vocab.json',
        help='Filepath for vocab object.'
    )
    add_arg(
        '-lc',
        '--label_col',
        help="Column in dataset to use for label (default is 'text' for ASR or LM and 'lang' for LID)."
    )
    add_arg(
        '--audio_cutoff',
        '-ac',
        help='Exclude any audio records with length greater than cutoff (in frames).',
        type=int
    )
    add_arg(
        '--sort_by_len',
        '-sbl',
        help='Sort dataset so that shorter inputs come first.',
        action='store_true',
    )
    add_arg(
        '--save_eval_preds',
        action='store_true',
        help='Save eval predictions to output file for each eval loop.',
    )
    add_arg(
        '--device_map',
        help='Device map for model (default is `auto`).',
        default='auto',
    )
    add_arg(
        '--train_data_percent',
        help='Percent of train data to load (if None, load all)',
        type=float,
    )
    add_arg(
        '--train_checkpoint',
        help='Path to checkpoint folder if resuming from checkpoint.',
        widget='DirChooser',
    )
    add_arg(
        '--use_accelerate',
        help='Run script as a subprocess using accelerate launch.',
        action='store_true',
    )
    add_arg(
        '--adapter',
        help='Path to adapter file to load (if any).'
    )
    add_remote_args(train_parser)
    add_hyperparameter_args(train_parser)

def init_infer_parser(infer_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(infer_parser, *args, **kwargs)
    add_arg(
        "WAV_FILE",
        type=lambda x: is_valid_file(infer_parser, x),
        widget='FileChooser',
        help='Path to .wav file to run inference on.'
    )
    add_arg(
        "MODEL_URL",
        widget='DirChooser',
        help="Path to HuggingFace model to use for inference."
    )
    add_arg(
        "TASK",
        help="Task to use for inference.", choices=['ASR', 'LID']
    )
    add_arg(
        "OUT",
        help='Path to .eaf file to save annotations to.'
    )
    add_remote_args(infer_parser)

def init_annotate_parser(annotate_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(annotate_parser, *args, **kwargs)
    add_arg(
        "WAV_FILE",
        type=lambda x: is_valid_file(annotate_parser, x),
        widget='FileChooser',
        help='Path to .wav file to run inference on.'
    )
    add_arg(
        "OUT",
        widget='FileChooser',
        help='Path to .eaf file to save annotations to.'
    )
    add_arg(
        "--asr_path",
        widget='DirChooser',
        help="Path to HuggingFace model to use for automatic speech recognition. If not passed, label utterances with detected language/speaker probabilities.")
    add_arg(
        "--lang",
        help="ISO code for target language to annotate."
    )
    add_arg(
        "--ac_path",
        widget='DirChooser',
        help="Path to HuggingFace model to use for audio classification. If not passed, annotate all detected utterances.")
    add_arg(
        "--inference_method",
        "-im",
        choices=['local', 'api', 'try_api'],
        default='try_api',
        help='Method for running inference. If local, download model if not already downloaded and '\
            +'run pipeline. If api, use HuggingFace inference API. If try_api, call HuggingFace API '\
            +'but run locally if API returns error.'        
    )
    add_arg(
        '-t',
        '--tier',
        default='default-lt',
        help="Add ASR annotations to given tier. By default `default-lt`."
    )
    add_arg(
        '--template',
        type=lambda x: is_valid_file(annotate_parser, x),
        widget='FileChooser',
        help='Template .etf file for generating output .eafs.'
    )
    
    add_batch_args(annotate_parser)
    add_remote_args(annotate_parser)

def init_eval_parser(eval_parser: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(eval_parser, *args, **kwargs)
    add_arg(
        'DATASET',
        help='Path to dataset to use for evaluation.'
    )
    add_arg(
        'MODEL',
        help='Path to model to evalute on.'
    )
    # funct: Optional[Callable] = None, #TODO: implement passing the path to python file
    add_arg(
        '--out',
        '-o',
        help='Path to save output to (default is $MODEL_eval.json)'
    )
    add_arg(
        '--metric',
        '-m',
        nargs='+',
        help='Metrics to evaluate on.',
        default='cer_and_wer'
    )
    add_arg(
        '--label_col',
        '-lc',
        help='Name of column in dataset containing label. Default `text`.',
        default='text',
    )
    add_arg(
        '--input_col',
        '-ic',
        help='Name of column in dataset containing input data. Default `audio`.',
        default='audio',
    )
    add_arg(
        '--split',
        '-s',
        help='Name of split to evaluate on. Default `test`.',
        default='test',
    )
    add_arg(
        '--task',
        '-t',
        help='Task to evaluate on. Default is ASR',
        default='ASR',
        choices=['ASR', 'LID'],
    )

def add_batch_args(parser: argparse.ArgumentParser) -> None:
    batch_args = parser.add_argument_group(
        'batch',
        'Arguments for running batch command (on directories of files instead of individual files).'
    )
    batch_args.add_argument(
        '-b',
        '--batch',
        action='store_true',
        help='Run on all wav files in a given folder.'
    )
    batch_args.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='If running a batch process, recurse over all subdirectories in folder.'
    )
    batch_args.add_argument(
        '--overwrite',
        action='store_true',
        help='If running batch process, overwrite applicable files in already in output directory. '\
         +'Default behavior is to skip files already present.'
    )

def add_remote_args(parser: argparse.ArgumentParser) -> None:
    remote_args = parser.add_argument_group(
        'remote',
        description='Arguments for running command on a remote server.'
    )
    add_arg = lambda *args, **kwargs: add_hybrid_arg(remote_args, *args, **kwargs)
    add_arg(
        "--remote",
        help='Run command on remote server. Defaults to True in versions without PyTorch.',
        action='store_true',
        default=not TORCH,
    )
    add_arg(
        "--server",
        help="Address for server to run command on.",
    )
    add_arg(
        "--password",
        help="Password to log in to server.",
        widget="PasswordField",
    )
    add_arg(
        "--server_python",
        help="Path to python interpreter to use on server."
    )
    add_arg(
        "--files_on_server",
        help="Indicates that datafiles for command are already located on server and do not need to be uploaded.",
        action='store_true',
    )

def add_hyperparameter_args(parser: argparse.ArgumentParser) -> None:
    hyper_args = parser.add_argument_group(
        'hyperparameters',
        description='Hyperparameter values for training'
    )
    add_args_from_dict(DEFAULT_HYPERPARAMS, hyper_args)

def add_args_from_dict(args_dict: Dict[str, Any], arg_group: argparse.ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: add_hybrid_arg(arg_group, *args, **kwargs)
    for k, v in args_dict.items():
        if type(v) is bool:
            add_arg('--'+k, default=v, action='store_true')
        else:
            add_arg('--'+k, type=type(v), default=v)

def handle_merge(args: Dict[str, Any]) -> int:
    from zugubul.elan_tools import merge

    eaf_source = args['EAF_SOURCE']
    eaf_matrix = args['EAF_MATRIX']
    eaf_out = args['eaf_out']
    tier = args['tier']
    overlap_behavior = args['overlap_behavior']
    batch = args['batch']
    recursive = args['recursive']
    overwrite = args['overwrite']

    if not eaf_out:
        # default behavior is to override eaf2 file
        eaf_out = eaf_source

    if batch or os.path.isdir(eaf_matrix):
        assert os.path.isdir(eaf_matrix) and os.path.isdir(eaf_source),\
        'If running batch process both EAF_SOURCE and EAF_MATRIX must be folders.'
        
        def safe_merge(**kwargs):
            try:
                return merge(**kwargs)
            except (ValueError, FileNotFoundError) as error:
                tqdm.write(f"{error}. Skipping file")

        batch_funct(
            safe_merge,
            dir=eaf_source,
            file_arg='eaf_source',
            suffix='.eaf',
            kwargs={
                'eaf_matrix': eaf_matrix,
                'tier': tier,
                'overlap_behavior': overlap_behavior,
            },
            recursive=recursive,
            overwrite=overwrite,
            save_f=lambda data_file, out: save_eaf_batch(data_file, out, eaf_out),
            out_path_f=lambda data_file: get_eaf_outpath(data_file, eaf_out),
            desc='Merging elan files',
        )
    else:
        eaf = merge(
            eaf_matrix=eaf_matrix,
            eaf_source=eaf_source,
            tier=tier,
            overlap_behavior=overlap_behavior
        )
        eaf_to_file_safe(eaf ,eaf_out)
    return 0

def handle_vad(args: Dict[str, Any]) -> int:
    from zugubul.vad_to_elan import label_speech_segments

    wav_fp = args['WAV_FILEPATH']
    eaf_fp = args['EAF_FILEPATH']
    eaf_source = args['source']
    batch = args['batch']
    recursive = args['recursive']
    overwrite = args['overwrite']
    vad = args['vad']
    tier = args['tier']
    etf = args['template']

    if batch or os.path.isdir(wav_fp):
        handle_vad_batch(
            wav_fp=wav_fp,
            eaf_fp=eaf_fp,
            tier=tier,
            vad=vad,
            eaf_source=eaf_source,
            etf=etf,
            recursive=recursive,
            overwrite=overwrite,
        )
    else: 
        if not vad:
            print(f"Performing VAD on {wav_fp}")
        else:
            print(f"Reading VAD data from {vad}")
        eaf = label_speech_segments(wav_fp=wav_fp, tier=tier, etf=etf)
        print(f"Saving EAF file to {eaf_fp}")
        eaf_to_file_safe(eaf, eaf_fp)
    if eaf_source:
        # if eaf_source is passed, perform merge
        args['EAF_MATRIX'] = eaf_fp
        args['EAF_SOURCE'] = eaf_source
        args['eaf_out'] = eaf_fp
        args['overwrite'] = True
        handle_merge(args)
    return 0

def handle_vad_batch(
        wav_fp: str,
        eaf_fp: str,
        tier: Optional[str] = None,
        vad: Optional[str] = None,
        eaf_source: Optional[str] = None,
        etf: Optional[str]= None,
        recursive: bool = False,
        overwrite: bool = False,
    ):
    """
    Helper for batch processing of handle_vad
    """
    from zugubul.vad_to_elan import label_speech_segments, RvadError


    assert os.path.isdir(wav_fp) and os.path.isdir(eaf_fp),\
            'If running batch process both WAV_FILEPATH and EAF_FILEPATH must be folders.'
    if eaf_source:
        assert os.path.isdir(eaf_source), 'If using SOURCE arg and running batch process, SOURCE must point to folder.'
    if vad:
        assert os.path.isdir(vad), 'If using VAD arg and running batch process, VAD must point to folder.'

    def label_speech_segments_safe(**kwargs):
        try:
            return label_speech_segments(**kwargs)
        except RvadError as error:
            tqdm.write(f"{error} Skipping file.")

    batch_funct(
        f=label_speech_segments_safe,
        dir=wav_fp,
        file_arg='wav_fp',
        suffix='.wav',
        kwargs = {
            'vad_fp': vad,
            'tier': tier,
            'etf': etf,
        },
        recursive=recursive,
        overwrite=overwrite,
        out_path_f=lambda data_file: get_eaf_outpath(data_file, eaf_fp),
        save_f=lambda data_file, out: save_eaf_batch(data_file, out, eaf_fp),
        desc='Running voice activity detection',
    )

def handle_trim(args: Dict[str, Any]) -> int:
    from zugubul.elan_tools import trim

    eaf_fp = args['EAF_FILEPATH']
    out_fp = args['eaf_out']
    if not out_fp:
        # default behavior is to override input file
        out_fp = eaf_fp
    stopword = args['stopword']
    tier = args['tier']
    batch = args['batch']
    recursive = args['recursive']
    overwrite = args['overwrite']

    if batch or os.path.isdir(eaf_fp):
        assert os.path.isdir(eaf_fp) and os.path.isdir(out_fp),\
        'If running batch process both EAF_FILEPATH and eaf_out (if provided) must be folders.'
        batch_funct(
            f=trim,
            dir=eaf_fp,
            file_arg='eaf',
            suffix='.eaf',
            kwargs={
                'tier': tier,
                'stopword': stopword,
            },
            save_f=lambda data_file, out: save_eaf_batch(data_file, out, out_fp),
            out_path_f=lambda data_file: get_eaf_outpath(data_file, out_fp),
            recursive=recursive,
            overwrite=overwrite,
            desc='Trimming elan files',
        )
    else:
        eaf = trim(eaf_fp, tier, stopword)
        eaf_to_file_safe(eaf, out_fp)
    return 0

def handle_eaf_data(args: Dict[str, Any]) -> int:
    from zugubul.elan_tools import eaf_data

    eaf_fp = args['EAF_FILEPATH']
    out_fp = args['output']
    if not out_fp:
        # default behavior is to change suffix of input file
        if os.path.isfile(eaf_fp):
            out_fp = eaf_fp.replace('.eaf', '.csv')
        # if eaf_fp is folder, create file eaf_data.csv in folder
        else:
            out_fp = os.path.join(eaf_fp, 'eaf_data.csv')
    if os.path.isdir(out_fp):
        out_fp = os.path.join(out_fp, 'eaf_data.csv')
    tier = args['tier']
    media = args['media']
    batch = args['batch']
    recursive = args['recursive']

    if batch or os.path.isdir(eaf_fp):
        out = batch_funct(
            f=eaf_data,
            dir=eaf_fp,
            suffix='.eaf',
            file_arg='eaf',
            kwargs={
                'tier': tier,
                'media': media
            },
            recursive=recursive,
            overwrite=True
        )
        df = pd.concat(out.values())
    else:
        df = eaf_data(
            eaf=eaf_fp,
            tier=tier,
            media=media
        )

    df.to_csv(out_fp, index=False)

    return 0

def handle_split_data(args: Dict[str, Any]) -> int:
    from zugubul.models.dataset import split_data

    eaf_data_fp = args['EAF_DATA']
    out_dir = args['OUT_DIR']
    lid = args['lid']
    splitsize = args['splitsize'] or (0.8, 0.1, 0.1)

    metadata_fp = split_data(
        eaf_data=eaf_data_fp,
        out_dir=out_dir,
        splitsize=splitsize,
        lid=lid
    )

    print('Metadata for training split saved to', metadata_fp)

    return 0

def handle_snip_audio(args: Dict[str, Any]) -> int:
    from zugubul.elan_tools import snip_audio

    annotations = Path(args['ANNOTATIONS'])
    out_dir = Path(args['OUT_DIR'])
    tier = args['tier']

    df = snip_audio(
        annotations=annotations,
        out_dir=out_dir,
        tier=tier
    )

    csv_path = out_dir/'eaf_data.csv'
    df.to_csv(csv_path, index=False)
    print(f'Audio clips and eaf_data saved to {out_dir}.')

    return 0

def handle_lid_labels(args: Dict[str, Any]) -> int:
    from zugubul.models.dataset import make_ac_labels

    annotations = args['ANNOTATIONS']
    out_path = args['out_path']
    categories=args['categories']
    default_category=args['default_category']
    empty = args['empty']
    balance = not args['no_balance']

    print(f"Making AC labels with {categories=}")

    if not out_path:
        # default behavior is to overwrite annotations
        out_path = annotations

    lid_df = make_ac_labels(
        annotations=annotations,
        categories=categories,
        default_category=default_category,
        empty=empty,
        balance=balance,
    )

    lid_df.to_csv(out_path, index=False)

    return 0

def handle_dataset(args: Dict[str, Any]) -> int:
    from datasets import load_dataset
    from huggingface_hub import login, HfApi
    from zugubul.models.vocab import vocab_from_csv
    from zugubul.models.dataset import make_asr_labels



    eaf_dir = Path(args['EAF_DIR'])
    ac_dir = args['ac']
    asr_dir = args['asr']
    hf_user = args['hf_user']


    if (not ac_dir) and (not asr_dir):
        print("Pass `ac`, `asr` or both in order to create a dataset for finetuning.")

    # eaf_data
    print('Generating eaf_data.csv...')
    args['batch'] = True
    args['EAF_FILEPATH'] = eaf_dir
    args['output'] = eaf_dir
    args['overwrite'] = None
    args['media'] = None
    handle_eaf_data(args)

    if ac_dir:
        ac_dir = Path(ac_dir)
        # lid_labels
        print('Normalizing LID labels...')
        args['ANNOTATIONS'] = eaf_dir/'eaf_data.csv'
        args['out_path'] = ac_dir/'metadata.csv'
        handle_lid_labels(args)

        # split_data
        print('Making train/validation/test splits for LID...')
        args['EAF_DATA'] = ac_dir/'metadata.csv'
        args['OUT_DIR'] = ac_dir
        args['lid'] = True
        handle_split_data(args)

        # make LID tokenizer
        print('Making vocab file for LID...')
        lid_vocab = vocab_from_csv(
            csv_path=ac_dir/'metadata.csv',
            vocab_dir=ac_dir,
            lid=True
        )
        # push to hf, if indicated by user
        if hf_user:
            login()
            lid_name = hf_user/ac_dir.stem
            dataset = load_dataset(ac_dir)
            dataset.push_to_hub(lid_name, private=True)
            api = HfApi()
            api.upload_file(
                path_or_fileobj=lid_vocab,
                path_in_repo='vocab.json',
                repo_id=lid_name,
                repo_type='dataset'
            )

    if asr_dir:
    # asr_labels
        print('Normalizing ASR labels')
        asr_dir = Path(asr_dir)
        make_asr_labels(
            annotations=eaf_dir/'eaf_data.csv',
            lid_labels=args['lang_labels'],
            #process_length = not args['no_length_processing'],
            #min_gap=int(args['min_gap']),
            #min_length=int(args['min_length']),
        ).to_csv(asr_dir/'metadata.csv')

        # split_data
        print('Making train/validation/test splits for ASR...')
        args['EAF_DATA'] = asr_dir/'metadata.csv'
        args['OUT_DIR'] = asr_dir
        args['lid'] = False
        handle_split_data(args)

        # make ASR tokenizer
        print('Making vocab file for ASR...')
        asr_vocab = vocab_from_csv(
            csv_path=asr_dir/'metadata.csv',
            vocab_dir=asr_dir,
            lid=False
        )
        # push to hf, if indicated by user
        if hf_user:
            login()
            asr_name = hf_user/asr_dir.stem
            dataset = load_dataset(asr_dir)
            dataset.push_to_hub(asr_name, private=True)
            api = HfApi()
            api.upload_file(
                path_or_fileobj=asr_vocab,
                path_in_repo='vocab.json',
                repo_id=asr_name,
                repo_type='dataset'
            )



    return 0

def handle_textnorm(args: Dict[str, Any]) -> int:
    from zugubul.textnorm import unicode_normalize, get_char_metadata, get_reps_from_chardata, make_replacements

    metadata = args['METADATA']
    df = pd.read_csv(metadata, keep_default_na=False)
    text = df['text']
    if args['keep_unnormalized']:
        df['unnormalized'] = df['text'].copy()
    if not args['skip_normalize_unicode']:
        print('Normalizing unicode symbols and diacritics...')
        text = text.apply(unicode_normalize)
    if not args['dont_lower']:
        print('Normalizing all letters to lowercase...')
        text = text.apply(str.lower)
    print('Stripping newlines and trailing whitespace...')
    text = text.apply(str.strip)
    text = text.apply(lambda s: s.replace('\n', '').replace('\r', ''))

    char_metadata = get_char_metadata(text)
    if GUI and PYSIMPLEGUI:
        from zugubul.textnorm_window import char_metadata_window
        reps = char_metadata_window(char_metadata)
    else:
        with open('charmetadata.json', 'w') as f:
            json.dump(char_metadata, f, ensure_ascii=False, indent=2)
        input(
            'Open charmetadata.json and for each character change the value of `replace` from False to a string or None '+\
            'to replace the character with another string or delete it. '+\
            'Additional replacement rules can be specified by creating new objects with the `replace` key. '+\
            'Hit `Enter` when done.'
        )
        with open('charmetadata.json') as f:
            char_metadata = json.load(f)
        reps = get_reps_from_chardata(char_metadata)
    text = text.apply(lambda s: make_replacements(s, reps))

    df['text'] = text
    outpath = args['outpath'] or metadata
    df.to_csv(outpath, index=False)
    return 0

def handle_vocab(args: Dict[str, Any]) -> int:
    from zugubul.models.vocab import vocab_from_csv
    
    csv_path = Path(args['CSV_PATH'])
    task = args['TASK']
    lid = task == 'LID'
    vocab_dir = args['out_dir'] or csv_path.parent
    vocab_from_csv(
        csv_path=csv_path,
        vocab_dir=vocab_dir,
        lid=lid
    )
    return 0

def handle_train(args: Dict[str, Any]) -> int:
    if args['remote']:
        from zugubul.remote import run_script_on_server
        return run_script_on_server(
            sys.argv,
            in_files=[args['DATA_PATH']],
            out_files=[],
            server=args['server'],
            passphrase=args['password'],
            server_python=args['server_python'],
            files_on_server=args['files_on_server'],
            use_accelerate=args['use_accelerate'],
        )

    if args.pop('use_accelerate'):
        from zugubul.remote import make_arg_str
        import subprocess
        arglist = sys.argv[2:]
        arglist = [x for x in arglist if x != '--use_accelerate']
        arglist = ['accelerate', 'launch', '-m', 'zugubul.models.train'] + arglist
        argstr = make_arg_str(arglist)
        os.environ.pop('GUI', None)
        print('Running accelerate as subprocess with command:', argstr)
        return subprocess.run(argstr, shell=True)

    if not TORCH:
        print("Cannot run train locally if using Zugubul without PyTorch.")
        return 1
    from zugubul.models.train import train

    data_dir = args.pop('DATA_PATH')
    vocab = args.pop('vocab_path')
    out_dir = args.pop('OUT_PATH')
    task = args.pop('TASK')
    model_name = args.pop('model_url')
    if not model_name:
        model_name = 'gpt2' if task=='LM' else 'facebook/mms-1b-all'

    # remove COMMAND arg if present
    args.pop('COMMAND', None)
    # remove remote args if running locally
    remote_args = ['remote', 'server', 'password', 'server_python', 'files_on_server']
    for argname in remote_args:
        args.pop(argname)

    train(
        out_dir=out_dir,
        model_str=model_name,
        dataset=data_dir,
        task=task,
        vocab=vocab,
        **args,
    )
    return 0

def handle_infer(args: Dict[str, Any]) -> int:
    from zugubul.models.infer import infer


    wav_file = args['WAV_FILE']
    model = args['MODEL_URL']
    task = args['TASK']
    out_fp = args['OUT']

    eaf = infer(
        source=wav_file,
        model=model,
        task=task,
    )
    eaf.to_file(out_fp)

    return 0

def handle_annotate(args: Dict[str, Any]) -> int:
    if args['remote']:
        from zugubul.remote import run_script_on_server
        in_files=[args['WAV_FILE']]
        if args['template']:
            in_files.append(args['template'])
        out_files=[args['OUT']]
        return run_script_on_server(
            argv=sys.argv,
            in_files=in_files,
            out_files=out_files,
            server=args['server'],
            server_python=args['server_python'],
            passphrase=args['password'],
            files_on_server=args['files_on_server'],
        )
    from zugubul.models.infer import annotate

    wav_file = args['WAV_FILE']
    lid_model = args['ac_path']
    asr_model = args['asr_path']
    tgt_lang = args['lang']
    out_fp = args['OUT']
    inference_method = args['inference_method']
    tier = args['tier']
    etf = args['template']

    batch = args['batch']
    recursive = args['recursive']
    overwrite = args['overwrite']

    if batch:
        batch_funct(
            f=annotate,
            dir=wav_file,
            suffix='.wav',
            file_arg='source',
            kwargs={
                'lid_model': lid_model,
                'asr_model': asr_model,
                'tgt_lang': tgt_lang,
                'inference_method': inference_method,
                'tier': tier,
                'etf': etf,
            },
            out_path_f=lambda data_file: get_eaf_outpath(data_file, wav_file),
            save_f=lambda data_file, out: save_eaf_batch(data_file, out, wav_file),
            recursive=recursive,
            overwrite=overwrite
        )
    else:
        eaf = annotate(
            source=wav_file,
            lid_model=lid_model,
            asr_model=asr_model,
            tgt_lang=tgt_lang,
            inference_method=inference_method,
            tier=tier,
            etf=etf
        )
        eaf.to_file(out_fp)

    return 0

def handle_eval(args: Dict[str, Any]) -> int:
    from zugubul.models.eval import eval
    import json
    dataset = args.pop("DATASET")
    model = args.pop("MODEL")
    out_path = args.pop("out")
    args.pop("COMMAND", None)
    if not out_path:
        out_path = model+'_eval.json'
    out = eval(
        dataset=dataset,
        model_str=model,
        **args
    )
    with open(out_path, 'w') as f:
        breakpoint()
        json.dump(out, f)
    return 0


def get_eaf_outpath(data_file: str, out_folder: str) -> str:
    """
    Returns path to .eaf file with same name as data_file
    in directory out_folder.
    """
    return os.path.join(out_folder, Path(data_file).stem + '.eaf')


def save_eaf_batch(data_file: str, out: Elan.Eaf, out_folder: str) -> str:
    """
    Saves output provided by batch_funct to filepath with same name as data_file
    in directory out_folder.
    Returns path output is saved to.
    """
    out_path = get_eaf_outpath(data_file, out_folder)
    tqdm.write(f'{data_file} \t ---> \t{out_path}')
    out_path = out_path
    if out:
        eaf_to_file_safe(out ,out_path)
    return out_path

@HybridGooey(
        program_name='Zugubul',
        image_dir=importlib_resources.files('zugubul_icons')
)
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = HybridGooeyParser(description='Tools for automatic transcription of audio files with ELAN.')
    subparsers = parser.add_subparsers(dest='COMMAND')
    
    merge_parser = subparsers.add_parser('merge', help='Add all annotations in eaf2 not present in eaf1')
    init_merge_parser(merge_parser)

    trim_parser = subparsers.add_parser('trim',
                        help='Remove all empty annotations in a given tier, '\
                            +'or all annotations with a given value indicated by the STOPWORD argument.'
                    )        
    init_trim_parser(trim_parser)

    vad_parser = subparsers.add_parser('vad',
                        help='Create ELAN file with annotations corresponding to detected speech segments in recording.'
                    )
    init_vad_parser(vad_parser)

    eaf_data_parser = subparsers.add_parser('eaf_data', help='Process metadata for given .eaf file(s) into a csv.')
    init_eaf_data_parser(eaf_data_parser)

    split_data_parser = subparsers.add_parser('split_data', help='Make training splits from a collection of ELAN files or from a .csv file output by the eaf_data command.')
    init_split_data_parser(split_data_parser)

    snip_audio_parser = subparsers.add_parser('snip_audio', help='Create a directory of audio clips corresponding to ELAN annotations.')
    init_snip_audio_parser(snip_audio_parser)

    lid_labels_parser = subparsers.add_parser('lid_labels', help='Create Language IDentification (LID) labels for eaf_data.csv.')
    # add this arg separately because all other args are shared between lid_labels and lid
    lid_labels_parser.add_argument('ANNOTATIONS', type=lambda x: is_valid_file(lid_labels_parser, x),
        help='Path to eaf_data.csv or .eaf file.'
    )
    lid_labels_parser.add_argument('--out_path', type=lambda x: is_valid_file(lid_labels_parser, x),
        help='Path to .csv file to output to (default is to overwrite ANNOTATIONS).'
    )
    init_lid_labels_parser(lid_labels_parser)

    dataset_parser = subparsers.add_parser('dataset', help='Initialize Audio Classification (AC) and Automatic Speech Recognition (ASR) datasets from directory of .eaf files.')
    init_dataset_parser(dataset_parser)

    textnorm_parser = subparsers.add_parser('textnorm', help='Normalize text column in a csv file.')
    init_textnorm_parser(textnorm_parser)

    vocab_parser = subparsers.add_parser('vocab', help='Create vocab.json for initializing a tokenizer from a datafile (eaf_data.csv or metadata.csv).')
    init_vocab_parser(vocab_parser)

    train_parser = subparsers.add_parser('train', help='Train Language IDentification (LID) or Automatic Speech Recognition (ASR) model on given dataset.')
    init_train_parser(train_parser)

    infer_parser = subparsers.add_parser('infer', help='Run inference on a given audio file.')
    init_infer_parser(infer_parser)

    annotate_parser = subparsers.add_parser('annotate', help='Automatically annotate a fieldwork recording in the target language.')
    init_annotate_parser(annotate_parser)

    eval_parser = subparsers.add_parser('eval', help='Evaluate a HF model on test data.')
    init_eval_parser(eval_parser)

    args = vars(parser.parse_args(argv))

    command = args['COMMAND']
    if command == 'trim':
        return handle_trim(args)
    elif command == 'merge':
        return handle_merge(args)
    elif command == 'vad':
        return handle_vad(args)
    elif command == 'eaf_data':
        return handle_eaf_data(args)
    elif command == 'split_data':
        return handle_split_data(args)
    elif command == 'lid_labels':
        return handle_lid_labels(args)
    elif command == 'dataset':
        return handle_dataset(args)
    elif command == 'textnorm':
        return handle_textnorm(args)
    elif command == 'vocab':
        return handle_vocab(args)
    elif command == 'train':
        return handle_train(args)
    elif command == 'infer':
        return handle_infer(args)
    elif command == 'annotate':
        return handle_annotate(args)
    elif command == 'eval':
        return handle_eval(args)
    return 1

if __name__ == '__main__':
    main()