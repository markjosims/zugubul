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
from pathlib import Path
import pandas as pd
import argparse
from typing import Optional, Sequence, Dict, Union, Any
from datasets import load_dataset
from huggingface_hub import login, HfApi
from zugubul.rvad_to_elan import label_speech_segments, RvadError
from zugubul.elan_tools import merge, trim, eaf_data, snip_audio
from zugubul.models.dataset import split_data, make_lid_labels
from zugubul.models.vocab import vocab_from_csv
from zugubul.models.train import train, init_train_parser
from zugubul.models.infer import infer, annotate
from zugubul.utils import is_valid_file, file_in_valid_dir, is_valid_dir, batch_funct, eaf_to_file_safe
from tqdm import tqdm
from pympi import Elan
import importlib.util

def init_merge_parser(merge_parser: argparse.ArgumentParser):
    add_arg = merge_parser.add_argument
    add_arg('EAF_SOURCE', type=lambda x: is_valid_file(merge_parser, x),
        help='Filepath for eaf file to take output from (or parent directory of eafs).'
    )
    add_arg('EAF_MATRIX', type=lambda x: is_valid_file(merge_parser, x),
        help='Filepath for eaf file to insert annotations into (or parent directory of eafs). '\
    )
    add_arg('-o', '--eaf_out', type=lambda x: file_in_valid_dir(merge_parser, x),
        help='Filepath for eaf file to output to (or parent directory of eafs).'
    )
    add_arg('-t', '--tier', required=False, nargs='+',
        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
            +'For multiple tiers write tier names separated by space.'
    ) 
    add_arg('--overlap_behavior', choices=['keep_source', 'keep_matrix', 'keep_both'], default='keep_source', nargs='?',
        help='Behavior for treating segments that overlap between the two .eafs. '
            +'If keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE. '\
            +'If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
            +'If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not. '\
            +'Default behavior is keep_source'
    )
    add_batch_args(merge_parser)
    
def init_trim_parser(trim_parser: argparse.ArgumentParser):
    add_arg = trim_parser.add_argument
    add_arg('EAF_FILEPATH', type=lambda x: is_valid_file(trim_parser, x),
        help='Filepath for eaf file to trim (or parent directory of eafs).'
    )
    add_arg('-o', '--eaf_out', type=lambda x: file_in_valid_dir(trim_parser, x),
        help='Filepath for eaf file to output to (or parent directory of eafs).'
    )
    add_arg('-t', '--tier', required=False, nargs='+',
        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
            +'For multiple tiers write tier names separated by space.'
    ) 
    add_arg('-s', '--stopword', type=str, default='',
        help='Remove all annotations with this value. By default removes all annotations with no text.'
    ) 
    add_batch_args(trim_parser)

def init_vad_parser(vad_parser: argparse.ArgumentParser):
    add_arg = vad_parser.add_argument
    add_arg('WAV_FILEPATH', type=lambda x: is_valid_file(vad_parser, x),
        help='Filepath for wav file to be processed (or parent directory).'
    )
    add_arg('EAF_FILEPATH', type=lambda x: file_in_valid_dir(vad_parser, x),
        help='Filepath for eaf file to output to (or parent directory).'
    )
    add_arg('-s', '--source', type=lambda x: is_valid_file(vad_parser, x),
        help='Source .eaf file to merge output into (or parent directory).'
    )
    # add_arg('-m', '--merge_output', help='Filepath to output  to merge output into (or parent directory).')
    add_arg('--template', type=lambda x: is_valid_file(vad_parser, x),
        help='Template .etf file for generating output .eafs.'
    )
    add_arg('--overlap_behavior', choices=['keep_source', 'keep_matrix', 'keep_both'], default='keep_source', nargs='?',
        help='Behavior for treating segments that overlap between the two .eafs. '
            +'If keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE. '\
            +'If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
            +'If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not. '\
    )
    add_arg('-v', '--vad', type=lambda x: is_valid_file(vad_parser, x),
        help='Filepath for .vad file linked to the recording. '\
            +'If passed, reads VAD data from file instead of running rVAD_fast.'
    )
    add_arg('-t', '--tier', nargs='+',
        help='Tier label(s) to add annotations to. If none specified uses `default-lt`'\
    )
    add_batch_args(vad_parser)

def init_eaf_data_parser(eaf_data_parser: argparse.ArgumentParser) -> None:
    add_arg = eaf_data_parser.add_argument
    add_arg('EAF_FILEPATH', type=lambda x: is_valid_file(eaf_data_parser, x),
        help='.eaf file to process metadata for.'
    )
    add_arg('-o', '--output', type=lambda x: is_valid_file(eaf_data_parser, x),
        help='.csv path to output to. Default behavior is to use same filename as EAF_FILEPATH, '\
            +'or if EAF_FILEPATH is a directory, create a file named eaf_data.csv in the given directory.'
    )
    add_arg('-t', '--tier', nargs='+',
        help='Tier label(s) to read annotation data from. If none specified read from all tiers.'
    )
    add_arg('-m', '--media',
        help='Path to media file, if different than media path specified in .eaf file.'
    )
    add_batch_args(eaf_data_parser)

def init_split_data_parser(split_data_parser: argparse.ArgumentParser) -> None:
    add_arg = split_data_parser.add_argument
    add_arg('EAF_DATA', type=lambda x: is_valid_file(split_data_parser, x),
        help='Path to eaf_data.csv (output of eaf_data command) to generate data splits from.'
    )
    add_arg('OUT_DIR', type=lambda x: is_valid_dir(split_data_parser, x),
        help='.csv path to output to. Default behavior is to use same filename as EAF_FILEPATH, '\
            +'or if EAF_FILEPATH is a directory, create a file named split_data.csv in the given directory.'
    )
    add_arg('-s', '--splitsize', type=float, nargs=3,
        help='Size of training, validation and test splits, given as floats where 0 < size < 1. Default is (0.8, 0.1, 0.1).'
    )
    add_arg('--lid', action='store_true', help="Indicates data split is being made for LID model (split_data will use 'lang' columns instead of 'text')")

def init_snip_audio_parser(snip_audio_parser: argparse.ArgumentParser) -> None:
    add_arg = snip_audio_parser.add_argument
    add_arg('ANNOTATIONS', type=lambda x: is_valid_file(snip_audio_parser, x),
        help='Path to eaf_data.csv or .eaf file.'
    )
    add_arg('OUT_DIR', type=lambda x: is_valid_dir(snip_audio_parser, x),
        help='Path to folder to save output in.'
    )
    add_arg('-t', '--tier', nargs='+', help='Tier label(s) to read annotations from (default is to use all tiers).')

def init_lid_labels_parser(lid_labels_parser: argparse.ArgumentParser) -> None:
    add_arg = lid_labels_parser.add_argument
    add_arg('-tl', '--targetlang', help='ISO code or other unique identifier for target (fieldwork) language.')
    add_arg('-ml', '--metalang', help='ISO code or other unique identifier for meta language.')
    add_arg('--target_labels', help="Strings to map to target language, or '*' to map all strings except those specified by --meta_labels.", nargs='+')
    add_arg('--meta_labels', help="Strings to map to meta language, or '*' to map all strings except those specified by --target_labels.", nargs='+')
    add_arg('-e', '--empty', choices=['target', 'meta', 'exclude'], help='Whether empty annotations should be mapped to target language, meta language, or excluded.')
    add_arg('--toml', type=lambda x: is_valid_file(lid_labels_parser, x),
        help='Path to a .toml file with metadata for args for this script.'
    )
    add_arg('--no_length_processing', action='store_true', help='Default behavior is to merge annotations belonging to the same language with a gap of <=2s and delete annotations shorter than 1s.'\
        +'Pass this argument to override this behavior and skip processing length for annotations.'
    )
    add_arg('--min_gap', type=int, help='If performing length processing, the minimum duration (in ms) between to annotations of the same language to avoid merging them.', default=200)
    add_arg('--min_length', type=int, help='If performing length processing, the minimum duration (in ms) an annotation must be to not be excluded.', default=1000)
    add_arg('--no_balance', action='store_true', help='Default behavior is to downsample overrepresented categories so that an equal number of each language is represented in the dataset. '\
        +'Pass this argument to override this behavior and allow for an unequal number of labels for each language.'
    )

def init_lid_data_parser(lid_parser: argparse.ArgumentParser) -> None:
    add_arg = lid_parser.add_argument
    add_arg('EAF_DIR', type=lambda x: is_valid_dir(lid_parser, x),
        help='Folder containing .eafs for processing into Language IDentification (LID) dataset.'
    )
    add_arg('OUT_DIR', type=lambda x: is_valid_dir(lid_parser, x),
        help='Folder to initalize Language IDentification (LID) dataset in.'
    )
    add_arg('-t', '--tier', nargs='+', help='Tier label(s) to read annotations from (default is to use all tiers).')

    add_arg('-s', '--splitsize', type=float, nargs=3,
        help='Size of training, validation and test splits, given as floats where 0 < size < 1. Default is (0.8, 0.1, 0.1).'
    )
    add_arg('-r', '--recursive', action='store_true',
        help='Recurse over all subdirectories in EAF_DIR.'
    )
    add_arg('--hf_user', help='User name for Huggingface Hub. Only pass if wanting to save dataset to Huggingface Hub, otherwise saves locally only.')
    add_arg('--hf_repo', help='Repo name for Huggingface Hub. Recommended format is lang-task, e.g. tira-asr or sjpm-lid.')
    init_lid_labels_parser(lid_parser)

def init_vocab_parser(vocab_parser: argparse.ArgumentParser) -> None:
    add_arg = vocab_parser.add_argument
    add_arg('CSV_PATH', type=lambda x: is_valid_file(vocab_parser, x),
        help='Path to eaf_data.csv or metadata.csv containing data to create vocab with.'                              
    )
    add_arg('TASK', choices=['LID', 'ASR'], help='Task to be trained, either Language IDentification (LID) or Automatic Speech Recognition (ASR).')
    add_arg('--out_dir', type=lambda x: is_valid_dir(vocab_parser, x),
        help='Path to directory to save vocab.json in. Default is to save it to parent directory of CSV_PATH.'                          
    )

def init_infer_parser(infer_parser: argparse.ArgumentParser) -> None:
    add_arg = infer_parser.add_argument
    add_arg("WAV_FILE", type=lambda x: is_valid_file(infer_parser, x),
        help='Path to .wav file to run inference on.'
    )
    add_arg("MODEL_URL", help="Path to HuggingFace model to use for inference.")
    add_arg("TASK", help="Task to use for inference.", choices=['ASR', 'LID'])
    add_arg("OUT",
        help='Path to .eaf file to save annotations to.'
    )

def init_annotate_parser(annotate_parser: argparse.ArgumentParser) -> None:
    add_arg = annotate_parser.add_argument
    add_arg("WAV_FILE", type=lambda x: is_valid_file(annotate_parser, x),
        help='Path to .wav file to run inference on.'
    )
    add_arg("LID_URL", help="Path to HuggingFace model to use for language identification.")
    add_arg("ASR_URL", help="Path to HuggingFace model to use for automatic speech recognition.")
    add_arg("LANG", help="ISO code for target language to annotate.")
    add_arg("OUT",
        help='Path to .eaf file to save annotations to.'
    )
    add_arg("--inference_method", "-im", choices=['local', 'api', 'try_api'], default='try_api',
        help='Method for running inference. If local, download model if not already downloaded and '\
            +'run pipeline. If api, use HuggingFace inference API. If try_api, call HuggingFace API '\
            +'but run locally if API returns error.'        
    )
    add_batch_args(annotate_parser)

def add_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-b', '--batch', action='store_true',
                        help='Run on all wav files in a given folder.'
                    )
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='If running a batch process, recurse over all subdirectories in folder.'
                    )
    parser.add_argument('--overwrite', action='store_true',
                        help='If running batch process, overwrite applicable files in already in output directory. '\
                            +'Default behavior is to skip files already present.'
                    )

def handle_merge(args: Dict[str, Any]) -> int:
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
        eaf = label_speech_segments(wav_fp=wav_fp, tier=tier, etf=etf)
        eaf_to_file_safe(eaf, eaf_fp)
    if eaf_source:
        # if eaf_source is passed, perform merge
        args['EAF_MATRIX'] = eaf_fp
        args['EAF_SOURCE'] = eaf_source
        args['eaf_out'] = eaf_fp
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
            'rvad_fp': vad,
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
    eaf_fp = args['EAF_FILEPATH']
    out_fp = args['output']
    if not out_fp:
        # default behavior is to change suffix of input file
        if os.path.isfile(eaf_fp):
            out_fp = eaf_fp.replace('.eaf', '.csv')
        # if eaf_fp is folder, create file eaf_data.csv in folder
        else:
            out_fp = os.path.join(eaf_fp, 'eaf_data.csv')
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
    annotations = args['ANNOTATIONS']
    out_path = args['out_path']
    targetlang = args['targetlang']
    metalang = args['metalang']
    target_labels = args['target_labels']
    meta_labels = args['meta_labels']
    empty = args['empty']
    process_length = not args['no_length_processing']
    min_gap = args['min_gap']
    min_length = args['min_length']
    balance = not args['no_balance']
    toml = args['toml']

    if target_labels == ['*',]:
        target_labels = '*'
    if meta_labels == ['*',]:
        meta_labels = '*'

    if not out_path:
        # default behavior is to overwrite annotations
        out_path = annotations

    lid_df = make_lid_labels(
        annotations=annotations,
        targetlang=targetlang,
        metalang=metalang,
        target_labels=target_labels,
        meta_labels=meta_labels,
        empty=empty,
        process_length=process_length,
        min_gap=min_gap,
        min_length=min_length,
        balance=balance,
        toml=toml
    )

    lid_df.to_csv(out_path, index=False)

    return 0

def handle_lid_dataset(args: Dict[str, Any]) -> int:
    eaf_dir = Path(args['EAF_DIR'])
    out_dir = Path(args['OUT_DIR'])

    # eaf_data
    print('Generating eaf_data.csv...')
    args['batch'] = True
    args['EAF_FILEPATH'] = eaf_dir
    args['output'] = out_dir / 'eaf_data.csv'
    args['overwrite'] = None
    args['media'] = None
    handle_eaf_data(args)

    # lid_labels
    print('Normalizing LID labels...')
    args['ANNOTATIONS'] = args['output']
    # overwrite eaf_data.csv
    args['out_path'] = args['output']
    handle_lid_labels(args)

    # split_data
    print('Making train/validation/test splits...')
    args['EAF_DATA'] = args['output']
    args['lid'] = True
    handle_split_data(args)

    # make tokenizer
    vocab_path = vocab_from_csv(
        csv_path=out_dir/ 'metadata.csv',
        vocab_dir=out_dir,
        lid=True
    )

    # push to hf, if indicated by user
    hf_user = args['hf_user']
    hf_repo = args['hf_repo']

    if hf_user or hf_repo:
        if not (hf_user and hf_repo):
            print('In order to save to Huggingface Hub, both hf_user and hf_repo must be passed.')
            return 1
        
        login()
        repo_name = hf_user + '/' + hf_repo
        dataset = load_dataset(out_dir)
        dataset.push_to_hub(repo_name, private=True)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=vocab_path,
            path_in_repo='vocab.json',
            repo_id=repo_name,
            repo_type='dataset'
        )

    return 0

def handle_vocab(args: Dict[str, Any]) -> int:
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
    data_dir = args['DATA_PATH']
    out_dir = args['OUT_PATH']
    hf = args['hf']

    task = args['TASK'].lower()
    model_name = args['model_url']
    train(
        out_dir=out_dir,
        model=model_name,
        dataset=data_dir,
        hf=hf,
        task=task,
        vocab=os.path.join(data_dir,'vocab.json') if not hf else None
    )
    return 0

def handle_infer(args: Dict[str, Any]) -> int:
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
    wav_file = args['WAV_FILE']
    lid_model = args['LID_URL']
    asr_model = args['ASR_URL']
    tgt_lang = args['LANG']
    out_fp = args['OUT']
    inference_method = args['inference_method']

    batch = args['batch']
    recursive = args['recursive']
    overwrite = args['overwrite']

    if batch:
        batch_funct(
            f=annotate,
            dir=wav_file,
            suffx='.wav',
            file_arg='source',
            kwargs={
                'lid_model': lid_model,
                'asr_model': asr_model,
                'tgt_lang': tgt_lang,
                'inference_method': inference_method
            },
            out_path_f=get_eaf_outpath,
            save_f=save_eaf_batch,
            recursive=recursive,
            overwrite=overwrite
        )

    eaf = annotate(
        source=wav_file,
        lid_model=lid_model,
        asr_model=asr_model,
        tgt_lang=tgt_lang,
        inference_method=inference_method,
    )
    eaf.to_file(out_fp)

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Tools for automatic transcription of audio files with ELAN.')
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

    lid_parser = subparsers.add_parser('lid_data', help='Initialize Language IDentification (LID) dataset from directory of .eaf files.')
    init_lid_data_parser(lid_parser)

    vocab_parser = subparsers.add_parser('vocab', help='Create vocab.json for initializing a tokenizer from a datafile (eaf_data.csv or metadata.csv).')
    init_vocab_parser(vocab_parser)

    train_parser = subparsers.add_parser('train', help='Train Language IDentification (LID) or Automatic Speech Recognition (ASR) model on given dataset.')
    init_train_parser(train_parser)

    infer_parser = subparsers.add_parser('infer', help='Run inference on a given audio file.')
    init_infer_parser(infer_parser)

    annotate_parser = subparsers.add_parser('annotate', help='Automatically annotate a fieldwork recording in the target language.')
    init_annotate_parser(annotate_parser)

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
    elif command == 'lid_data':
        return handle_lid_dataset(args)
    elif command == 'vocab':
        return handle_vocab(args)
    elif command == 'train':
        return handle_train(args)
    elif command == 'infer':
        return handle_infer(args)
    elif command == 'annotate':
        return handle_annotate(args)
    return 1

if __name__ == '__main__':
    main()