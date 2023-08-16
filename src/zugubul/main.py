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
from typing import Optional, Sequence, Mapping
from zugubul.rvad_to_elan import label_speech_segments, RvadError
from zugubul.elan_tools import merge, trim, eaf_data, snip_audio
from zugubul.models.dataset import split_data, make_lid_labels
from zugubul.utils import is_valid_file, file_in_valid_dir, is_valid_dir, batch_funct, eaf_to_file_safe
from tqdm import tqdm


def init_merge_parser(merge_parser: argparse.ArgumentParser):
    merge_parser.add_argument('EAF_SOURCE', type=lambda x: is_valid_file(merge_parser, x),
                        help='Filepath for eaf file to take output from (or parent directory of eafs).'
                    )
    merge_parser.add_argument('EAF_MATRIX', type=lambda x: is_valid_file(merge_parser, x),
                        help='Filepath for eaf file to insert annotations into (or parent directory of eafs). '\
                    )
    merge_parser.add_argument('-o', '--eaf_out', type=lambda x: file_in_valid_dir(merge_parser, x),
                        help='Filepath for eaf file to output to (or parent directory of eafs).'
                    )
    merge_parser.add_argument('-t', '--tier', required=False, nargs='+',
                        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
                         +'For multiple tiers write tier names separated by space.'
                    ) 
    merge_parser.add_argument('--overlap_behavior', choices=['keep_source', 'keep_matrix', 'keep_both'], default='keep_source', nargs='?',
                    help='Behavior for treating segments that overlap between the two .eafs. '
                        +'If keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE. '\
                        +'If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
                        +'If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not. '\
                        +'Default behavior is keep_source'
                )
    add_batch_args(merge_parser)
    
def init_trim_parser(trim_parser: argparse.ArgumentParser):
    trim_parser.add_argument('EAF_FILEPATH', type=lambda x: is_valid_file(trim_parser, x),
                        help='Filepath for eaf file to trim (or parent directory of eafs).'
                    )
    trim_parser.add_argument('-o', '--eaf_out', type=lambda x: file_in_valid_dir(trim_parser, x),
                        help='Filepath for eaf file to output to (or parent directory of eafs).'
                    )
    trim_parser.add_argument('-t', '--tier', required=False, nargs='+',
                        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
                            +'For multiple tiers write tier names separated by space.'
                    ) 
    trim_parser.add_argument('-s', '--stopword', type=str, default='',
                        help='Remove all annotations with this value. By default removes all annotations with no text.'
                    ) 
    add_batch_args(trim_parser)

def init_vad_parser(vad_parser: argparse.ArgumentParser):
    vad_parser.add_argument('WAV_FILEPATH', type=lambda x: is_valid_file(vad_parser, x),
                        help='Filepath for wav file to be processed (or parent directory).'
                       )
    vad_parser.add_argument('EAF_FILEPATH', type=lambda x: file_in_valid_dir(vad_parser, x),
                        help='Filepath for eaf file to output to (or parent directory).'
                       )
    vad_parser.add_argument('-s', '--source', type=lambda x: is_valid_file(vad_parser, x),
                        help='Source .eaf file to merge output into (or parent directory).'
                       )
    # vad_parser.add_argument('-m', '--merge_output', help='Filepath to output  to merge output into (or parent directory).')
    vad_parser.add_argument('--template', type=lambda x: is_valid_file(vad_parser, x),
                        help='Template .etf file for generating output .eafs.'
                       )
    vad_parser.add_argument('--overlap_behavior', choices=['keep_source', 'keep_matrix', 'keep_both'], default='keep_source', nargs='?',
                    help='Behavior for treating segments that overlap between the two .eafs. '
                        +'If keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE. '\
                        +'If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
                        +'If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not. '\
                )
    vad_parser.add_argument('-v', '--vad', type=lambda x: is_valid_file(vad_parser, x),
                        help='Filepath for .vad file linked to the recording. '\
                            +'If passed, reads VAD data from file instead of running rVAD_fast.'
                       )
    vad_parser.add_argument('-t', '--tier',
                        help='Tier label to add annotations to. If none specified uses `default-lt`'\
                       )
    add_batch_args(vad_parser)

def init_eaf_data_parser(eaf_data_parser: argparse.ArgumentParser) -> None:
    eaf_data_parser.add_argument('EAF_FILEPATH', type=lambda x: is_valid_file(eaf_data_parser, x),
        help='.eaf file to process metadata for.'
    )
    eaf_data_parser.add_argument('-o', '--output', type=lambda x: is_valid_file(eaf_data_parser, x),
        help='.csv path to output to. Default behavior is to use same filename as EAF_FILEPATH, '\
            +'or if EAF_FILEPATH is a directory, create a file named eaf_data.csv in the given directory.'
    )
    eaf_data_parser.add_argument('-t', '--tier',
        help='Tier label to read eaf_data from. If none specified read from all tiers.'
    )
    eaf_data_parser.add_argument('-m', '--media',
        help='Path to media file, if different than media path specified in .eaf file.'
    )
    add_batch_args(eaf_data_parser)

def init_split_data_parser(split_data_parser: argparse.ArgumentParser) -> None:
    split_data_parser.add_argument('EAF_DATA', type=lambda x: is_valid_file(split_data_parser, x),
        help='Path to eaf_data.csv (output of eaf_data command) to generate data splits from.'
    )
    split_data_parser.add_argument('OUT_DIR', type=lambda x: is_valid_dir(split_data_parser, x),
        help='.csv path to output to. Default behavior is to use same filename as EAF_FILEPATH, '\
            +'or if EAF_FILEPATH is a directory, create a file named split_data.csv in the given directory.'
    )
    split_data_parser.add_argument('-s', '--splitsize', type=float, nargs=3,
        help='Size of training, validation and test splits, given as floats where 0 < size < 1. Default is (0.8, 0.1, 0.1).'
    )

def init_snip_audio_parser(snip_audio_parser: argparse.ArgumentParser) -> None:
    snip_audio_parser.add_argument('ANNOTATIONS', type=lambda x: is_valid_file(snip_audio_parser, x),
        help='Path to eaf_data.csv or .eaf file.'
    )
    snip_audio_parser.add_argument('OUT_DIR', type=lambda x: is_valid_dir(snip_audio_parser, x),
        help='Path to folder to save output in.'
    )
    snip_audio_parser.add_argument('-t', '--tier', help='Tier name to read annotations from (default is to use all tiers).')

def init_lid_labels_parser(lid_labels_parser: argparse.ArgumentParser) -> None:
    lid_labels_parser.add_argument('ANNOTATIONS', type=lambda x: is_valid_file(lid_labels_parser, x),
        help='Path to eaf_data.csv or .eaf file.'
    )
    lid_labels_parser.add_argument('-tl', '--targetlang', help='ISO code or other unique identifier for target (fieldwork) language.')
    lid_labels_parser.add_argument('-ml', '--metalang', help='ISO code or other unique identifier for meta language.')
    lid_labels_parser.add_argument('--target_labels', help="Strings to map to target language, or '*' to map all strings except those specified by --meta_labels.", nargs='+')
    lid_labels_parser.add_argument('--meta_labels', help="Strings to map to meta language, or '*' to map all strings except those specified by --target_labels.", nargs='+')
    lid_labels_parser.add_argument('-e', '--empty', choices=['target', 'meta', 'exclude'], help='Whether empty annotations should be mapped to target language, meta language, or excluded.')
    lid_labels_parser.add_argument('--toml', type=lambda x: is_valid_file(lid_labels_parser, x),
        help='Path to a .toml file with metadata for args for this script.'
    )
    lid_labels_parser.add_argument('out_path', type=lambda x: is_valid_file(lid_labels_parser, x),
        help='Path to .csv file to output to (default is to overwrite ANNOTATIONS).'
    )

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

def handle_merge(args: Mapping) -> int:
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

def handle_vad(args: Mapping) -> int:
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

def handle_trim(args: Mapping) -> int:
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

def handle_eaf_data(args: Mapping) -> int:
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
    overwrite = args['overwrite']

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
            overwrite=overwrite,
        )
        df = pd.concat(out.values())
    else:
        df = eaf_data(
            eaf=eaf_fp,
            tier=tier,
            media=media
        )

    df.to_csv(out_fp)

    return 0

def handle_split_data(args: Mapping) -> int:
    eaf_data_fp = args['EAF_DATA']
    out_dir = args['OUT_DIR']
    splitsize = args['splitsize'] or (0.8, 0.1, 0.1)

    metadata_fp = split_data(
        eaf_data=eaf_data_fp,
        out_dir=out_dir,
        splitsize=splitsize
    )

    print('Metadata for training split saved to', metadata_fp)

    return 0

def handle_snip_audio(args: Mapping) -> int:
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

def handle_lid_labels(args: Mapping) -> int:
    annotations = args['ANNOTATIONS']
    out_path = args['out_path']
    targetlang = args['targetlang']
    metalang = args['metalang']
    target_labels = args['target_labels']
    meta_labels = args['meta_labels']
    empty = args['empty']
    toml = args['toml']

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
        toml=toml
    )

    lid_df.to_csv(out_path, index=False)

    return 0

def get_eaf_outpath(data_file: str, out_folder: str) -> str:
    """
    Returns path to .eaf file with same name as data_file
    in directory out_folder.
    """
    return os.path.join(out_folder, Path(data_file).stem + '.eaf')


def save_eaf_batch(data_file: str, out, out_folder: str) -> str:
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
    init_lid_labels_parser(lid_labels_parser)

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
    return 1

if __name__ == '__main__':
    main()