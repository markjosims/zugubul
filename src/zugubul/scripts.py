#!/usr/bin/env python

"""
Usage: scripts COMMAND ARGS...
If zugubul package installed, zugubul COMMAND ARGS...
Run scripts/zugubul -h for more information.
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
import argparse
import warnings
from typing import Optional, Sequence
from zugubul.rvad_to_elan import label_speech_segments, RvadError
from zugubul.elan_tools import merge, trim
from zugubul.utils import is_valid_file, file_in_valid_dir, batch_funct
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
    merge_parser.add_argument('--exclude-overlap', action='store_true',
                        help='Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
                         +'Default behavior is to add.'
                    )
    merge_parser.add_argument('-b', '--batch', action='store_true',
                        help='Run on all wav files in a given folder '\
                            +'(EAF1_FILEPATH and EAF2_FILEPATH must be paths to folders and not files)'
                    )
    merge_parser.add_argument('-r', '--recursive', action='store_true',
                        help='If running a batch process, recurse over all subdirectories in WAV_FILEPATH.'
                    )
    merge_parser.add_argument('--overwrite', action='store_true',
                        help='If running batch process, overwrite applicable files in output directory. '\
                            +'Default behavior is to skip merging a file already present in output.'
                    )
    
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
    trim_parser.add_argument('-b', '--batch', action='store_true',
                        help='Run on all wav files in a given folder '\
                            +'(EAF1_FILEPATH and EAF2_FILEPATH must be paths to folders and not files)'
                       )
    trim_parser.add_argument('-r', '--recursive', action='store_true',
                        help='If running a batch process, recurse over all subdirectories in WAV_FILEPATH.'
                       )

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
    vad_parser.add_argument('--exclude-overlap', action='store_true',
                    help='Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX. '\
                        +'Default behavior is to add.'
                )
    vad_parser.add_argument('-b', '--batch', action='store_true',
                        help='Run on all wav files in a given folder '\
                            +'(WAV_FILEPATH and EAF_FILEPATH must be paths to folders and not files)'
                       )
    vad_parser.add_argument('-r', '--recursive', action='store_true',
                        help='If running a batch process, recurse over all subdirectories in WAV_FILEPATH.'
                       )
    vad_parser.add_argument('--overwrite', action='store_true',
                        help='If running a batch process, overwrite files already in destination folder.'
                       )
    vad_parser.add_argument('-v', '--vad', type=lambda x: is_valid_file(vad_parser, x),
                        help='Filepath for .vad file linked to the recording. '\
                            +'If passed, reads VAD data from file instead of running rVAD_fast.'
                       )
    vad_parser.add_argument('-t', '--tier', required=False,
                        help='Tier label to add annotations to. If none specified uses `default-lt`'\
                       )

def handle_merge(args):
    eaf_source = args['EAF_SOURCE']
    eaf_matrix = args['EAF_MATRIX']
    eaf_out = args['eaf_out']
    tier = args['tier']
    exclude_overlap = args['exclude-overlap']
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
                'exclude_overlap': exclude_overlap,
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
            exclude_overlap=exclude_overlap
        )
        eaf.to_file(eaf_out)

def handle_vad(args):
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
        eaf.to_file(eaf_fp)
    if eaf_source:
        # if eaf_source is passed, perform merge
        args['EAF_MATRIX'] = eaf_fp
        args['EAF_SOURCE'] = eaf_source
        args['eaf_out'] = eaf_fp
        handle_merge(args)

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

def handle_trim(args):
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
        eaf.to_file(out_fp)

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
        out.to_file(out_path)
        return out_path


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description='Tools for automatic transcription of audio files with ELAN.')
    subparsers = parser.add_subparsers(dest='COMMAND')
    
    merge_parser = subparsers.add_parser('merge', help='Add all annotations in eaf2 not present in eaf1')
    merge_parser = init_merge_parser(merge_parser)

    trim_parser = subparsers.add_parser('trim',
                        help='Remove all empty annotations in a given tier, '\
                            +'or all annotations with a given value indicated by the STOPWORD argument.'
                    )        
    trim_parser = init_trim_parser(trim_parser)

    vad_parsers = subparsers.add_parser('vad',
                        help='Create ELAN file with annotations corresponding to detected speech segments in recording.'
                    )
    vad_parsers = init_vad_parser(vad_parsers)

    args = vars(parser.parse_args(argv))

    command = args['COMMAND']
    if command == 'trim':
        handle_trim(args)
    elif command == 'merge':
        handle_merge(args)
    elif command == 'vad':
        handle_vad(args)

if __name__ == '__main__':
    main()