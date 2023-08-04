#!/usr/bin/env python

"""
Usage: wav_to_elan WAV_FILEPATH EAF_FILEPATH (-s, --source EAF_OUT_FILEPATH) (--overlap OVERLAP) (-r)
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
from typing import Optional, Sequence
from pympi import Elan
from zugubul.rvad_to_elan import label_speech_segments
from zugubul.elan_tools import merge, trim
from zugubul.utils import is_valid_file, file_in_valid_dir


def init_merge_parser(merge_parser: argparse.ArgumentParser):
    merge_parser.add_argument('EAF1_FILEPATH', type=lambda x: is_valid_file(merge_parser, x),
                        help='Filepath for eaf file to output to (or parent directory of eafs).'
                    )
    merge_parser.add_argument('EAF2_FILEPATH', type=lambda x: is_valid_file(merge_parser, x),
                        help='Filepath for eaf file to take output from (or parent directory of eafs).'
                    )
    merge_parser.add_argument('-o', '--output_eaf', type=lambda x: file_in_valid_dir(merge_parser, x),
                        help='Filepath for eaf file to output to (or parent directory of eafs).'
                    )
    merge_parser.add_argument('-t', '--tier', required=False, nargs='+',
                        help='Remove annotations on tier(s). By default removes annotations for all tiers. '\
                         +'For multiple tiers write tier names separated by space.'
                    ) 
    merge_parser.add_argument('--overlap', type=int, default=200,
                        help='An overlapping annotation will only be added '\
                            +'if the non-overlapping portion is longer than the value passed (in ms). '\
                            +'Default 200ms.'
                    )
    merge_parser.add_argument('-b', '--batch', action='store_true',
                        help='Run on all wav files in a given folder '\
                            +'(EAF1_FILEPATH and EAF2_FILEPATH must be paths to folders and not files)'
                    )
    merge_parser.add_argument('-r', '--recursive', action='store_true',
                        help='If running a batch process, recurse over all subdirectories in WAV_FILEPATH.'
                    )
    
def init_trim_parser(trim_parser: argparse.ArgumentParser):
    trim_parser.add_argument('EAF_FILEPATH', type=lambda x: is_valid_file(trim_parser, x),
                        help='Filepath for eaf file to trim (or parent directory of eafs).'
                    )
    trim_parser.add_argument('-o', '--output_eaf', type=lambda x: file_in_valid_dir(trim_parser, x),
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
    vad_parser.add_argument('-s', '--source', help='Source .eaf file to merge output into (or parent directory).')
    vad_parser.add_argument('--overlap', type=int, default=200,
                        help='If source is passed, an overlapping annotation will only be added '\
                            +'if the non-overlapping portion is longer than the value passed (in ms). '\
                            +'Default 200ms.'
                       )
    vad_parser.add_argument('-b', '--batch', action='store_true',
                        help='Run on all wav files in a given folder '\
                            +'(WAV_FILEPATH and EAF_FILEPATH must be paths to folders and not files)'
                       )
    vad_parser.add_argument('-r', '--recursive', action='store_true',
                        help='If running a batch process, recurse over all subdirectories in WAV_FILEPATH.'
                       )
    vad_parser.add_argument('-v', '--vad', type=lambda x: is_valid_file(vad_parser, x),
                        help='Filepath for .vad file linked to the recording. '\
                            +'If passed, reads VAD data from file instead of running rVAD_fast.'
                       )

def handle_merge(args):
    eaf1 = args['EAF1_FILEPATH']
    eaf2 = args['EAF2_FILEPATH']
    out_fp = args['output_eaf']
    tiers = args['tier']
    overlap = args['overlap']
    batch = args['batch']
    recursive = args['recursive']

    if not out_fp:
        # default behavior is to override eaf2 file
        out_fp = eaf2

    if batch or os.path.isdir(eaf1):
        assert os.path.isdir(eaf1) and os.path.isdir(eaf2),\
        'If running batch process both EAF1_FILEPATH and EAF2_FILEPATH must be folders.'
        save_funct=lambda data_file, out: save_eaf_batch(data_file, out, out_fp)
        merge(eaf1, eaf2, tiers, overlap, recursive=recursive, save_funct=save_funct)
    else:
        eaf = merge(eaf1, eaf2, tiers, overlap)
        eaf.to_file(out_fp)

def handle_vad(args):
    wav_fp = args['WAV_FILEPATH']
    eaf_fp = args['EAF_FILEPATH']
    eaf_source = args['source']
    overlap = args['overlap']
    batch = args['batch']
    recursive = args['recursive']
    vad = args['vad']

    if batch or os.path.isdir(wav_fp):
        wav_fp, eaf_source = handle_vad_batch(wav_fp, eaf_fp, vad, eaf_source, recursive)
    else:        
        eaf = label_speech_segments(wav_fp)
        if eaf_source:
            eaf = merge(eaf_source, eaf, overlap=overlap)
        eaf.to_file(eaf_fp)

def handle_vad_batch(wav_fp: str, eaf_fp: str, vad: Optional[str] = None, eaf_source: Optional[str] = None, recursive: bool = False):
    """
    Helper for batch processing of handle_vad
    """
    assert os.path.isdir(wav_fp) and os.path.isdir(eaf_fp),\
            'If running batch process both WAV_FILEPATH and EAF_FILEPATH must be folders.'
    if eaf_source:
        assert os.path.isdir(eaf_source), 'If using SOURCE arg and running batch process, SOURCE must point to folder.'
    if vad:
        assert os.path.isdir(vad), 'If using VAD arg and running batch process, VAD must point to folder.'

    eafs = label_speech_segments(wav_fp=wav_fp, rvad_fp=vad, recursive=recursive)
    for wav_fp, eaf in eafs.items():
        file_stem = Path(wav_fp).stem
        if eaf_source:
            try:
                eaf_source_path = os.path.join(eaf_source, file_stem+'.eaf')
                eaf_source = Elan.Eaf(eaf_source_path)
            except FileNotFoundError():
                raise Warning(f'Source .eaf file not found for recording {wav_fp}')
            eaf = merge(eaf_source, eaf)
        out_path = os.path.join(eaf_fp, file_stem+'.eaf')
        eaf.to_file(out_path)
    return wav_fp, eaf_source

def handle_trim(args):
    eaf_fp = args['EAF_FILEPATH']
    out_fp = args['output_eaf']
    if not out_fp:
        # default behavior is to override input file
        out_fp = eaf_fp
    stopword = args['stopword']
    tiers = args['tier']
    batch = args['batch']
    recursive = args['recursive']

    if batch or os.path.isdir(eaf_fp):
        save_funct=lambda data_file, out: save_eaf_batch(data_file, out, out_fp)
        assert os.path.isdir(eaf_fp) and os.path.isdir(out_fp),\
        'If running batch process both EAF_FILEPATH and output_eaf (if provided) must be folders.'
        trim(eaf_fp, tiers, stopword, save_funct=save_funct, recursive=recursive)
    else:
        eaf = trim(eaf_fp, tiers, stopword)
        eaf.to_file(out_fp)

def save_eaf_batch(data_file: str, out, out_folder: str) -> str:
    """
    Saves output provided by batch_funct to filepath with same name as data_file
    in directory out_folder.
    Returns path output is saved to.
    """
    out_path = os.path.join(
        out_folder, os.path.split(data_file)[-1]
    )
    out.to_file(out_path)
    return out_path

def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description='Tools for automatic transcription of audio files with ELAN.')
    subparsers = parser.add_subparsers()
    
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