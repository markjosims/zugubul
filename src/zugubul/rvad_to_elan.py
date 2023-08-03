#!/usr/bin/env python

from pympi import Elan
from typing import Literal, List, Optional, Callable
from rVAD.rVAD_fast import rVAD_fast, frames_to_segs
from zugubul.utils import batch_funct
import sys
import os
import numpy as np

"""
Creates a .eaf file with empty annotations for each speech segment
indicated in a .vad file.

Usage: rvad_to_elan WAV_FILEPATH VAD_FILEPATH EAF_FILEPATH (DIALECT)
- DIALECT arg is optional, can either be 'seg' or 'frame',
'seg' for a .vad file with start and endpoints for speech segments,
'frame' for a .vad file with 0 or 1 for each frame
"""

def read_rvad_segs(vad_fp: Optional[str] = None, wav_fp: Optional[str] = None, dialect: Literal['seg', 'frame']='seg') -> List[tuple]:
    """
    Read .vad file from vad_fp and return list of tuples
    containing start and end of speech segments.
    By default assumes .vad file is of 'seg' dialect,
    containing start and end frames for segments.
    Use 'frame' dialect for .vad files containing 1s and 0s for each frame.
    If vad_fp is None, run rVAD_fast on wav_fp
    """
    if vad_fp:
        data = np.loadtxt(vad_fp)
    elif wav_fp:
        data = rVAD_fast(wav_fp)
    else:
        raise ValueError('Either vad_fp or wav_fp must be provided.')
    if dialect == 'frame':
        data = frames_to_segs(data)
    midpoint = len(data)//2
    startpoints = data[:midpoint]
    endpoints = data[midpoint:]

    return [(int(start), int(end)) for start, end in zip(startpoints, endpoints)]


def rvad_segs_to_ms(segs: List) -> List:
    """
    Convert list of tuples with frame indices
    to list of same shape with time values in ms.
    """
    frame_width = 10
    return [(start*frame_width, end*frame_width) for start, end in segs]


def label_speech_segments(wav_fp: str, rvad_fp: Optional[str] = None, dialect: str = 'seg', save_funct: Optional[Callable] = None) -> Elan.Eaf:
    """
    Returns an Eaf object with empty annotations for each detected speech segment.
    If rvad_fp is passed, read speech segments from the associated .vad file,
    otherwise run rVAD_fast on wav file indicated by wav_fp.
    """
    if os.path.isdir(wav_fp):
        return batch_funct(label_speech_segments, wav_fp, '.wav', 'wav_fp', save_f=save_funct)

    if rvad_fp:
        segs = read_rvad_segs(vad_fp=rvad_fp, dialect=dialect)
    else:
        segs = read_rvad_segs(wav_fp=wav_fp)
    times = rvad_segs_to_ms(segs)
    eaf = Elan.Eaf()
    eaf.add_linked_file(wav_fp)
    # TODO: enable retrieving ELAN tier info from .etf, .cfg or .toml file
    eaf.add_tier('default-lt')
    for start, end in times:
        eaf.add_annotation('default-lt', start, end)
    return eaf


def main():
    wav_fp = sys.argv[1]
    rvad_fp = sys.argv[2]
    eaf_fp = sys.argv[3]
    dialect = sys.argv[4] if len(sys.argv) > 4 else 'seg'

    eaf = label_speech_segments(wav_fp, rvad_fp, dialect)
    Elan.to_eaf(eaf_fp, eaf)
    

if __name__ == '__main__':
    main()