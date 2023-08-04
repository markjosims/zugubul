#!/usr/bin/env python

from pympi import Elan
from typing import Literal, List, Optional, Callable
from rVAD.rVAD_fast import rVAD_fast, frames_to_segs
from zugubul.utils import batch_funct
from pathlib import Path
import os
import numpy as np

docstr = """
Creates a .eaf file with empty annotations for each speech segment
indicated in a .vad file.
"""
"""
Usage: rvad_to_elan WAV_FILEPATH VAD_FILEPATH EAF_FILEPATH (DIALECT)
- DIALECT arg is optional, can either be 'seg' or 'frame',
'seg' for a .vad file with start and endpoints for speech segments,
'frame' for a .vad file with 0 or 1 for each frame
"""

def read_rvad_segs(vad_fp: Optional[str] = None, wav_fp: Optional[str] = None, dialect: Literal['seg', 'frame', 'auto']='auto') -> List[tuple]:
    """
    Read .vad file from vad_fp and return list of tuples
    containing start and end of speech segments.
    By default assumes .vad file is of 'seg' dialect,
    containing start and end frames for segments.
    Use 'frame' dialect for .vad files containing 1s and 0s for each frame.
    If dialect set to 'auto', detect from file.
    If vad_fp is None, run rVAD_fast on wav_fp
    """
    if vad_fp:
        data = np.loadtxt(vad_fp)
    elif wav_fp:
        data = rVAD_fast(wav_fp)
    else:
        raise ValueError('Either vad_fp or wav_fp must be provided.')
    if dialect == 'auto':
        if (data[0] in [0, 1]) and (data[1] in [0, 1]):
            dialect = 'frame'
        else:
            # dialect = 'seg'
            # nothing else needs to be done
            pass
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


def label_speech_segments(
        wav_fp: str,
        rvad_fp: Optional[str] = None,
        dialect: str = 'seg',
        save_funct: Optional[Callable] = None,
        recursive: bool = False
    ) -> Elan.Eaf:
    """
    Returns an Eaf object with empty annotations for each detected speech segment.
    If rvad_fp is passed, read speech segments from the associated .vad file,
    otherwise run rVAD_fast on wav file indicated by wav_fp.
    """
    if os.path.isdir(wav_fp):
        kwargs = {'rvad_fp': rvad_fp}
        return batch_funct(
            label_speech_segments,
            wav_fp,
            '.wav',
            'wav_fp',
            kwargs=kwargs,
            save_f=save_funct,
            recursive=recursive
        )

    if rvad_fp:
        if os.path.isdir(rvad_fp):
            # if rvad_fp is dir, assume .vad file has same name as wav file
            wav_stem = Path(wav_fp).stem
            rvad_fp = os.path.join(rvad_fp, wav_stem+'.vad')
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