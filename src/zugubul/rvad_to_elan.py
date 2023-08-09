#!/usr/bin/env python

from pympi import Elan
from typing import Literal, List, Optional, Union
from rVAD.rVAD_fast import rVAD_fast, frames_to_segs
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

class RvadError(Exception):
    def __init__(self, filepath: str, error: Optional[Exception] = None) -> Exception:
        message = f'Error running voice activity detection on file {filepath}.'
        if error:
            message = message + f' Original exception: {error}'
        self.message = message
        super().__init__(self.message)


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
        try:
            data = rVAD_fast(wav_fp)
        except Exception as error:
            raise RvadError(wav_fp, error=error)
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

    return [(int(start), int(end)) for start, end in zip(startpoints, endpoints) if start!=end]


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
        tier: str = 'default-lt',
        dialect: str = 'seg',
        etf: Optional[Union[str, Elan.Eaf]] = None
    ) -> Elan.Eaf:
    """
    Returns an Eaf object with empty annotations for each detected speech segment.
    If rvad_fp is passed, read speech segments from the associated .vad file,
    otherwise run rVAD_fast on wav file indicated by wav_fp.
    If etf is passed, add all tiers from etf file.
    """

    # avoid issues w/ adding linked files in pympi
    wav_fp = wav_fp.replace('.WAV', '.wav')

    if rvad_fp:
        if os.path.isdir(rvad_fp):
            # if rvad_fp is dir, assume .vad file has same name as wav file
            wav_stem = Path(wav_fp).stem
            rvad_fp = os.path.join(rvad_fp, wav_stem+'.vad')
        segs = read_rvad_segs(vad_fp=rvad_fp, dialect=dialect)
    else:
        segs = read_rvad_segs(wav_fp=wav_fp)
    times = rvad_segs_to_ms(segs)

    if etf:
        if type(etf) is str:
            eaf = Elan.Eaf(etf)
        else:
            eaf = etf
        etf_tiers = eaf.get_tier_names()
        if tier not in etf_tiers:
            raise ValueError(f'tier argument must correspond to tier in .etf file. {tier=}, {etf_tiers=}')
    else:
        eaf = Elan.Eaf()
        eaf.add_tier(tier)
    eaf.add_linked_file(wav_fp)
    
    for start, end in times:
        eaf.add_annotation(tier, start, end)
    return eaf