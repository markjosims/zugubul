#!/usr/bin/env python

from pympi import Elan
from typing import Literal, List, Optional, Union, Dict
from pathlib import Path
import os
import json

from zugubul.main import TORCH, PYANNOTE
from zugubul.vad.rVAD import run_rVAD_fast
DEFAULT_VAD = 'rVAD'
VAD_METHODS = {'rVAD': run_rVAD_fast}
if TORCH:
    from zugubul.vad.silero_vad import run_silero_vad
    DEFAULT_VAD = 'silero'
    VAD_METHODS['silero'] = run_silero_vad
if PYANNOTE:
    from zugubul.vad.pyannote import run_pyannote_vad
    DEFAULT_VAD = 'pyannote'
    VAD_METHODS['pyannote'] = run_pyannote_vad

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

# class RvadError(Exception):
#     def __init__(self, filepath: str, error: Optional[Exception] = None) -> Exception:
#         message = f'Error running voice activity detection on file {filepath}.'
#         if error:
#             message = message + f' Original exception: {error}'
#         self.message = message
#         super().__init__(self.message)

# def run_rVAD_safe(wav_fp):
#     try:
#         data = run_rVAD_fast(wav_fp)
#     except Exception as error:
#         raise RvadError(wav_fp, error=error)

#     return data


# TODO: add option to run process_length before returning
def label_speech_segments(
        wav_fp: str,
        vad_fp: Optional[str] = None,
        tier: Union[str, List[str]] = 'default-lt',
        method: Literal['silero', 'rVAD', 'pyannote'] = DEFAULT_VAD,
        etf: Optional[Union[str, Elan.Eaf]] = None
    ) -> Elan.Eaf:
    """
    Returns an Eaf object with empty annotations for each detected speech segment.
    If vad_fp is passed, read speech segments from the associated .vad file,
    otherwise run rVAD_fast on wav file indicated by wav_fp.
    If etf is passed, add all tiers from etf file.
    """

    # avoid issues w/ adding linked files in pympi
    wav_fp = wav_fp.replace('.WAV', '.wav')

    if type(tier) is str:
        tier = [tier,]

    vad_funct = VAD_METHODS[method]

    if vad_fp:
        if os.path.isdir(vad_fp):
            # if vad_fp is dir, assume .vad file has same name as wav file
            # used in batch processing
            wav_stem = Path(wav_fp).stem
            with open(vad_fp/wav_stem+'.json') as f:
                segs = json.load(f)
        segs = vad_funct(vad_fp=vad_fp)
    else:
        segs = vad_funct(wav_fp=wav_fp)

    if etf:
        if (type(etf) is not Elan.Eaf):
            eaf = Elan.Eaf(etf)
        else:
            eaf = etf
        etf_tiers = eaf.get_tier_names()
        for t in tier:
            if t not in etf_tiers:
                raise ValueError(f'tier argument must correspond to tier in .etf file. {tier=}, {etf_tiers=}')
    else:
        eaf = Elan.Eaf()
        for t in tier:
            eaf.add_tier(t)
    eaf.add_linked_file(wav_fp)
    
    for seg in segs:
        start, end = seg['start'], seg['end']
        for t in tier:
            eaf.add_annotation(t, start, end)
    return eaf