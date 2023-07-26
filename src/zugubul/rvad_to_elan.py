#!/usr/bin/env python

from pympi import Elan
from typing import Literal, List
import sys

"""
Creates a .eaf file with empty annotations for each speech segment
indicated in a .vad file.

Usage: rvad_to_elan.py WAV_FILEPATH VAD_FILEPATH EAF_FILEPATH (DIALECT)
- DIALECT arg is optional, can either be 'seg' or 'frame',
'seg' for a .vad file with start and endpoints for speech segments,
'frame' for a .vad file with 0 or 1 for each frame
"""

def read_rvad_segs(fp: str, dialect: Literal['seg', 'frame']='seg') -> List[tuple]:
    """
    Read .vad file and return list of tuples
    containing start and end of speech segments.
    By default assumes .vad file is of 'seg' dialect,
    containing start and end frames for segments.
    Use 'frame' dialect for .vad files containing 1s and 0s for each frame.
    """
    if dialect == 'seg':
        with open(fp, 'r') as f:
            segs = f.readlines()
        segs = [int(s) for s in segs]
        startpoints = segs[:len(segs)//2]
        endpoints = segs[len(segs)//2:]
    else:
        # dialect == 'frame'
        with open(fp, 'r') as f:
            frames = f.readlines()
        frames = [int(f) for f in frames]
        in_seg = False
        startpoints = []
        endpoints = []
        for i, f in enumerate(frames):
            if in_seg:
                if f == 0:
                    # end of segment
                    in_seg = False
                    endpoints.append(i)
            elif f == 1:
                # beginning of segment
                in_seg = True
                startpoints.append(i+1)
    return [(start, end) for start, end in zip(startpoints, endpoints)]


def rvad_segs_to_ms(segs: List) -> List:
    """
    Convert list of tuples with frame indices
    to list of same shape with time values in ms.
    """
    frame_width = 10
    return [(start*frame_width, end*frame_width) for start, end in segs]

def main():
    wav_fp = sys.argv[1]
    rvad_fp = sys.argv[2]
    eaf_fp = sys.argv[3]
    dialect = sys.argv[4] if len(sys.argv) > 4 else 'seg'

    segs = read_rvad_segs(rvad_fp, dialect)
    times = rvad_segs_to_ms(segs)
    eaf = Elan.Eaf()
    eaf.add_linked_file(wav_fp)
    # TODO: enable retrieving ELAN tier info from .cfg file
    eaf.add_tier('default-lt')
    for start, end in times:
        eaf.add_annotation('default-lt', start, end)
    Elan.to_eaf(eaf_fp, eaf)
    

if __name__ == '__main__':
    main()