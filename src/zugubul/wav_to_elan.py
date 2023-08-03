#!/usr/bin/env python

"""
Usage: wav_to_elan WAV_FILEPATH EAF_FILEPATH (EAF_OUT_FILEPATH) (OVERLAP)

Runs rVAD_fast (Copyright (c) 2022 Zheng-Hua Tan and Achintya Kumar Sarkar)
on WAV_FILEPATH then generates elan annotations of speech segments at EAF_FILEPATH
If both EAF_FILEPATH and EAF_OUT_FILEPATH are provided,
merges the generated Eaf object with a preexisting eaf at EAF_FILEPATH,
using the provided OVERLAP value (or default 200ms), see documentation at elan_tools.py.
"""

import sys
from zugubul.rvad_to_elan import label_speech_segments
from zugubul.elan_tools import merge

def main():
    try:
        wav_fp = sys.argv[1]
        eaf_fp = sys.argv[2]
        eaf_fp2 = sys.argv[3] if len(sys.argv) > 3 else None
        try:
            overlap = float(sys.argv[4]) if len(sys.argv) > 4 else 200
        except ValueError():
            print('Gap must be a number.')
            overlap = 200
    except IndexError():
        print('Usage: wav_to_elan WAV_FILEPATH EAF_FILEPATH (EAF_OUT_FILEPATH) (OVERLAP)')
        return
    eaf = label_speech_segments(wav_fp)
    if eaf_fp2:
        eaf = merge(eaf_fp, eaf, overlap=overlap)
        eaf.to_file(eaf_fp2)
    else:
        eaf.to_file(eaf_fp)

if __name__ == '__main__':
    main()