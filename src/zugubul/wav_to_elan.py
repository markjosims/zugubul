#!/usr/bin/env python

"""
Usage: wav_to_elan.py WAV_FILEPATH VAD_FILEPATH EAF_FILEPATH

Runs rVAD_fast.py (Copyright (c) 2022 Zheng-Hua Tan and Achintya Kumar Sarkar)
on WAV_FILEPATH, outputting speech segments to VAD_FILEPATH,
then runs rvad_to_elan.py and generates elan annotations at EAF_FILEPATH
"""

import os
import sys

def main():
    try:
        wav_fp = sys.argv[1]
        rvad_fp = sys.argv[2]
        eaf_in_fp = sys.argv[3]
        eaf_out_fp = sys.argv[4] if len(sys.argv) > 4 else eaf_in_fp
    except IndexError():
        print('Usage: wav_to_elan.py WAV_FILEPATH VAD_FILEPATH EAF_FILEPATH')
        return
    os.system(f'rvad {wav_fp} {rvad_fp}')
    os.system(f'rvad_to_elan {wav_fp} {rvad_fp} {eaf_out_fp} frame')
    if eaf_in_fp != eaf_out_fp:
        os.system(f'elan_tools merge {eaf_in_fp} {eaf_out_fp} {eaf_out_fp}')

if __name__ == '__main__':
    main()