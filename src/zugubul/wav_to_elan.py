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
    wav_fp = sys.argv[1]
    rvad_fp = sys.argv[2]
    eaf_fp = sys.argv[3]
    os.system(f'python zugubul/rVAD/rVAD_fast.py {wav_fp} {rvad_fp} ')
    os.system(f'python zugubul/src/rvad_to_elan.py {wav_fp} {rvad_fp} {eaf_fp} frame')

if __name__ == '__main__':
    main()