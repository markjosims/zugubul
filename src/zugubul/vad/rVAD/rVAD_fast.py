from __future__ import division
import numpy as np
import os
import sys
import json
from scipy.signal import lfilter
from zugubul.vad.rVAD import speechproc
from copy import deepcopy
from zugubul.utils import batch_funct
from typing import Optional, Literal, Callable, List, Dict, Union

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, vol. 59, pp. 1-21, 2020. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# Version: 2.0
# 02 Dec 2017, Achintya Kumar Sarkar and Zheng-Hua Tan

# Usage: python rVAD_fast_2.0.py inWaveFile  outputVadLabel

# MODIFICATION BY Mark J Simmons, 3 August 2023
# Extracted code for saving vad file to a separate method
# Added logic to allow for either passing a path to a wav file
# or an array containing vad data
def save_vad_file(vad_fp: str, wav_fp: Optional[str] = None, vad_array: Optional[np.ndarray] = None) -> str:
    """
    If wav_fp is passed, perform voice activity detection on the indicated wav file and save output to vad_fp
    Else, save vad_array to vad_fp.
    Returns vad_fp.
    """
    if wav_fp:
        vad_array = run_rVAD_fast(wav_fp, dialect='seg')
    elif vad_array is None:
        raise ValueError('Either wav_fp or vad_array must be non-null.')
    vad_json = rVAD_to_json(segs=vad_array)
    with open(vad_fp, 'w') as f:
        json.dump(vad_json, f)
    print("%s --> %s " %(wav_fp, vad_fp))
    return vad_fp
# END MODIFICATION


def frames_to_segs(frames: np.ndarray) -> np.ndarray:
    """
    Takes an array of 0 and 1 frame values and returns an array with segment start and endpoints
    """
    diff = np.diff(frames)
    startpoints = np.where(diff==1)[0]
    # np.diff function will miss speech segment edges on the first or last frame of the array
    if frames[0] == 1:
        startpoints = np.insert(startpoints, 0, 0)
    endpoints = np.where(diff==-1)[0]
    if frames[-1] == 1:
        endpoints = np.insert(endpoints, len(endpoints), len(endpoints))
    startpoints+=2
    endpoints+=1
    return np.concatenate([startpoints, endpoints]).astype('int')

def rVAD_to_json(
        frames: Optional[np.ndarray]=None,
        segs: Optional[np.ndarray]=None,
        convert_to_ms: bool = True,
    ) -> List[Dict[str, int]]:
    if segs is None:
        if frames is None:
            raise ValueError('Either frames or segs must be passed')
        segs = frames_to_segs(frames)
    
    midpoint = len(segs)//2
    startpoints = segs[:midpoint]
    endpoints = segs[midpoint:]

    if convert_to_ms:
        frame_width = 10
        return [(int(start*frame_width), int(end*frame_width)) for start, end in zip(startpoints, endpoints)]

    return [{'start': int(start), 'end': int(end)} for start, end in zip(startpoints, endpoints)]

def run_rVAD_fast(
        finwav: str,
        dialect: Literal['seg', 'frame', 'json']='json',
        save_funct: Optional[Callable]= None,
    ) -> Union[np.ndarray, List[Dict[str, int]]]:
    """
    Run rVAD_fast on the wav file indicated by finwav,
    return array indicating segments of speech.
    If "dialect" is frame, return array of 0s and 1s for each frame (10ms) of audio,
    where 1 indicates speech and 0 noise or silence.
    If "dialect" is seg, return array of startpoints and endpoints for speech segments.
    If "dialect" is json, return a list of dicts containing start and endpoints.
    """

    # MODIFICATION BY Mark J Simmons, 3 August 2023
    # call batch_funct if directory path is passed
    # default to saving vad file to same directory and filename as wav files
    # with .wav suffix replaced with .vad
    # if save_funct is passed, use that instead
    if os.path.isdir(finwav):
        if not save_funct:
            save_funct = lambda data_file, array: save_vad_file(
                vad_array=array,
                vad_fp=str(data_file).replace('.wav', '.vad')
            )
        return batch_funct(run_rVAD_fast, finwav, '.wav', 'finwav', save_f=save_funct)
    # END MODIFICATION
    winlen, frame_shift, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres=0.5; vadThres=0.4
    opts=1

    sampling_rate, data = speechproc.speech_wave(finwav)   
    ft, samples_per_frame, samples_per_frameshift, num_frameshifts =speechproc.sflux(data, sampling_rate, winlen, frame_shift, nftt)


    # --spectral flatness --
    pv01=np.zeros(num_frameshifts)
    pv01[np.less_equal(ft, ftThres)]=1 
    pitch=deepcopy(ft)

    pvblk=speechproc.pitchblockdetect(pv01, pitch, num_frameshifts, opts)


    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b=np.array([0.9770,   -0.9770])
    a=np.array([1.0000,   -0.9540])
    fdata=lfilter(b, a, data, axis=0)


    #--pass 1--
    noise_samp, noise_seg, n_noise_samp=speechproc.snre_highenergy(fdata, num_frameshifts, samples_per_frame, samples_per_frameshift, ENERGYFLOOR, pv01, pvblk)

    #sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j,0]),  int(noise_samp[j,1]) +1)] = 0 


    vad_out=speechproc.snre_vad(fdata,  num_frameshifts, samples_per_frame, samples_per_frameshift, ENERGYFLOOR, pv01, pvblk, vadThres)

    if dialect == 'seg':
        vad_out = frames_to_segs(vad_out)
    elif dialect == 'json':
        vad_out = rVAD_to_json(frames=vad_out)

    return vad_out

if __name__ == '__main__':
    # MODIFICATION BY Mark J Simmons 7/25/2023
    # Indicate original copyright upon execution
    print("rVAD fast 2.0: Copyright (c) 2022 Zheng-Hua Tan and Achintya Kumar Sarkar")
    # END MODIFICATION
    finwav=str(sys.argv[1])
    fvad=str(sys.argv[2])
    # MODIFICATION BY Mark J Simmons 8/3/2023
    if os.path.isdir(finwav) and os.path.isdir(fvad):
        save_funct = lambda data_file, array: save_vad_file(
            vad_array=array,
            vad_fp=str(data_file)
                .replace('.wav', '.vad')
                .replace(finwav, fvad)
        )
        run_rVAD_fast(finwav, save_funct=save_funct)
    else:
        vad_out = run_rVAD_fast(finwav)
        save_vad_file(vad_array=vad_out, wav_fp=finwav)
    # END MODIFICATION