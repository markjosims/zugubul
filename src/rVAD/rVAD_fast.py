from __future__ import division
import numpy as np
import pickle
import os
import sys
import math
import code
from scipy.signal import lfilter
from rVAD import speechproc
from copy import deepcopy

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, vol. 59, pp. 1-21, 2020. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# Version: 2.0
# 02 Dec 2017, Achintya Kumar Sarkar and Zheng-Hua Tan

# Usage: python rVAD_fast_2.0.py inWaveFile  outputVadLabel

def frames_to_segs(frames: np.ndarray) -> np.ndarray:
    """
    Takes an array of 0 and 1 frame values and returns an array with segment start and endpoints
    """
    diff = np.diff(frames)
    startpoints = np.where(diff==1)[0]
    if frames[0] == 1:
        startpoints = np.insert(startpoints, 0, 0)
    endpoints = np.where(diff==-1)[0]
    if frames[-1] == 1:
        endpoints = np.insert(endpoints, len(endpoints), len(endpoints))
    endpoints-=1
    return np.concatenate([startpoints, endpoints])



def main():
    # MODIFICATION BY Mark Simmons 7/25/2023
    # Indicate original copyright upon execution
    print("rVAD fast 2.0: Copyright (c) 2022 Zheng-Hua Tan and Achintya Kumar Sarkar")
    # END MODIFICATION

    winlen, frame_shift, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres=0.5; vadThres=0.4
    opts=1

    finwav=str(sys.argv[1])
    fvad=str(sys.argv[2])

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


    vad_frame=speechproc.snre_vad(fdata,  num_frameshifts, samples_per_frame, samples_per_frameshift, ENERGYFLOOR, pv01, pvblk, vadThres)

    vad_seg = vad_frame
    #vad_seg = frames_to_segs(vad_frame)

    np.savetxt(fvad, vad_seg.astype(int),  fmt='%i')
    print("%s --> %s " %(finwav, fvad))

    data=None; pv01=None; pitch=None; fdata=None; pvblk=None; vad_seg=None

if __name__ == '__main__':
    main()
     


