#!/usr/bin/env python

import unittest
import numpy as np
import random
#import torch
import math
from linetimer import CodeTimer
from scipy.signal import lfilter
from copy import deepcopy
from rVAD import speechproc

# variables and constants required by rVAD functions
# unceremoniously dumped here in the preamble
num_frameshifts = random.randint(2000, 10000)
samples_per_frame = random.randint(500, 5000)
samples_per_frameshift = int(np.floor(samples_per_frame / 2.5))
len_data = int(num_frameshifts * samples_per_frameshift + (samples_per_frame+samples_per_frameshift))
fdata = np.random.randint(low=-100, high=100, size=len_data).astype('float64')
ft = np.random.randint(low=-100, high=100, size=num_frameshifts)
ENERGYFLOOR = np.exp(-50)

Dexpl, Dexpr=18, 18
Dsmth=np.zeros(num_frameshifts, dtype='float64'); Dsmth=np.insert(Dsmth,0,'inf')    

ftThres=0.5
vadThres=0.4
opts=1

pv01=np.zeros(num_frameshifts)
pv01[np.less_equal(ft, ftThres)]=1
pitch=deepcopy(ft)

pvblk=speechproc.pitchblockdetect(pv01, pitch, num_frameshifts, opts)

fdata_=deepcopy(fdata)
pv01_=deepcopy(pv01)
pvblk_=deepcopy(pvblk)   

fdata_=np.insert(fdata_.astype('float64'),0,'inf')
pv01_=np.insert(pv01_.astype('float64'),0,'inf')
pvblk_=np.insert(pvblk_.astype('float64'),0,'inf')

e=np.zeros(num_frameshifts,  dtype='float64')
e=np.insert(e,0,'inf')

D=np.zeros(num_frameshifts); D=np.insert(D,0,'inf')
postsnr=np.zeros(num_frameshifts, dtype='float64'); postsnr=np.insert(postsnr,0,'inf')
snre_vad=np.zeros(num_frameshifts); snre_vad=np.insert(snre_vad,0,'inf')


class TestSnreHighEnergy(unittest.TestCase):
    """
    Test code blocks from the snre_highenergy function from speechproc.py
    """
    def test_first_forloop(self):
        """
        Test the first forloop from snre_highenergy.
        Note same as first forloop for snre_vad,
        so this code block can be used for both.
        """
        e=np.zeros(num_frameshifts,  dtype='float64')
        e=np.insert(e,0,'inf')  
        
        def cpu_forloop(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift):
            e_cpu = deepcopy(e)
            for i in range(1, num_frameshifts+1):
                for j in range(1, samples_per_frame+1):
                    frame_i = int((i-1)*samples_per_frameshift+j)
                    e_cpu[i]=e_cpu[i]+np.square(fdata[frame_i])
            
                if np.less_equal(e_cpu[i], ENERGYFLOOR):
                    e_cpu[i]=ENERGYFLOOR
            return e_cpu
        
        def gpu_matrix(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift):
            max_frame = (num_frameshifts-1)*samples_per_frameshift+samples_per_frame+1
            fdata = fdata[1:max_frame]
            # fdata contains all speech frames used for analysis
            # need to reshape into matrix of overlapping input frames
            # frame length = 25ms, stride length = 10ms
            # need one row for each stride, so make three matrices
            # first is for first 10ms, second for second 10ms
            # third is for last 5ms

            first_partition_end = num_frameshifts*samples_per_frameshift
            first_partition = fdata[:first_partition_end].copy().reshape(num_frameshifts, samples_per_frameshift)

            scnd_partition_start = samples_per_frameshift
            scnd_partition_end = (num_frameshifts+1)*samples_per_frameshift
            scnd_partition = fdata[scnd_partition_start:scnd_partition_end].copy().reshape(num_frameshifts, samples_per_frameshift)

            margin = samples_per_frame-(samples_per_frameshift*2)
            third_partition_start = samples_per_frameshift*2
            third_partition = np.resize(fdata[third_partition_start:], (num_frameshifts, samples_per_frameshift))
            third_partition = third_partition[:,:margin]

            fdata_reshaped = np.concatenate([first_partition, scnd_partition, third_partition], axis=1)
            fdata_gpu = torch.tensor(fdata_reshaped).cuda()

            squares = torch.square(fdata_gpu)
            e_gpu = torch.sum(squares, 1)
            e_gpu[e_gpu <= ENERGYFLOOR] = ENERGYFLOOR
            e_np = np.array(e_gpu.cpu()).astype('float64')
            e_np = np.insert(e_np,0,'inf')

            return e_np

        forloop_timer = CodeTimer('CPU forloop', unit='s')
        cpu_timer = CodeTimer('CPU matrix', unit='s')
        # gpu_timer = CodeTimer('GPU matrix', unit='s')
        with forloop_timer:
            forloop_out = cpu_forloop(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift)
            
        # with gpu_timer:
        #     gpu_out = gpu_matrix(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift)

        with cpu_timer:
            cpu_out = speechproc.estimate_energy(fdata, num_frameshifts, samples_per_frame, samples_per_frameshift, ENERGYFLOOR)

        cpu_time = cpu_timer.took
        forloop_time = forloop_timer.took
        # gpu_time = gpu_timer.took
        # print(f'GPU matrix function was {forloop_time/gpu_time} times faster than for loop')
        print(f'CPU matrix function was {forloop_time/cpu_time} times faster than for loop')

        # self.assertTrue(np.array_equal(forloop_out, gpu_out) and np.array_equal(forloop_out, cpu_out))
        #self.assertTrue(np.allclose(forloop_out, cpu_out))
        self.assertTrue(np.array_equal(forloop_out, cpu_out))

class TestOutFiles(unittest.TestCase):
    def test_dendi1(self):
        rvad_fp = r'C:\projects\zugubul\tests\test_dendi1_frames.vad'
        rvad_refactor_fp = r'C:\projects\zugubul\tests\test_dendi1_frames_refactor.vad'

        rvad_frames = np.loadtxt(rvad_fp)
        rvad_refactor_frames = np.loadtxt(rvad_refactor_fp)

        self.assertTrue(np.array_equal(rvad_frames, rvad_refactor_frames))

if __name__ == '__main__':
    unittest.main()