#!/usr/bin/env python

import unittest
import numpy as np
import random
import torch
from linetimer import CodeTimer
from scipy.signal import lfilter
from copy import deepcopy

class TestSnreHighEnergy(unittest.TestCase):
    def test_first_forloop(self):
        num_frameshifts = random.randint(200, 1000)
        samples_per_frame = random.randint(100, 1000)
        samples_per_frameshift = int(np.floor(samples_per_frame / 2.5))
        len_data = int(num_frameshifts * samples_per_frameshift + (samples_per_frame+samples_per_frameshift))
        fdata = np.random.randint(low=-100, high=100, size=len_data)
        ENERGYFLOOR = np.exp(-50)

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

        def cpu_matrix(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift):
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
            squares = fdata_reshaped**2

            e_cpu = squares.sum(axis=1)
            e_cpu[e_cpu <= ENERGYFLOOR] = ENERGYFLOOR
            e_np = e_cpu.astype('float64')
            e_np = np.insert(e_np,0,'inf')

            return e_np

        forloop_timer = CodeTimer('CPU forloop', unit='s')
        cpu_timer = CodeTimer('CPU matrix', unit='s')
        gpu_timer = CodeTimer('GPU matrix', unit='s')
        with forloop_timer:
            forloop_out = cpu_forloop(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift)
            
        with gpu_timer:
            gpu_out = gpu_matrix(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift)

        with cpu_timer:
            cpu_out = cpu_matrix(e, fdata, num_frameshifts, samples_per_frame, samples_per_frameshift)

        cpu_time = cpu_timer.took
        forloop_time = forloop_timer.took
        gpu_time = gpu_timer.took
        print(f'For loop function took {forloop_time} seconds')
        print(f'GPU matrix function took {gpu_time} seconds')
        print(f'CPU matrix function took {cpu_time} seconds')
        print(f'GPU matrix function was {forloop_time/gpu_time} times faster than for loop')
        print(f'CPU matrix function was {forloop_time/cpu_time} times faster than for loop')

        self.assertTrue(np.array_equal(forloop_out, gpu_out) and np.array_equal(forloop_out, cpu_out))

if __name__ == '__main__':
    unittest.main()