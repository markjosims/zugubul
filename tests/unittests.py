#!/usr/bin/env python

import unittest
import os
import numpy as np

from pympi import Elan
from zugubul.rvad_to_elan import read_rvad_segs, label_speech_segments
from zugubul.elan_tools import merge, trim
from zugubul.utils import batch_funct
from rVAD.rVAD_fast import rVAD_fast


class TestRvadToElan(unittest.TestCase):

    def test_read_rvad_segs(self):
        self.assertEqual(
            read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_tira1_matlab_segs.vad'),
            [(18, 114), (123,205)]
        )
        self.assertEqual(
            read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_tira1_matlab_frames.vad', dialect='frame'),
            [(18, 114), (123,205)]
        )

    def test_read_rvad_dialects(self):
        self.assertEqual(
            read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_segs.vad'),
            read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_frames.vad', dialect='frame')
        )

    def test_links_media_file(self):
        wav_fp = r'C:\projects\zugubul\tests\wavs\test_dendi1.wav'
        vad_fp = r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_segs.vad'

        eaf = label_speech_segments(wav_fp, vad_fp)

        self.assertEqual(eaf.media_descriptors[0]['MEDIA_URL'], wav_fp)
    
    def test_wav_to_elan(self):        
        wav_fp = r'C:\projects\zugubul\tests\wavs\test_tira1.wav'
        
        try:
            eaf = label_speech_segments(wav_fp)
        except:
            self.fail('Could not run make Eaf object from wav file.')

        annotations = eaf.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            annotations,
            [(100, 1150, ''), (1170, 2150, '')]
        )

    def test_wav_to_elan_script(self):
        wav_fp = r'C:\projects\zugubul\tests\wavs\test_tira1.wav'
        vad_fp = r'C:\projects\zugubul\tests\vads\test_tira1_frames.vad'
        eaf_in_fp = r'C:\projects\zugubul\tests\eafs\test_tira1_nonempty.eaf'
        eaf_out_fp = r'C:\projects\zugubul\tests\eafs\test_tira1_out.eaf'

        if os.path.exists(eaf_out_fp):
            os.remove(eaf_out_fp)

        if os.path.exists(vad_fp):
            os.remove(vad_fp)
        
        try:
            os.system(f'zugubul vad {wav_fp} {eaf_out_fp} -s {eaf_in_fp}')
        except:
            self.fail('Could not run command zugubul vad')
        try:
            out_annotations = Elan.Eaf(eaf_out_fp).get_annotation_data_for_tier('default-lt')
        except:
            self.fail('Could not open .eaf file')
        self.assertEqual(
            sorted(out_annotations, key=lambda l: l[0]),
            [(100, 1150, ''), (1170, 2150, 'jicelo')]
        )

    def test_rvad_to_elan_script(self):
        wav_fp = r'C:\projects\zugubul\tests\wavs\test_dendi1.wav'
        vad_fp = r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_segs.vad'
        eaf_fp = r'C:\projects\zugubul\tests\eafs\test_dendi1_matlab.eaf'

        if os.path.exists(eaf_fp):
            os.remove(eaf_fp)
        try:
            os.system(f'zugubul vad {wav_fp} {eaf_fp} -v {vad_fp} ')
        except:
            self.fail('Could not run script zugubul vad')
        try:
            Elan.Eaf(eaf_fp)
        except:
            self.fail('Could not open .eaf file output.')


class TestElanTools(unittest.TestCase):
    def test_trim(self):
        # make dummy .eaf file
        eaf = Elan.Eaf()
        eaf.add_tier('default-lt')

        for i in range(10):
            eaf.add_annotation(id_tier='default-lt', start=i, end=i+1, value='')
        eaf.add_annotation(id_tier='default-lt', start=10, end=11, value='include')

        # trim
        eaf = trim(eaf)

        # read annotations
        annotations = eaf.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            annotations,
            [(10, 11, 'include')]
        )

    def test_trim_stopword(self):
        # make dummy .eaf file
        eaf = Elan.Eaf()
        eaf.add_tier('default-lt')

        for i in range(10):
            eaf.add_annotation(id_tier='default-lt', start=i, end=i+1, value='stopword')
        eaf.add_annotation(id_tier='default-lt', start=10, end=11, value='include')

        # trim
        eaf = trim(eaf, 'default-lt', 'stopword')

        # read annotations
        annotations = eaf.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            annotations,
            [(10, 11, 'include')]
        )

    def test_merge(self):
        non_empty_eaf = Elan.Eaf()
        non_empty_eaf.add_tier('default-lt')
        non_empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='jicelo')

        empty_eaf = Elan.Eaf()
        empty_eaf.add_tier('default-lt')
        empty_eaf.add_annotation(id_tier='default-lt', start=100, end=1150, value='')
        empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='')

        out_eaf = merge(non_empty_eaf, empty_eaf)

        non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            non_empty_annotations,
            [(1170, 2150, 'jicelo')]
        )
        empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            sorted(empty_annotations, key=lambda l: l[0]),
            [(100, 1150, ''), (1170, 2150, '')]
        )
        out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            sorted(out_annotations, key=lambda l: l[0]),
            [(100, 1150, ''), (1170, 2150, 'jicelo')]
        )

class TestBatchScripts(unittest.TestCase):
    def test_batch_rvad(self):
        wav_dir = r'C:\projects\zugubul\tests\wavs'
        vad_dir = r'C:\projects\zugubul\tests\vads'

        def save_funct(fp, out) -> str:
            fp = str(fp)
            fp = fp.replace('.wav', '_batch.vad')
            fp = fp.replace(wav_dir, vad_dir)
            np.savetxt(fp, out)
            return fp
        
        out = batch_funct(rVAD_fast, wav_dir, '.wav', 'finwav', save_f=save_funct)

        for _, vad in out.items():
            gold_vad = vad.replace('_batch.vad', '_gold.vad')
            vad_array = np.loadtxt(vad)
            gold_vad_array = np.loadtxt(gold_vad)
            self.assertTrue(np.array_equal(vad_array, gold_vad_array))
            os.remove(vad)

    def test_batch_wav_to_elan(self):
        wav_dir = r'C:\projects\zugubul\tests\wavs'
        eaf_dir = r'C:\projects\zugubul\tests\eafs'

        def save_funct(fp, out) -> str:
            fp = str(fp)
            fp = fp.replace('.wav', '_batch.eaf')
            fp = fp.replace(wav_dir, eaf_dir)
            out.to_file(fp)
            return fp

        out = label_speech_segments(wav_dir, save_funct=save_funct)

        for _, eaf in out.items():
            gold_eaf = eaf.replace('_batch.eaf', '_gold.eaf')
            eaf_annotations = Elan.Eaf(eaf).get_annotation_data_for_tier('default-lt')
            gold_eaf_annotations = Elan.Eaf(gold_eaf).get_annotation_data_for_tier('default-lt')
            self.assertEqual(sorted(eaf_annotations), sorted(gold_eaf_annotations))
            os.remove(eaf)

if __name__ == '__main__':
    unittest.main()