#!/usr/bin/env python

import unittest
import os

from pympi import Elan
from zugubul.rvad_to_elan import read_rvad_segs


class TestRvadToElan(unittest.TestCase):

    def test_read_rvad_segs(self):
        self.assertEqual(
            read_rvad_segs('tests/test_tira1_segs.vad'),
            [(18, 114), (123,205)]
        )
        self.assertEqual(
            read_rvad_segs('tests/test_tira1_frames.vad', dialect='frame'),
            [(18, 114), (123,205)]
        )

    def test_read_rvad_dialects(self):
        self.assertEqual(
            read_rvad_segs('tests/test_dendi1_segs.vad'),
            read_rvad_segs('tests/test_dendi1_frames.vad', dialect='frame')
        )

    def test_links_media_file(self):
        wav_fp = r'C:\projects\zugubul\tests\test_dendi1.wav'
        vad_fp = r'C:\projects\zugubul\tests\test_dendi1_segs.vad'
        eaf_fp = r'C:\projects\zugubul\tests\test_dendi1.eaf'

        if os.path.exists(eaf_fp):
            os.remove(eaf_fp)
        try:
            os.system(f'rvad_to_elan {wav_fp} {vad_fp} {eaf_fp}')
        except:
            self.fail('Could not run script rvad_to_elan.py')
        eaf = Elan.Eaf(eaf_fp)

        self.assertEqual(eaf.media_descriptors[0]['MEDIA_URL'], wav_fp)
    
    def test_wav_to_elan(self):        
        wav_fp = r'C:\projects\zugubul\tests\test_tira1.wav'
        vad_fp = r'C:\projects\zugubul\tests\test_tira1_frames.vad'
        eaf_fp = r'C:\projects\zugubul\tests\test_tira1.eaf'

        if os.path.exists(eaf_fp):
            os.remove(eaf_fp)
        
        try:
            os.system(f'wav_to_elan {wav_fp} {vad_fp} {eaf_fp}')
        except:
            self.fail('Could not run script wav_to_elan.py')
        try:
            Elan.Eaf(eaf_fp)
        except:
            self.fail('Could not open .eaf file')

if __name__ == '__main__':
    unittest.main()