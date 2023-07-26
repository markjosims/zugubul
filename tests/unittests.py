#!/usr/bin/env python

import unittest
import os

from pympi import Elan
from zugubul.rvad_to_elan import read_rvad_segs


class TestRvadToElan(unittest.TestCase):

    def test_read_rvad_segs(self):
        self.assertEqual(
            read_rvad_segs('tests/test_tira1_matlab_segs.vad'),
            [(18, 114), (123,205)]
        )
        self.assertEqual(
            read_rvad_segs('tests/test_tira1_matlab_frames.vad', dialect='frame'),
            [(18, 114), (123,205)]
        )

    def test_read_rvad_dialects(self):
        self.assertEqual(
            read_rvad_segs('tests/test_dendi1_matlab_segs.vad'),
            read_rvad_segs('tests/test_dendi1_matlab_frames.vad', dialect='frame')
        )

    def test_links_media_file(self):
        wav_fp = r'C:\projects\zugubul\tests\test_dendi1_matlab.wav'
        vad_fp = r'C:\projects\zugubul\tests\test_dendi1_matlab_segs.vad'
        eaf_fp = r'C:\projects\zugubul\tests\test_dendi1_matlab.eaf'

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

        if os.path.exists(vad_fp):
            os.remove(vad_fp)
        
        try:
            os.system(f'wav_to_elan {wav_fp} {vad_fp} {eaf_fp}')
        except:
            self.fail('Could not run script wav_to_elan.py')
        try:
            Elan.Eaf(eaf_fp)
        except:
            self.fail('Could not open .eaf file')

class TestElanTools(unittest.TestCase):
    
    def test_usage_str(self):
        help_str = os.popen('elan_tools -h').read()
        self.assertIn('Usage', help_str, 'Usage not found in help string.')

    def test_trim(self):
        in_fp = r'C:\projects\zugubul\tests\test_trim_in.eaf'
        out_fp = r'C:\projects\zugubul\tests\test_trim_out.eaf'

        if os.path.exists(in_fp):
            os.remove(in_fp)

        # make dummy .eaf file
        eaf_obj = Elan.Eaf()
        eaf_obj.add_tier('default-lt')

        for i in range(10):
            eaf_obj.add_annotation(id_tier='default-lt', start=i, end=i+1, value='')
        eaf_obj.add_annotation(id_tier='default-lt', start=10, end=11, value='include')

        eaf_obj.to_file(in_fp)

        # trim
        os.system(f'elan_tools trim {in_fp} {out_fp}')

        # read annotations
        eaf_obj = Elan.Eaf(out_fp)
        annotations = eaf_obj.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            annotations,
            [(10, 11, 'include')]
        )

    def test_trim_stopword(self):
        in_fp = r'C:\projects\zugubul\tests\test_trim_stopword_in.eaf'
        out_fp = r'C:\projects\zugubul\tests\test_trim_stopword_out.eaf'

        if os.path.exists(in_fp):
            os.remove(in_fp)

        # make dummy .eaf file
        eaf_obj = Elan.Eaf()
        eaf_obj.add_tier('default-lt')

        for i in range(10):
            eaf_obj.add_annotation(id_tier='default-lt', start=i, end=i+1, value='stopword')
        eaf_obj.add_annotation(id_tier='default-lt', start=10, end=11, value='include')

        eaf_obj.to_file(in_fp)

        # trim
        os.system(f'elan_tools trim {in_fp} {out_fp} default-lt stopword')

        # read annotations
        eaf_obj = Elan.Eaf(out_fp)
        annotations = eaf_obj.get_annotation_data_for_tier('default-lt')
        self.assertEqual(
            annotations,
            [(10, 11, 'include')]
        )

if __name__ == '__main__':
    unittest.main()