#!/usr/bin/env python

import unittest
import os

from pympi import Elan
from zugubul.src.rvad_to_elan import read_rvad_segs


class TestRvadToElan(unittest.TestCase):

    def test_read_rvad_segs(self):
        self.assertEqual(
            read_rvad_segs('zugubul/tests/test_tira1_segs.vad'),
            [(18, 114), (123,205)]
        )
        self.assertEqual(
            read_rvad_segs('zugubul/tests/test_tira1_frames.vad', dialect='frame'),
            [(18, 114), (123,205)]
        )

    def test_read_rvad_dialects(self):
        self.assertEqual(
            read_rvad_segs('zugubul/tests/test_dendi1_segs.vad'),
            read_rvad_segs('zugubul/tests/test_dendi1_frames.vad', dialect='frame')
        )

    def test_links_media_file(self):
        vad_fp = r'C:\projects\zugubul\tests\test_dendi1_segs.vad'
        wav_fp = r'C:\projects\zugubul\tests\test_dendi1.wav'
        eaf_fp = r'C:\projects\zugubul\tests\test_dendi1.eaf'

        os.system(f'zugubul/src/rvad_to_elan.py {vad_fp} {wav_fp} {eaf_fp}')
        eaf = Elan.Eaf(eaf_fp)

        self.assertEqual(eaf.media_descriptors[0]['MEDIA_URL'], wav_fp)


if __name__ == '__main__':
    unittest.main()