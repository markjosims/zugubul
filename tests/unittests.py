import unittest

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

if __name__ == '__main__':
    unittest.main()