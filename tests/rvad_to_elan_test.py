from pympi import Elan
from zugubul.rvad_to_elan import read_rvad_segs, label_speech_segments
from zugubul.utils import batch_funct
from rVAD.rVAD_fast import rVAD_fast
import os

def test_read_rvad_segs():
    assert\
        read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_tira1_matlab_segs.vad') ==\
        [(18, 114), (123,205)]
    
def test_read_rvad_franes():
    assert\
        read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_tira1_matlab_frames.vad', dialect='frame') ==\
        [(18, 114), (123,205)]

def test_read_rvad_dialects():
    assert \
        read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_segs.vad') ==\
        read_rvad_segs(r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_frames.vad', dialect='frame')

def test_links_media_file():
    wav_fp = r'C:\projects\zugubul\tests\wavs\test_dendi1.wav'
    vad_fp = r'C:\projects\zugubul\tests\vads\test_dendi1_matlab_segs.vad'
    eaf = label_speech_segments(wav_fp, vad_fp)

    assert eaf.media_descriptors[0]['MEDIA_URL'] == wav_fp

def test_wav_to_elan():        
    wav_fp = r'C:\projects\zugubul\tests\wavs\test_tira1.wav'
    eaf = label_speech_segments(wav_fp)
    annotations = eaf.get_annotation_data_for_tier('default-lt')
    assert annotations == [(100, 1150, ''), (1170, 2150, '')]

def test_wav_to_elan_template():
    etf_fp = r'C:\projects\zugubul\tests\eafs\test_tira_template.etf'
    etf = Elan.Eaf(etf_fp)
    etf_tiers = list(sorted(etf.get_tier_names()))

    wav_fp = r'C:\projects\zugubul\tests\wavs\test_tira1.wav'
    rvad_fp = r'C:\projects\zugubul\tests\vads\test_tira1_gold.vad'
    eaf = label_speech_segments(wav_fp, rvad_fp=rvad_fp, tier='IPA Transcription', etf=etf)
    eaf_tiers = list(sorted(eaf.get_tier_names()))

    out_fp = r'C:\projects\zugubul\tests\eafs\test_tira_template_out.eaf'
    if os.path.exists(out_fp):
        os.remove(out_fp)
    eaf.to_file(out_fp)

    assert etf_tiers == eaf_tiers