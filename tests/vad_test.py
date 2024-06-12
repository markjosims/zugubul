from zugubul.vad.silero_vad import run_silero_vad
from zugubul.vad.pyannote import run_pyannote_vad
from zugubul.vad.rVAD import run_rVAD_fast

wav_path = 'tests/wavs/test_dendi1.wav'

def test_silero_vad():
    out = run_silero_vad(wav_fp=wav_path)
    for segment in out:
        assert type(segment['start']) is int
        assert type(segment['end']) is int

def test_pyannote_vad():
    out = run_pyannote_vad(wav_fp=wav_path)
    for segment in out:
        assert type(segment['start']) is int
        assert type(segment['end']) is int

def test_rvad():
    out = run_rVAD_fast(wav_path)
    for segment in out:
        assert type(segment['start']) is int
        assert type(segment['end']) is int