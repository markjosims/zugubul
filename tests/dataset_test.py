from zugubul.models.dataset import RawDatum, Dataset

def test_raw_datum1():
    datum = RawDatum('potato', 'potato.wav')
    assert datum.text == 'potato'
    assert datum.audio_fp == 'potato.wav'
    assert datum.start == 0
    assert datum.end == -1

def test_raw_datum2():
    datum = RawDatum('potato', 'potato.wav', 10, 30)
    assert datum.text == 'potato'
    assert datum.audio_fp == 'potato.wav'
    assert datum.start == 10
    assert datum.end == 30