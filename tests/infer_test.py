from zugubul.models.infer import infer_lid

from pympi import Elan

def test_infer_lid():
    wav_fp = r'C:\projects\zugubul\tests\wavs\test\test.wav'
    model_path = 'markjosims/wav2vec2-large-mms-1b-tira-lid'

    eaf = infer_lid(wav_fp, model_path)

    annotations = eaf.get_annotation_data_for_tier('default-lt')

    for a in annotations:
        assert a[2] in ('ENG', 'TIC')