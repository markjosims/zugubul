import numpy as np
import os
import pytest
import pandas as pd
from pympi import Elan
from pydub import AudioSegment
from zugubul.elan_tools import trim, merge, metadata, snip_audio

def test_trim():
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
    assert annotations == [(10, 11, 'include')]


def test_trim_stopword():
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
    assert annotations == [(10, 11, 'include')]

@pytest.fixture
def dummy_eafs():
    non_empty_eaf = Elan.Eaf()
    non_empty_eaf.add_tier('default-lt')
    non_empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='jicelo')
    non_empty_eaf.add_annotation(id_tier='default-lt', start=2200, end=2250, value='ngamhare')

    empty_eaf = Elan.Eaf()
    empty_eaf.add_tier('default-lt')
    empty_eaf.add_annotation(id_tier='default-lt', start=100, end=1150, value='')
    empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='')
    return non_empty_eaf,empty_eaf

@pytest.fixture
def tira_eaf():
    eaf = Elan.Eaf()
    eaf.add_linked_file(r'C:\projects\zugubul\tests\wavs\test_tira1.wav')
    eaf.add_tier('default-lt')
    eaf.add_annotation(id_tier='default-lt', start=100, end=1150, value='apri')
    eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='jicelo')
    return eaf

@pytest.fixture
def dendi_eaf():
    eaf = Elan.Eaf()
    eaf.add_linked_file(r'C:\projects\zugubul\tests\wavs\test_dendi1.wav')
    eaf.add_tier('default-lt')
    eaf.add_annotation(id_tier='default-lt', start=2000, end=3000, value='foo')
    eaf.add_annotation(id_tier='default-lt', start=4000, end=5000, value='baz')
    eaf.add_annotation(id_tier='default-lt', start=6000, end=7000, value='bar')
    eaf.add_annotation(id_tier='default-lt', start=8000, end=9000, value='faz')
    return eaf


def test_merge_keep_both(dummy_eafs):
    non_empty_eaf, empty_eaf = dummy_eafs


    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, overlap_behavior='keep_both')

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, ''), (1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

def test_merge_keep_matrix(dummy_eafs):
    non_empty_eaf, empty_eaf = dummy_eafs

    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, overlap_behavior='keep_matrix')

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, ''), (2200, 2250, 'ngamhare')]

def test_merge_keep_source(dummy_eafs):
    non_empty_eaf, empty_eaf = dummy_eafs

    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, overlap_behavior='keep_source')

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

def test_metadata(dummy_eafs):
    non_empty_eaf, empty_eaf = dummy_eafs
    non_empty_eaf.add_linked_file('foo.wav')
    empty_eaf.add_linked_file('bar.wav')

    metadata_df = metadata('foo.eaf', non_empty_eaf)

    assert np.array_equal(metadata_df['start'], [1170, 2200])
    assert np.array_equal(metadata_df['end'], [2150, 2250])
    assert np.array_equal(metadata_df['tier'], ['default-lt', 'default-lt'])
    assert np.array_equal(metadata_df['text'], ['jicelo', 'ngamhare'])
    assert np.array_equal(metadata_df['wav_source'], ['foo.wav', 'foo.wav'])

    metadata_df = metadata('bar.eaf', empty_eaf)

    assert np.array_equal(metadata_df['start'], [100, 1170])
    assert np.array_equal(metadata_df['end'], [1150, 2150])
    assert np.array_equal(metadata_df['tier'], ['default-lt', 'default-lt'])
    assert np.array_equal(metadata_df['text'], ['', ''])
    assert np.array_equal(metadata_df['wav_source'], ['bar.wav', 'bar.wav'])

def test_snip_audio(tmp_path, tira_eaf: Elan.Eaf):
    clips_dir = os.path.join(tmp_path, 'clips')
    os.mkdir(clips_dir)

    clips_df = snip_audio(tira_eaf, clips_dir)

    assert len(clips_df) == 2

    for a in tira_eaf.get_annotation_data_for_tier('default-lt'):
        start = a[0]
        end = a[1]
        this_annotation = clips_df[clips_df['start']==start]
        audio_clip_path = this_annotation['wav_clip'].iloc[0]
        audio_clip = AudioSegment.from_wav(audio_clip_path)

        assert len(audio_clip) == end-start

def test_snip_audio1(tmp_path, tira_eaf: Elan.Eaf, dendi_eaf: Elan.Eaf):
    clips_dir = os.path.join(tmp_path, 'clips1')
    os.mkdir(clips_dir)

    tira_path = os.path.join(clips_dir, 'tira.eaf')
    dendi_path = os.path.join(clips_dir, 'dendi.eaf')
    tira_eaf.to_file(tira_path)
    dendi_eaf.to_file(dendi_path)

    tira_metadata = metadata(tira_path, tira_eaf)
    dendi_metadata = metadata(dendi_path, dendi_eaf)

    metadata_df = pd.concat([tira_metadata, dendi_metadata])

    clips_df = snip_audio(metadata_df, out_dir=clips_dir)

    for a in tira_eaf.get_annotation_data_for_tier('default-lt'):
        start = a[0]
        end = a[1]
        this_annotation = clips_df[clips_df['start']==start]
        audio_clip_path = this_annotation['wav_clip'].iloc[0]
        audio_clip = AudioSegment.from_wav(audio_clip_path)

        assert len(audio_clip) == end-start
    
    for a in dendi_eaf.get_annotation_data_for_tier('default-lt'):
        start = a[0]
        end = a[1]
        this_annotation = clips_df[clips_df['start']==start]
        audio_clip_path = this_annotation['wav_clip'].iloc[0]
        audio_clip = AudioSegment.from_wav(audio_clip_path)

        assert len(audio_clip) == end-start