import numpy as np
from pympi import Elan
from zugubul.elan_tools import trim, merge, metadata

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

def test_merge_keep_both():
    non_empty_eaf, empty_eaf = dummy_eafs()


    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, overlap_behavior='keep_both')

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, ''), (1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

def test_merge_keep_matrix():
    non_empty_eaf, empty_eaf = dummy_eafs()

    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, overlap_behavior='keep_matrix')

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, ''), (2200, 2250, 'ngamhare')]

def test_merge_keep_source():
    non_empty_eaf, empty_eaf = dummy_eafs()

    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, overlap_behavior='keep_source')

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

def test_metadata():
    non_empty_eaf, empty_eaf = dummy_eafs()
    non_empty_eaf.add_linked_file('foo.wav')
    empty_eaf.add_linked_file('bar.wav')

    metadata_df = metadata('foo.eaf', non_empty_eaf)

    assert np.array_equal(metadata_df['start'], [1170, 2200])
    assert np.array_equal(metadata_df['end'], [2150, 2250])
    assert np.array_equal(metadata_df['tier'], ['default-lt', 'default-lt'])
    assert np.array_equal(metadata_df['text'], ['jicelo', 'ngamhare'])
    assert np.array_equal(metadata_df['file_name'], ['foo.wav', 'foo.wav'])

    metadata_df = metadata('bar.eaf', empty_eaf)

    assert np.array_equal(metadata_df['start'], [100, 1170])
    assert np.array_equal(metadata_df['end'], [1150, 2150])
    assert np.array_equal(metadata_df['tier'], ['default-lt', 'default-lt'])
    assert np.array_equal(metadata_df['text'], ['', ''])
    assert np.array_equal(metadata_df['file_name'], ['bar.wav', 'bar.wav'])