from pympi import Elan
from zugubul.elan_tools import trim, merge

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

def test_merge():
    non_empty_eaf = Elan.Eaf()
    non_empty_eaf.add_tier('default-lt')
    non_empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='jicelo')

    empty_eaf = Elan.Eaf()
    empty_eaf.add_tier('default-lt')
    empty_eaf.add_annotation(id_tier='default-lt', start=100, end=1150, value='')
    empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='')

    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf)

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, ''), (1170, 2150, 'jicelo')]

def test_merge_exclude_overlap():
    non_empty_eaf = Elan.Eaf()
    non_empty_eaf.add_tier('default-lt')
    non_empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='jicelo')
    non_empty_eaf.add_annotation(id_tier='default-lt', start=2200, end=2250, value='ngamhare')

    empty_eaf = Elan.Eaf()
    empty_eaf.add_tier('default-lt')
    empty_eaf.add_annotation(id_tier='default-lt', start=100, end=1150, value='')
    empty_eaf.add_annotation(id_tier='default-lt', start=1170, end=2150, value='')

    out_eaf = merge(eaf_source=non_empty_eaf, eaf_matrix=empty_eaf, exclude_overlap=True)

    non_empty_annotations = non_empty_eaf.get_annotation_data_for_tier('default-lt')
    assert non_empty_annotations == [(1170, 2150, 'jicelo'), (2200, 2250, 'ngamhare')]

    empty_annotations = empty_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(empty_annotations) == [(100, 1150, ''), (1170, 2150, '')]

    out_annotations = out_eaf.get_annotation_data_for_tier('default-lt')
    assert sorted(out_annotations) == [(100, 1150, ''), (1170, 2150, ''), (2200, 2250, 'ngamhare')]