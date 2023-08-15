from zugubul.utils import eaf_to_file_safe, batch_funct
from zugubul.elan_tools import metadata
from pympi import Elan
import os
import pytest
import tempfile
import numpy as np
import pandas as pd

@pytest.fixture
def tmp_eafdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, 'eafs')
        os.mkdir(dir_path)

        annotations = {
            'eaf1': [
                (0, 200, 'foo'),
                (300, 400, 'bar'),
                (600, 1000, 'baz potato foo'),
            ],
            'eaf2': [
                (100, 900, 'foo'),
                (1100, 1300, 'ficelo'),
                (1500, 2000, 'fevelezo'),
                (2200, 2400, 'bar bico'),
            ],
            'eaf3': [
                (50, 200, 'n na suba'),
                (250, 300, 'n gono baanii'),
                (400, 600, 'a gono baani'),
                (700, 900, 'a ci deesu'),
            ]
        }
        linked_files = {
            'eaf1': 'foo.wav',
            'eaf2': 'bar.wav',
            'eaf3': 'baz.wav'
        }
        for k, v in annotations.items():
            eaf = Elan.Eaf()
            eaf.add_linked_file(linked_files[k])
            eaf.add_tier('default-lt')
            for a in v:
                eaf.add_annotation('default-lt', a[0], a[1], a[2])
            eaf.to_file(os.path.join(dir_path, k+'.eaf'))

        yield dir_path, annotations, linked_files

def test_eaf_to_file_safe():
    eaf_fp = r'C:\projects\zugubul\tests\eafs\test_tira1_gold.eaf'
    bak_fp = r'C:\projects\zugubul\tests\eafs\test_tira1_gold.bak'
    eaf = Elan.Eaf(eaf_fp)
    bak = Elan.Eaf(bak_fp)

    eaf_to_file_safe(eaf, eaf_fp)

    assert os.path.isfile(eaf_fp)   
    assert os.path.isfile(bak_fp)

def test_metadata_batch(tmp_eafdir):
    dir_path, annotations, linked_files = tmp_eafdir

    df_dict = batch_funct(
        metadata,
        dir=dir_path,
        suffix='.eaf',
        file_arg='eaf',
        overwrite=True
    )

    df = pd.concat(df_dict.values())

    print(df['eaf_name'].iloc[0])

    for eaf, eaf_annotations in annotations.items():
        eaf_path = os.path.join(dir_path, eaf+'.eaf')
        assert df[df['eaf_name'] == eaf_path].shape[1] > 0
        eaf_df = df[df['eaf_name']==eaf_path]

        eaf_starts = sorted(a[0] for a in eaf_annotations)
        eaf_ends = sorted(a[1] for a in eaf_annotations)
        eaf_values = sorted(a[2] for a in eaf_annotations)

        assert np.array_equal(eaf_starts, eaf_df['start'].sort_values())
        assert np.array_equal(eaf_ends, eaf_df['end'].sort_values())
        assert np.array_equal(eaf_values, eaf_df['text'].sort_values())
        assert np.array_equal(
            df['eaf_name']==eaf_path,
            df['wav_source']==linked_files[eaf]
        )