from zugubul.models.dataset import init_lid_dataset, split_data, make_lid_labels
import pytest
import csv
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import tomli_w

@pytest.fixture
def tmp_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, 'dataset')
        data_dir = os.path.join(dir_path, 'data')
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        val_dir = os.path.join(data_dir, 'val')

        os.mkdir(dir_path)
        os.mkdir(data_dir)
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        os.mkdir(val_dir)

        shutil.copy(r'C:\projects\zugubul\tests\wavs\train\train.wav', os.path.join(train_dir, 'train.wav'))
        shutil.copy(r'C:\projects\zugubul\tests\wavs\test\test.wav', os.path.join(test_dir, 'test.wav'))
        shutil.copy(r'C:\projects\zugubul\tests\wavs\val\val.wav', os.path.join(val_dir, 'val.wav'))
        csv_path = os.path.join(dir_path, 'metadata.csv')
        with open(csv_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            rows = [
                ['file_name', 'text'],
                ['data/train/train.wav', 'train'],
                ['data/test/test.wav', 'test'],
                ['data/val/val.wav', 'val'],
            ]
            writer.writerows(rows)
        yield dir_path

@pytest.fixture
def tmp_unsplit_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, 'dataset1')
        os.mkdir(dir_path)

        dendi_path = r'C:\projects\zugubul\tests\wavs\test_dendi1.wav'
        tira_path = r'C:\projects\zugubul\tests\wavs\test_tira1.wav'

        csv_path = os.path.join(dir_path, 'eaf_data.csv')

        with open(csv_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            rows = [
                ['eaf_name', 'wav_source', 'text', 'start', 'end'],
                ['tira.eaf', tira_path, 'apri', 100, 500],
                ['tira.eaf', tira_path, 'jicelo', 600, 1000],
                ['tira.eaf', tira_path, 'ngamhare', 1100, 1500],
                ['dendi.eaf', dendi_path, 'n na suba', 100, 500],
                ['dendi.eaf', dendi_path, 'a ci deesu', 600, 1000],
                ['dendi.eaf', dendi_path, 'n na gbei', 1100, 1500],
            ]
            writer.writerows(rows)

        yield dir_path, csv_path

@pytest.fixture
def tmp_lid_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, 'dataset1')
        os.mkdir(dir_path)

        dendi_path = r'C:\projects\zugubul\tests\wavs\test_dendi1.wav'
        tira_path = r'C:\projects\zugubul\tests\wavs\test_tira1.wav'

        csv_path = os.path.join(dir_path, 'eaf_data.csv')

        with open(csv_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            rows = [
                ['eaf_name', 'wav_source', 'text', 'start', 'end'],
                ['tira.eaf', tira_path, 'TIC', 100, 500],
                ['tira.eaf', tira_path, 'TIC', 600, 1000],
                ['tira.eaf', tira_path, 'ngamhare', 1100, 1500],
                ['dendi.eaf', dendi_path, '', 100, 500],
                ['dendi.eaf', dendi_path, 'DDN1', 600, 1000],
                ['dendi.eaf', dendi_path, 'DDN', 1100, 1500],
            ]
            writer.writerows(rows)

        toml_path = os.path.join(dir_path, 'tira.toml')

        with open(toml_path, 'wb') as f:
            toml_data = {
                'LID': {
                    'targetlang': 'TIC',
                    'metalang': 'DDN',
                    'target_labels': '*',
                    'meta_labels': ['DDN', 'DDN1'],
                    'empty': 'meta'
                }
            }
            tomli_w.dump(toml_data, f)

        yield dir_path, csv_path, toml_path

def test_init_lid_dataset(tmp_dataset):
    dataset = init_lid_dataset(tmp_dataset)
    assert dataset['train'][0]['text'] == 'train'

def test_split_data(tmp_unsplit_data):
    dir_path, csv_path = tmp_unsplit_data
    csv_path = split_data(csv_path, dir_path)
    df = pd.read_csv(csv_path)

    assert np.array_equal(df.columns, ['file_name', 'text'])

    clip_files = {
        'test_tira1[100-500].wav': 'apri',
        'test_tira1[600-1000].wav': 'jicelo',
        'test_tira1[1100-1500].wav': 'ngamhare',
        'test_dendi1[100-500].wav': 'n na suba',
        'test_dendi1[600-1000].wav': 'a ci deesu',
        'test_dendi1[1100-1500].wav': 'n na gbei',
    }

    splits = ['train', 'val', 'test']

    for clip, text in clip_files.copy().items():
        possible_filenames = [
            os.path.join(dir_path, s, clip)
            for s in splits
        ]
        for f in possible_filenames:
            has_f = df[df['file_name']==f]
            if len(has_f) > 0:
                assert has_f['text'].iloc[0] == text
                clip_files.pop(clip)
    
    assert not clip_files

def test_make_lid_labels(tmp_lid_data):
    _, csv_path, _ = tmp_lid_data
    
    targetlang = 'TIC'
    metalang = 'DDN'
    target_labels = '*'
    meta_labels = ['DDN', 'DDN1']

    df = make_lid_labels(
        csv_path,
        targetlang=targetlang,
        metalang=metalang,
        target_labels=target_labels,
        meta_labels=meta_labels
    )

    assert np.array_equal(
        df['eaf_name'] == 'tira.eaf',
        df['lang'] == 'TIC'
    )

    assert np.array_equal(
        df['eaf_name'] == 'dendi.eaf',
        df['lang'] == 'DDN'
    )

    assert len(df[df['lang']=='TIC']) == 3
    assert len(df[df['lang']=='DDN']) == 2

def test_make_lid_labels1(tmp_lid_data):
    _, csv_path, toml_path = tmp_lid_data
    
    df = make_lid_labels(
        csv_path,
        toml=toml_path
    )

    assert np.array_equal(
        df['eaf_name'] == 'tira.eaf',
        df['lang'] == 'TIC'
    )

    assert np.array_equal(
        df['eaf_name'] == 'dendi.eaf',
        df['lang'] == 'DDN'
    )

    assert len(df[df['lang']=='TIC']) == 3
    assert len(df[df['lang']=='DDN']) == 3