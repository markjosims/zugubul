from zugubul.models.dataset import split_data, make_lid_labels, balance_lid_data, process_annotation_length, make_lm_dataset
from datasets import load_dataset
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

        shutil.copy('tests/wavs/train/train.wav', os.path.join(train_dir, 'train.wav'))
        shutil.copy('tests/wavs/test/test.wav', os.path.join(test_dir, 'test.wav'))
        shutil.copy('tests/wavs/val/val.wav', os.path.join(val_dir, 'val.wav'))
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

        dendi_path = 'tests/wavs/test_dendi1.wav'
        tira_path = 'tests/wavs/test_tira1.wav'

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

        dendi_path = 'tests/wavs/test_dendi1.wav'
        tira_path = 'tests/wavs/test_tira1.wav'

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
                    'target_labels': False,
                    'meta_labels': ['DDN', 'DDN1'],
                    'empty': 'meta',
                    'balance': False,
                    'process_length': False
                }
            }
            tomli_w.dump(toml_data, f)

        yield dir_path, csv_path, toml_path

@pytest.fixture
def lid_df():
    lid_df = pd.DataFrame(data={
        'lang': ['ENG']*5+['TIC']*2+['DDN']*10,
        'wav_source': ['file1.wav']*6+['file2.wav']*11,
        'start':     [0,   400, 700, 1000, 10000] + [12050, 20000] + [100,  3999, 4200, 100000,  100950, 150000, 170000, 20950,  30000, 40000],
        'end':       [300, 600, 900, 1200, 12000] + [15000, 20900] + [2000, 4050, 4300, 100900,  100999, 155000, 190000, 21900,  30950, 41000],
        #            ENG                            TIC              DDN
        #            merge---------------  keep      keep   delete   merge------------  merge+delete---  keep    keep    delete  delete keep
        #new_start:  [0,                   10000] + [12050       ] + [100,                               150000, 170000,                40000],
        #new_end:    [               1200, 12000] + [15000       ] + [            4300,                  155000, 190000,                41000],
    })
    
    return lid_df

def test_load_dataset(tmp_dataset):
    dataset = load_dataset(tmp_dataset)
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
            os.path.join(s, clip)
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
    target_labels = None
    meta_labels = ['DDN', 'DDN1']

    df = make_lid_labels(
        csv_path,
        targetlang=targetlang,
        metalang=metalang,
        target_labels=target_labels,
        meta_labels=meta_labels,
        balance=False,
        process_length=False
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

def test_make_lm_data(tmp_unsplit_data):
    _, csv_path = tmp_unsplit_data
    text =['apri. jicelo. ngamhare. n na suba. a ci deesu. n na gbei.']
    dataset = make_lm_dataset(csv_path, make_splits=False)
    assert list(dataset['text']) == text

def test_balance_lid_data(lid_df):
    out = balance_lid_data(lid_df)

    assert out['lang'].value_counts()['ENG'] == 2
    assert out['lang'].value_counts()['TIC'] == 2
    assert out['lang'].value_counts()['DDN'] == 2

def test_process_annotation_length(lid_df):
    out = process_annotation_length(lid_df, min_gap=2000, lid=True)

    new_start = [0, 10000] + [12050] + [100, 150000, 170000, 40000]
    new_end = [1200, 12000] + [15000] + [4300, 155000, 190000, 41000]

    assert np.array_equal(out['start'], new_start)
    assert np.array_equal(out['end'], new_end)