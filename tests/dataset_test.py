from zugubul.models.dataset import init_lid_dataset
import pytest
import csv
import os
import shutil
import tempfile

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

def test_init_lid_dataset(tmp_dataset):
    dataset = init_lid_dataset(tmp_dataset)
    assert dataset['train'][0]['text'] == 'train'