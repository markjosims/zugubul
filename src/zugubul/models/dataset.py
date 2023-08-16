from typing import Union, Optional, Tuple
import os
import pandas as pd
from datasets import Dataset, load_dataset
from zugubul.elan_tools import snip_audio
from math import ceil
from pathlib import Path
import shutil

def init_lid_dataset(dirpath: Union[str, os.PathLike]) -> Dataset:
    """
    dirpath indicates dataset directory
    directory must be of following structure:
    dirpath
        metadata.csv
        train
            recording1.wav
            recording2.wav
            ...
        val
            recording3.wav
            recording4.wav
            ...
        test
            recording5.wav
            recording6.wav
            ...
    """
    return load_dataset('audiofolder', data_dir=dirpath)

def split_data(
        eaf_data: Union[str, os.PathLike, pd.DataFrame],
        out_dir: Union[str, os.PathLike],
        lid: bool = False,
        copy: bool = False,
        splitsize: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> str:
    """
    eaf_data is the path to a csv produced by the eaf_data function in Elan tools,
    or an equivalent pandas DataFrame object.
    out_dir is the path to a data folder to save the new metadata.csv and all clipped recordings to.
    lid is a bool indicating whether the split is being made for a language identification model or not (default False).
    copy is a bool indicating whether the .wav clips should be copied to the new directories or moved
    (default False, i.e. clips will be moved).
    splitsize is a tuple of len 3 containing floats the length of each split (training, validation, test), default (0.8, 0.1, 0.1).
    Returns path to the new metadata.csv.
    """
    if type(eaf_data) is str:
        eaf_data = pd.read_csv(eaf_data)
    if type(out_dir) is str:
        out_dir = Path(out_dir)

    eaf_data: pd.DataFrame

    if 'wav_clip' not in eaf_data.columns:
        eaf_data = snip_audio(eaf_data, out_dir)

    # drop unlabeled rows
    eaf_data = eaf_data[eaf_data['text'] != '']

    eaf_data=eaf_data.sample(frac=1)

    train_size = ceil(splitsize[0]*len(eaf_data))
    train_split = eaf_data[:train_size]

    val_size = ceil(splitsize[1]*len(eaf_data))
    val_split = eaf_data[train_size:train_size+val_size]

    test_split = eaf_data[train_size+val_size:]

    split_dict = {
        'train': train_split,
        'val': val_split,
        'test': test_split,
    }

    def move_clip_to_split(clip_fp: Path, split_dir: Path) -> str:
        out_path = split_dir / clip_fp.name
        if copy:
            shutil.copy(clip_fp, out_path)
            return out_path
        shutil.move(clip_fp, out_path)
        return str(out_path)

        

    for split, split_df in split_dict.items():
        split_dir = out_dir / split
        os.mkdir(split_dir)
        split_clips = split_df['wav_clip'].apply(
            lambda fp: move_clip_to_split(Path(fp), split_dir)
        )

        eaf_data.loc[split_df.index, 'file_name'] = split_clips
    
    eaf_data = eaf_data[['file_name', 'text']]
    out_path = out_dir/'metadata.csv'
    eaf_data.to_csv(out_path, index=False)

    return out_path
