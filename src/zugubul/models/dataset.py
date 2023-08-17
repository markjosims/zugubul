from typing import Union, Sequence, Tuple, Optional, Literal
import os
import pandas as pd
from datasets import Dataset, load_dataset
from zugubul.elan_tools import snip_audio, eaf_data
from math import ceil
from pathlib import Path
import shutil
import tomli

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
    if (type(eaf_data) is str) or isinstance(eaf_data, Path):
        eaf_data = pd.read_csv(eaf_data)
    if type(out_dir) is str:
        out_dir = Path(out_dir)

    eaf_data: pd.DataFrame

    label_col = 'text'
    if lid:
        label_col = 'lang'

    if 'wav_clip' not in eaf_data.columns:
        eaf_data = snip_audio(eaf_data, out_dir)

    # drop unlabeled rows
    eaf_data = eaf_data[eaf_data[label_col] != '']

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
    
    metadata = eaf_data[['file_name', label_col]]
    out_path = out_dir/'metadata.csv'
    metadata.to_csv(out_path, index=False)

    return out_path

def make_lid_labels(
        annotations: Union[str, pd.DataFrame],
        targetlang : Optional[str] = None,
        metalang: Optional[str] = None,
        target_labels: Union[Sequence[str], Literal['*'], None] = None,
        meta_labels: Union[Sequence[str], Literal['*'], None] = None,
        empty: Union[Literal['target'], Literal['meta'], Literal['exclude']] = 'exclude',
        toml: Optional[Union[str, os.PathLike]] = None
    ) -> pd.DataFrame:
    """
    annotations is a dataframe or str pointing to csv with a 'text' column.
    targetlang and metalang are strs indicating ISO codes or other unique identifiers
    for the target and metalanguage to be identified.
    target_labels is a list indicating all strs to be mapped to targetlang,
    meta_labels is a list indicating all strs to be mapped to metalang.
    Either target_labels or meta_labels may be the str '*',
    in which case all non-empty strs not specified by the other arg will be mapped to that language.
    If toml is not None, ignore other kwargs and read their values from the provided .toml file. 
    Creates a new column 'lang' containing the label for each language.
    Returns dataframe.
    """
    if toml:
        with open(toml, 'rb') as f:
            toml_obj = tomli.load(f)
        lid_params: dict = toml_obj['LID']
        targetlang = lid_params['targetlang']
        metalang = lid_params['metalang']
        target_labels = lid_params['target_labels']
        meta_labels = lid_params['meta_labels']
        empty = lid_params.get('empty', 'exclude')
    else:
        try:
            assert targetlang
            assert metalang
            assert target_labels
            assert meta_labels
        except AssertionError:
            raise ValueError(
                'If no toml argument is passed, targetlang, metalang, target_labels and meta_labels must all be passed. '\
                + f'{targetlang=}, {metalang=}, {target_labels=}, {meta_labels=}'
            )

    if (type(annotations) is str) or isinstance(annotations, Path):
        annotations = Path(annotations)
        if annotations.suffix == '.csv':
            annotations = pd.read_csv(annotations)
        elif annotations.suffix == '.eaf':
            annotations = eaf_data(annotations)
        else:
            raise ValueError('If annotations arg is a filepath, must be .csv or .eaf file.')
    annotations : pd.DataFrame

    annotations['lang'] = ''
    is_empty = annotations['text'].isna()
    if empty == 'target':
        annotations.loc[is_empty, 'lang'] = targetlang
    elif empty == 'meta':
        annotations.loc[is_empty, 'lang'] = metalang
    else:
        # empty = exclude
        annotations = annotations[~is_empty]
    

    if (target_labels == '*') and (meta_labels == '*'):
        raise ValueError("target_labels and meta_labels cannot both be '*'")

    if target_labels == '*':
        annotations.loc[~is_empty,'lang'] = annotations.loc[~is_empty, 'text']\
            .apply(lambda t: metalang if t in meta_labels else targetlang)
    elif meta_labels == '*':
        annotations.loc[~is_empty, 'lang'] = annotations.loc[~is_empty, 'text']\
            .apply(lambda t: targetlang if t in target_labels else metalang)
    else:
        annotations.loc[annotations['text'].isin(target_labels), 'lang'] = targetlang
        annotations.loc[annotations['text'].isin(meta_labels), 'lang'] = metalang
        annotations = annotations[annotations['lang']!='']
    
    return annotations