from typing import Union, Sequence, Tuple, Optional, Literal
import os
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import login
from zugubul.elan_tools import snip_audio, eaf_data
from math import ceil
from pathlib import Path
from pympi import Elan
import shutil
import tomli
import numpy as np

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
    if (type(eaf_data) is not pd.DataFrame):
        eaf_data = pd.read_csv(eaf_data)

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
        print(f'Moving {len(split_df)} clips into {split} folder.')
        split_dir = Path(split)
        os.makedirs(split_dir, exist_ok=True)
        split_clips = split_df['wav_clip'].apply(
            lambda fp: move_clip_to_split(Path(fp), split_dir)
        )

        eaf_data.loc[split_df.index, 'file_name'] = split_clips
    
    metadata = eaf_data[['file_name', label_col]]
    out_path = out_dir/'metadata.csv'
    metadata.to_csv(out_path, index=False)

    return out_path

def make_asr_labels(
        annotations: Union[str, pd.DataFrame],
        lid_labels: Optional[Sequence[str]] = None,
        process_length: bool = True,
        min_gap: int = 200,
        min_length: int = 1000,
) -> pd.DataFrame:
    if type(annotations) is not pd.DataFrame:
        annotations = Path(annotations)
        if annotations.suffix == '.csv':
            annotations = pd.read_csv(annotations)
        elif annotations.suffix == '.eaf':
            annotations = eaf_data(annotations)
        else:
            raise ValueError('If annotations arg is a filepath, must be .csv or .eaf file.')
    annotations : pd.DataFrame

    print('Removing LID and empty labels from ASR dataset.')
    num_rows = len(annotations)
    is_lid_label = annotations['text'].isin(lid_labels)
    annotations = annotations[~is_lid_label]

    is_empty = annotations['text'] == ''
    annotations = annotations[~is_empty]

    print(f'Of {num_rows} original labels {len(annotations)} remain.')

    if process_length:
        annotations = process_annotation_length(annotations, min_gap=min_gap, min_length=min_length, lid=False)
    
    return annotations

def make_lid_labels(
        annotations: Union[str, pd.DataFrame],
        targetlang : Optional[str] = None,
        metalang: Optional[str] = None,
        target_labels: Union[Sequence[str], Literal['*'], None] = None,
        meta_labels: Union[Sequence[str], Literal['*'], None] = None,
        empty: Union[Literal['target'], Literal['meta'], Literal['exclude']] = 'exclude',
        process_length: bool = True,
        min_gap: int = 200,
        min_length: int = 1000,
        balance: bool = True,
        sample_strategy: Literal['downsample', 'upsample'] = 'downsample',
        toml: Optional[Union[str, os.PathLike]] = None,
    ) -> pd.DataFrame:
    """
    annotations is a dataframe or str pointing to csv with a 'text' column.
    targetlang and metalang are strs indicating ISO codes or other unique identifiers
    for the target and metalanguage to be identified.
    target_labels is a list indicating all strs to be mapped to targetlang,
    meta_labels is a list indicating all strs to be mapped to metalang.
    Either target_labels or meta_labels may be the str '*',
    in which case all non-empty strs not specified by the other arg will be mapped to that language.
    process_length is a bool indicating whether process_annotation_length should be called.
    The min_gap and min_length parameters are passed to process_annotation length.
    balance is a bool indicating whether balance_lid_data should be called to ensure each category has an equal number of rows.
    The sample_strategy arg is passed to balance_lid_data if called.
    If toml is not None, ignore other kwargs and read their values from the provided .toml file.
    Creates a new column 'lang' containing the label for each language.
    Returns dataframe.
    """
    if toml:
        with open(toml, 'rb') as f:
            toml_obj = tomli.load(f)
        lid_params: dict = toml_obj['LID']
        targetlang = lid_params.get('targetlang', targetlang)
        metalang = lid_params.get('metalang', metalang)
        target_labels = lid_params.get('target_labels', target_labels)
        meta_labels = lid_params.get('meta_labels', meta_labels)
        empty = lid_params.get('empty', empty)
        process_length = lid_params.get('process_length', process_length)
        min_gap = lid_params.get('min_gap', min_gap)
        min_length = lid_params.get('min_length', min_length)
        balance - lid_params.get('balance', balance)
        sample_strategy = lid_params.get('sample_strategy', sample_strategy)
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

    if type(annotations) is not pd.DataFrame:
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
    
    if process_length:
        annotations = process_annotation_length(annotations, min_gap=min_gap, min_length=min_length, lid=True)

    if balance:
        annotations = balance_lid_data(df=annotations, sample_strategy=sample_strategy)

    return annotations

def balance_lid_data(df: pd.DataFrame, sample_strategy: Literal['upsample', 'downsample'] = 'downsample') -> pd.DataFrame:
    """
    Takes a dataframe with column 'lang', counts the number of each language represented,
    and either upsamples the underrepresented categories using Gaussian noise (TODO: implement upsampling),
    or downsamples overrepresented categories until each category has an equal distribution.
    """
    num_rows = len(df)
    if sample_strategy == 'upsample':
        raise NotImplementedError('Upsampling LID data has not been implemented yet.')
    min_ct = df['lang'].value_counts().min()

    print(f'Downsampling all categories to {min_ct} rows.')
    for lang in df['lang'].unique():
        lang_ct = df['lang'].value_counts()[lang]
        downsample_ct = lang_ct - min_ct
        if not downsample_ct:
            continue

        drop_rows = df[df['lang']==lang].sample(downsample_ct)
        print(f'Language {lang} was downsampled from {lang_ct} rows to {min_ct}.')
        df = df.drop(drop_rows.index)

    print(f'Dataset resampled from {num_rows} rows to {len(df)}.')

    return df

def process_annotation_length(
        data: Union[pd.DataFrame, Elan.Eaf, str, os.PathLike],
        min_gap: int = 200,
        min_length: int = 1000,
        lid: bool = False
    ) -> pd.DataFrame: 
    """
    df is a dataframe or .csv filepath containing 'start', 'end', 'wav_source' and either 'text' or 'lang' columns.
    min_gap is a time in milliseconds indicating how far away two annotations must be to not be merged.
    min_length is a time in milliseconds indicating how long an annotation must be to not be deleted
    (after adjacent annotations have been merged).
    If lid = True, use 'lang' column and only merge adjacent annotations that have the same value for 'lang',
    else merge any adjacent annotations and concatenate the value for the 'text' column.
    """
    if type(data) is not pd.DataFrame:
        if type(data) is Elan.Eaf:
            data = eaf_data(eaf_obj=data)
        else:
            suffix = Path(data).suffix
            if suffix == '.csv':
                data = pd.read_csv(data)
            elif suffix == '.eaf':
                data = eaf_data(eaf=data)
            else:
                raise ValueError(
                    "data must be pandas DataFrame, path to a .csv or .eaf file, or an Eaf object."
                )
    data: pd.DataFrame
    num_rows = len(data)
    print(f'Merging annotations within gap of {min_gap} ms...')
    if lid:
        lang_dfs = []
        for lang in data['lang'].unique():
            has_lang = data['lang']==lang
            lang_df = process_annotation_length_innerloop(data[has_lang], min_gap, lid)
            lang_dfs.append(lang_df)
        data = pd.concat(lang_dfs)
    else:
        data = process_annotation_length_innerloop(data, min_gap, lid)
    merged_len = len(data)
    print(f'After merging {merged_len} rows remain from {num_rows}.')

    print(f'Dropping rows shorter than {min_length} ms. in duration...')

    duration = data['end'] - data['start']
    is_min_duration = duration >= min_length
    data = data[is_min_duration]

    drop_len = len(data)
    print(f'After dropping {drop_len} rows remain from {merged_len}.')

    return data

def process_annotation_length_innerloop(df: pd.DataFrame, min_gap: int, lid: bool):
    for file in df['wav_source'].unique():
        has_file = df['wav_source']==file
        sorted_by_start = df[has_file].sort_values(by='start')

        gap = np.array(sorted_by_start['start'])[1:] - np.array(sorted_by_start['end'])[:-1]
        within_gap = gap < min_gap
        sorted_by_start['is_w_gap'] = np.append(within_gap, [False])
        last_start = None
        last_text = ''
        for i, is_w_gap in enumerate(sorted_by_start['is_w_gap']):
            # when this annotation is within gap of following,
            # set last_start to annotation start if last_start is None
            # if not LID, append annotation text to last_text
            if is_w_gap and (last_start is None):
                last_start = sorted_by_start.iloc[i]['start']
            if is_w_gap and (not lid):
                last_text += sorted_by_start.iloc[i]['text']

            # when this annotation is not within gap of following, add data to dataframe and reset last_start and last_text
            if (not is_w_gap) and (not last_start is None):
                sorted_by_start.iloc[i, sorted_by_start.columns.get_loc('start')] = last_start
                last_start = None
            if (not is_w_gap) and (not lid):
                sorted_by_start.iloc[i, sorted_by_start.columns.get_loc('text')] = last_text + ' ' + sorted_by_start.iloc[i]['text']
                last_text = ''
        
        resorted = sorted_by_start.sort_index()

        df.loc[has_file, 'start'] = resorted['start']
        if not lid:
            df.loc[has_file, 'text'] = resorted['text']
        df = df.drop(sorted_by_start[sorted_by_start['is_w_gap']].index)
    return df
