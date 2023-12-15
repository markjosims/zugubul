from typing import Union, Sequence, Tuple, Optional, Literal
import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import login
from zugubul.elan_tools import snip_audio, eaf_data
from math import ceil
from pathlib import Path
from pympi import Elan
import shutil
import tomli
import numpy as np
import string

def load_dataset_safe(dataset: str, split: Optional[str] = None) -> Union[Dataset, DatasetDict]:
    if os.path.exists(dataset):
        if split and os.path.exists(os.path.join(dataset, split)):
            return load_from_disk(os.path.join(dataset, split))
        return load_from_disk(dataset)
    return load_dataset(dataset, split=split)

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
    copy is a bool indicating whether the .wav clips should be copied to the new directories or moved (default False, i.e. move).
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
        out_relpath = split_dir / clip_fp.name
        out_fpath = out_dir / out_relpath
        if copy:
            shutil.copy(clip_fp, out_fpath)
            return out_relpath
        shutil.move(clip_fp, out_fpath)
        return str(out_relpath)

        

    for split, split_df in split_dict.items():
        print(f'Moving {len(split_df)} clips into {split} folder.')
        split_dir = Path(split)
        os.makedirs(out_dir / split_dir, exist_ok=True)
        split_clips = split_df['wav_clip'].apply(
            lambda fp: move_clip_to_split(Path(fp), split_dir)
        )

        eaf_data.loc[split_df.index, 'file_name'] = split_clips
    
    metadata = eaf_data[['file_name', label_col]]
    out_path = out_dir/'metadata.csv'
    metadata.to_csv(out_path, index=False)

    return out_path

def make_lm_dataset(
        annotations: Union[str, pd.DataFrame],
        text_col: str = 'text',
        splitsize: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        make_splits: bool = True,
) -> Union[Dataset, DatasetDict]:
    """
    Concatenates all text from a dataframe or .csv file containing ASR labels
    and returns a dataframe to be used for LM training.
    Assumes that all language labels have been trimmed,
    meaning you should run make_asr_labels on your .eaf files before this.
    """
    if type(annotations) is not pd.DataFrame:
        if not Path(annotations).suffix == '.csv':
            raise ValueError('Annotations must be pandas Dataframe or path to .csv file.')
        annotations = pd.read_csv(annotations)
    annotations=annotations.dropna(subset=text_col)
    add_period = lambda s: s+'.' if s.strip()[-1] not in string.punctuation else s.strip()

    text = ' '.join(annotations[text_col].apply(add_period))
    if not make_splits:
        return Dataset.from_dict({'text': [text]})
    train_size, val_size, _ = splitsize
    train_chars = int(len(text)*train_size)
    val_chars = train_chars+int(len(text)*val_size)
    train = Dataset.from_dict({'text': [text[:train_chars]]})
    val = Dataset.from_dict({'text': [text[train_chars:val_chars]]})
    test = Dataset.from_dict({'text': [text[val_chars:]]})
    
    return DatasetDict({
        'train': train,
        'val': val,
        'test': test
    })


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

    is_empty = annotations['text'].isna()
    annotations = annotations[~is_empty]

    print(f'Of {num_rows} original labels {len(annotations)} remain.')

    if process_length:
        annotations = process_annotation_length(annotations, min_gap=min_gap, min_length=min_length, lid=False)
    
    return annotations

def make_lid_labels(
        annotations: Union[str, pd.DataFrame],
        targetlang : Optional[str] = None,
        metalang: Optional[str] = None,
        target_labels: Optional[Sequence[str]] = None,
        meta_labels: Optional[Sequence[str]] = None,
        empty: Union[Literal['target'], Literal['meta'], Literal['exclude']] = 'exclude',
        process_length: bool = True,
        min_gap: int = 200,
        min_length: int = 1000,
        balance: bool = True,
        sample_strategy: Literal['downsample', 'upsample'] = 'downsample',
    ) -> pd.DataFrame:
    """
    annotations is a dataframe or str pointing to csv with a 'text' column.
    targetlang and metalang are strs indicating ISO codes or other unique identifiers
    for the target and metalanguage to be identified.
    target_labels is a list indicating all strs to be mapped to targetlang,
    meta_labels is a list indicating all strs to be mapped to metalang.
    Either target_labels or meta_labels may be None,
    in which case all non-empty strs not specified by the other arg will be mapped to that language.
    process_length is a bool indicating whether process_annotation_length should be called.
    The min_gap and min_length parameters are passed to process_annotation length.
    balance is a bool indicating whether balance_lid_data should be called to ensure each category has an equal number of rows.
    The sample_strategy arg is passed to balance_lid_data if called.
    If toml is not None, ignore other kwargs and read their values from the provided .toml file.
    Creates a new column 'lang' containing the label for each language.
    Returns dataframe.
    """
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
    

    if (not target_labels) and (not meta_labels):
        raise ValueError("target_labels and meta_labels cannot both be empty.")

    if not target_labels:
        annotations.loc[~is_empty,'lang'] = annotations.loc[~is_empty, 'text']\
            .apply(lambda t: metalang if t in meta_labels else targetlang)
    elif not meta_labels:
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
