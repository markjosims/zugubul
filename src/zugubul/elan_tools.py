import os
import pandas as pd
from pympi import Elan
from typing import Union, Optional, Sequence, Literal
from copy import deepcopy
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

"""
Contains following helper functions for processing .eaf files:
- trim:
        Remove all empty annotations from a .eaf file with name INFILE and writes to OUTFILE.
        If no OUTFILE argument supplied, overrwrites INFILE.
        TIER indicates which Elan tier to use, or 'default-lt' if no argument is passed.
        If a STOPWORD is passed after the TIER argument,
        remove all annotations with the indicated string value
        (e.g. elan_tools annotation.eaf annotation_out.eaf ipa x  removes all annotations that contain only the letter "x" on tier "ipa")
- merge:
        For a given TIER, add all annotations from FILE2 that don't overlap with those already present in FILE1.
        When an annotation from FILE2 overlaps with one from FILE1, cut annotation from FILE2 to only non-overlapping part,
        and add to FILE1, but only if non-overlapping part is less than OVERLAP (default 200ms).
        If OVERLAP=inf, do not add any overlapping annotations from FILE2.
- eaf_data:
        For a given elan file, create a csv with all annotation data for the whole file or for a given tier.
"""

def trim(
        eaf: Union[str, Elan.Eaf],
        tier: Optional[Union[str, Sequence]] = None,
        stopword: str = '',
        keepword: Optional[str] = None,
    ) -> Elan.Eaf:
    """
    Remove all annotations of the given tier which contain only the given stopword,
    or which don't contain the given keepword.
    By default, remove empty annotations from all tiers.
    """
    if (type(eaf) is not Elan.Eaf):
        eaf = Elan.Eaf(eaf)
    else:
        # avoid side effects from editing original eaf object
        eaf = deepcopy(eaf)

    if stopword and keepword:
        raise ValueError("Either stopword or keepword may be passed, not both.")

    if tier is None:
        tier = eaf.get_tier_names()
    elif type(tier) is str:
        tier = [tier,]
    for t in tier:
        annotations = eaf.get_annotation_data_for_tier(t)
        for a in annotations:
            a_start = a[0]
            a_end = a[1]
            a_mid = (a_start + a_end)/2
            a_value = a[2]
            if (keepword) and (a_value != keepword):
                removed = eaf.remove_annotation(t, a_mid)
                assert removed >= 1
            elif (not keepword) and (a_value == stopword):
                removed = eaf.remove_annotation(t, a_mid)
                assert removed >= 1
    return eaf

def merge(
        eaf_source: Union[str, Elan.Eaf],
        eaf_matrix: Union[str, Elan.Eaf],
        tier: Optional[Union[str, Sequence]] = None,
        overlap_behavior: Literal['keep_source', 'keep_matrix', 'keep_both'] = 'keep_source',
    ) -> Elan.Eaf:
    """
    eaf_matrix and eaf_source may be strings containing .eaf filepaths or Eaf objects
    For tier, add all annotations in eaf_source which are not already in eaf_matrix.
    If overlap_behavior=keep_source: Do not add annotations from EAF_MATRIX that overlap with annotations found in EAF_SOURCE.
    If keep_matrix: Do not add annotations from EAF_SOURCE that overlap with annotations found in EAF_MATRIX.
    If keep_both: Add all annotations from EAF_SOURCE, whether they overlap with EAF_MATRIX or not.
    Default behavior is keep_matrix
    """
    try:
        eaf_source_obj = open_eaf_safe(eaf_source, eaf_matrix)
        # don't overwrite eaf_source yet in case its filepath is needed to open eaf_matrix
        eaf_matrix = open_eaf_safe(eaf_matrix, eaf_source)
        eaf_source = eaf_source_obj
    except ValueError as error:
        raise ValueError(f'{error} {eaf_matrix=}, {eaf_source=}')


    if tier is None:
        tier = eaf_source.get_tier_names()
    elif type(tier) is str:
        tier = [tier,]

    for t in tier:
        eaf_source_annotations = eaf_source.get_annotation_data_for_tier(t)

        for start, end, value in eaf_source_annotations:
            if overlap_behavior != 'keep_both':
                matrix_overlap = eaf_matrix.get_annotation_data_between_times(t, start, end)

                if matrix_overlap and overlap_behavior == 'keep_matrix':
                    continue
                elif matrix_overlap and overlap_behavior == 'keep_source':
                    for interval_start, _, _ in matrix_overlap:
                        eaf_matrix.remove_annotation(id_tier=t, time=interval_start)

            eaf_matrix.add_annotation(t, start, end, value)

    return eaf_matrix

def open_eaf_safe(eaf1: Union[str, Elan.Eaf], eaf2: Union[str, Elan.Eaf]) -> Elan.Eaf:
    """
    If eaf1 is an Eaf object, return.
    If it is a str containing a filepath to an Eaf object, instantiate the Eaf and return.
    If it is a str containing a path to a directory, then eaf2 must be a path to an eaf file.
    Join filename of eaf2 to directory indicated by eaf1, instantiate Eaf object and return.
    """
    if type(eaf1) is Elan.Eaf:
        return deepcopy(eaf1)
    if os.path.isdir(eaf1):
        try:
            assert (type(eaf2) is not Elan.Eaf) and (os.path.isfile(eaf2))
            eaf2_name = os.path.split(eaf2)[-1]
        except AssertionError:
            raise ValueError(
                'If either eaf_source or eaf_matrix is a directory path, '\
                    +'the other must be a str containing a filepath, '\
                    +'not a directory filepath or an Eaf object.'
            )
        eaf1 = os.path.join(eaf1, eaf2_name)
    return Elan.Eaf(eaf1)

def eaf_data(
        eaf: Optional[str] = None,
        eaf_obj: Optional[Elan.Eaf] = None,
        tier: Union[str, Sequence[str], None] = None,
        media: Optional[str] = None,
    ) -> pd.DataFrame:
    if not eaf_obj:
        eaf_obj = Elan.Eaf(eaf)

    media_paths = [x['MEDIA_URL'] for x in eaf_obj.media_descriptors]
    if (media) and (media not in media_paths):
        raise ValueError(f'If media argument passed must be found in eaf linked files, {media=}, {media_paths=}')
    if not media:
        media = media_paths[0]
        # trim prefix added by ELAN
        media = media.replace('file:///', '')
        if len(media_paths) > 1:
            print(f'No media argument provided and eaf has multiple linked files. {media=}, {media_paths=}')

    if tier and type(tier) is str:
        tier = [tier,]
    else:
        tier = eaf_obj.get_tier_names()

    dfs = []

    for t in tier:
        annotations = eaf_obj.get_annotation_data_for_tier(t)
        start_times = [a[0] for a in annotations]
        end_times = [a[1] for a in annotations]
        values = [a[2] for a in annotations]

        t_df = pd.DataFrame(data={
            'start': start_times,
            'end': end_times,
            'text': values,
            'tier': t,
        })
        dfs.append(t_df)

    df = pd.concat(dfs)
    df['wav_source'] = media
    df['eaf_name'] = eaf

    return df

def snip_audio(
        annotations: Union[str, os.PathLike, Elan.Eaf, pd.DataFrame],
        out_dir: Union[str, os.PathLike],
        audio: Union[str, os.PathLike, AudioSegment, None] = None,
        tier: Union[str, Sequence[str], None] = None,
        allow_skip: bool = True,
    ) -> pd.DataFrame:
    """
    annotations may be an Eaf object, a pandas dataframe with the columns 'start', 'end' and 'wav_source',
    or a filepath pointing to a .eaf file or a .csv file with the same column structure.
    allow_skip determines whether to raise an error or merely print a warning if an audio file is not found.
    Returns a dataframe containing information for each annotation with the added column 'wav_clip',
    which points to a .wav file snipped to the start and end value for each annotation.
    """
    if type(annotations) is Elan.Eaf:
        df = eaf_data(eaf='', eaf_obj=annotations, tier=tier, media=audio)
        # no filepath for .eaf file passed so name is unknown
        del df['eaf_name']
    elif type(annotations) is pd.DataFrame:
        df = annotations
    else:
        # os.path.isfile(annotations)
        path = Path(annotations)
        suffix = path.suffix
        if suffix == '.eaf':
            df = eaf_data(annotations, tier=tier, media=audio)
        elif suffix == '.csv':
            df = pd.read_csv(annotations)
        else:
            raise ValueError('If passing filepath for annotations, must point to .eaf or .csv file.')
        
    if tier and (type(tier) is str):
        df = df[df['tier']==tier]
    elif tier: # type(tier) is Sequence
        df = df[df['tier'].isin(tier)]

    df['wav_clip'] = ''

    for wav_source in tqdm(df['wav_source'].unique(), desc='Snipping audio'):
        from_source = df['wav_source'] == wav_source
        try:
            wav_obj = AudioSegment.from_wav(wav_source)
        except FileNotFoundError as error:
            if allow_skip:
                num_skip = from_source.value_counts()[True]
                tqdm.write(f'{wav_source=} not found, skipping audio file and excluding {num_skip} rows from dataset.')
                df = df[~from_source]
                continue
            else:
                raise error

        wav_clips = df[from_source].progress_apply(
            lambda r: save_clip(
                start = r['start'],
                end = r['end'],
                source_fp = wav_source,
                out_dir = out_dir,
                wav_obj = wav_obj,
            ),
            axis=1,
            desc=f'Snipping clips from audio source {wav_source}.'
        )
        df.loc[from_source, 'wav_clip'] = wav_clips
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
    return df

def save_clip(
    start: int,
    end: int,
    source_fp: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
    wav_obj: Optional[AudioSegment] = None
    ) -> str:
    """
    Cut wav file from source_fp into clip from start to end timestamps (in number of milliseconds).
    Save clip to path/to/out_dir/source_fp[start-end].wav.
    Return path clip was saved to.
    """
    if not wav_obj:
        wav_obj = AudioSegment.from_wav(source_fp)
    
    source_stem = Path(source_fp).stem
    out_name = f'{source_stem}[{start}-{end}].wav'
    out_path = os.path.join(out_dir, out_name)
    
    clip = wav_obj[start:end]
    clip.export(out_path, format='wav')

    return out_path