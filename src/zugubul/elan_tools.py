import argparse
import os
from pympi import Elan
from typing import Union, Optional, Sequence, Callable
from zugubul.utils import is_valid_file, file_in_valid_dir, batch_funct
from copy import deepcopy

"""
Usage: elan_tools COMMAND...
Commands include:
- trim INFILE (OUTFILE) (TIER) (STOPWORD):
        Remove all empty annotations from a .eaf file with name INFILE and writes to OUTFILE.
        If no OUTFILE argument supplied, overrwrites INFILE.
        TIER indicates which Elan tier to use, or 'default-lt' if no argument is passed.
        If a STOPWORD is passed after the TIER argument,
        remove all annotations with the indicated string value
        (e.g. elan_tools annotation.eaf annotation_out.eaf ipa x  removes all annotations that contain only the letter "x" on tier "ipa")
- merge FILE1 FILE2 (OUTFILE) (TIER) (OVERLAP):
        For a given TIER, add all annotations from FILE2 that don't overlap with those already present in FILE1.
        When an annotation from FILE2 overlaps with one from FILE1, cut annotation from FILE2 to only non-overlapping part,
        and add to FILE1, but only if non-overlapping part is less than OVERLAP (default 200ms).
        If OVERLAP=inf, do not add any overlapping annotations from FILE2.
"""

def save_eaf_batch(data_file: str, out, out_folder: str) -> str:
    """
    Saves output provided by batch_funct to filepath with same name as data_file
    in directory out_folder.
    Returns path output is saved to.
    """
    out_path = os.path.join(
        out_folder, os.path.split(data_file)[-1]
    )
    out.to_file(out_path)
    return out_path

def trim(
        eaf: Union[str, Elan.Eaf],
        tier: Optional[Union[str, Sequence]] = None,
        stopword: str = '',
        save_funct: Optional[Callable] = None,
        recursive: bool = False
    ) -> Elan.Eaf:
    """
    Remove all annotations of the given tier which contain only the given stopword.
    By default, remove empty annotations from all tiers.
    """
    if type(eaf) is str:
        # batch processing
        if os.path.isdir(eaf):
            kwargs = {'stopword': stopword}
            return batch_funct(
                f=trim,
                dir=eaf,
                suffix='.eaf',
                file_arg='eaf',
                kwargs=kwargs,
                save_f=save_funct,
                recursive=recursive)
        eaf = Elan.Eaf(eaf)
    else:
        # avoid side effects from editing original eaf object
        eaf = deepcopy(eaf)

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
            if a_value == stopword:
                removed = eaf.remove_annotation(t, a_mid)
                assert removed >= 1
    return eaf

def merge(
        eaf1: Union[str, Elan.Eaf],
        eaf2: Union[str, Elan.Eaf],
        tier: Optional[Union[str, Sequence]] = None,
        overlap: int = 200,
        save_funct: Optional[Callable] = None,
        recursive: bool = False
    ) -> Elan.Eaf:
    """
    eaf1 and eaf2 may be strings containing .eaf filepaths or Eaf objects
    For tier, add all annotations in eaf2 which are not already in eaf1.
    If annotation from eaf2 overlaps with eaf1, add only the non-overlapping part.
    If the non-overlapping part is shorter than the overlap value, do not add.
    """
    if type(eaf1) is str:
        if os.path.isdir(eaf1):
            # batch processing
            if not os.path.isdir(eaf2):
                raise ValueError(
                    'If running batch process (by passing a folder path for eaf 1), '\
                    +'eaf2 must also be a folder path.'
                )
            kwargs = {'eaf2': eaf2}
            return batch_funct(
                f=merge,
                dir=eaf1,
                suffix='.eaf',
                file_arg='eaf1',
                kwargs=kwargs,
                save_f=save_funct,
                recursive=recursive)
        eaf1 = Elan.Eaf(eaf1)
    else:
        # avoid side effects from editing original eaf object
        eaf1 = deepcopy(eaf1)
    if type(eaf2) is str:
        # if folder path is passed for eaf2,
        # join name of eaf1 to folder
        if os.path.sidir(eaf2):
            eaf1_name = os.path.split(eaf1)[-1]
            eaf2 = os.path.join(eaf2, eaf1_name)
        eaf2 = Elan.Eaf(eaf2)
        # not editing eaf2 so no chance of side effects

    if tier is None:
        tier = eaf2.get_tier_names()
    elif type(tier) is str:
        tier = [tier,]

    for t in tier:
        eaf2_annotations = eaf2.get_annotation_data_for_tier(t)

        for start, end, value in eaf2_annotations:
            eaf1_overlap_eaf2_start = eaf1.get_annotation_data_at_time(t, start)
            f1_endtimes = [t[1] for t in eaf1_overlap_eaf2_start]
            start = max(f1_endtimes) if f1_endtimes else start
            
            eaf1_overlap_eaf2_end = eaf1.get_annotation_data_at_time(t, end)
            f1_startimes = [t[0] for t in eaf1_overlap_eaf2_end]
            end = min(f1_startimes) if f1_startimes else end

            does_overlap = f1_endtimes or f1_startimes

            if does_overlap and (end-start) < overlap:
                continue

            eaf1.add_annotation(t, start, end, value)

    return eaf1