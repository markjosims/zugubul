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

def trim(
        eaf: Union[str, Elan.Eaf],
        tier: Optional[Union[str, Sequence]] = None,
        stopword: str = '',
    ) -> Elan.Eaf:
    """
    Remove all annotations of the given tier which contain only the given stopword.
    By default, remove empty annotations from all tiers.
    """
    if type(eaf) is str:
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
        eaf_source: Union[str, Elan.Eaf],
        eaf_matrix: Union[str, Elan.Eaf],
        tier: Optional[Union[str, Sequence]] = None,
        overlap: int = 200,
    ) -> Elan.Eaf:
    """
    eaf_matrix and eaf_source may be strings containing .eaf filepaths or Eaf objects
    For tier, add all annotations in eaf_source which are not already in eaf_matrix.
    If annotation from eafs overlaps with eaf_matrix, add only the non-overlapping part.
    If the non-overlapping part is shorter than the overlap value, do not add.
    """
    try:
        eaf_source_obj = open_eaf_safe(eaf_source, eaf_matrix)
        # don't overwrite eaf_source in case its filepath is needed to open eaf_matrix
        eaf_matrix = open_eaf_safe(eaf_matrix, eaf_source)
        eaf_source = eaf_source_obj
    except ValueError as error:
        raise ValueError(f'{error} {eaf_matrix=}, {eaf_source=}')


    if tier is None:
        tier = eaf_source.get_tier_names()
    elif type(tier) is str:
        tier = [tier,]

    for t in tier:
        eaf2_annotations = eaf_source.get_annotation_data_for_tier(t)

        for start, end, value in eaf2_annotations:
            eafm_overlap_eaf2_start = eaf_matrix.get_annotation_data_at_time(t, start)
            f1_endtimes = [t[1] for t in eafm_overlap_eaf2_start]
            start = max(f1_endtimes) if f1_endtimes else start
            
            eafm_overlap_eaf2_end = eaf_matrix.get_annotation_data_at_time(t, end)
            f1_startimes = [t[0] for t in eafm_overlap_eaf2_end]
            end = min(f1_startimes) if f1_startimes else end

            does_overlap = f1_endtimes or f1_startimes

            if does_overlap and (end-start) < overlap:
                continue

            eaf_matrix.add_annotation(t, start, end, value)

    return eaf_matrix

def open_eaf_safe(eaf1: Union[str, Elan.Eaf], eaf2: Union[str, Elan.Eaf]) -> Elan.Eaf:
    """
    If eaf1 is an Eaf object, return.
    If it is a str containing a filepath to an Eaf object, instantiate the Eaf and return.
    If it is a str containing a path to a directory, then eaf2 must be a path to an eaf file.
    Join filename of eaf2 to directory indicated by eaf1, instantiate Eaf object and return.
    """
    if type(eaf1) is str:
        if os.path.isdir(eaf1):
            try:
                assert (type(eaf2) is str) and (os.path.isfile(eaf2))
                eaf2_name = os.path.split(eaf2)[-1]
            except AssertionError:
                raise ValueError(
                    'If either eaf_source or eaf_matrix is a directory path, '\
                        +'the other must be a str containing a filepath, '\
                        +'not a directory filepath or an Eaf object.'
                )
            eaf1 = os.path.join(eaf1, eaf2_name)
        return Elan.Eaf(eaf1)
    return eaf1