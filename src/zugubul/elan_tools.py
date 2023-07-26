import sys
import os
from pympi import Elan

docstr = """
Usage: elan_tools COMMAND...
Commands include:
- trim INFILE (OUTFILE) (TIER) (STOPWORD):
        Remove all empty annotations from a .eaf file with name INFILE and writes to OUTFILE.
        If no OUTFILE argument supplied, overrwrites INFILE.
        TIER indicates which Elan tier to use, or 'default-lt' if no argument is passed.
        If a STOPWORD is passed after the TIER argument,
        remove all annotations with the indicated string value
        (e.g. elan_tools annotation.eaf annotation_out.eaf ipa x  removes all annotations that contain only the letter "x" on tier "ipa")
- merge FILE1 FILE2 (OUTFILE) (TIER) (GAP):
        For a given TIER, add all annotations from FILE2 that don't overlap with those already present in FILE1.
        When an annotation from FILE2 overlaps with one from FILE1, cut annotation from FILE2 to only non-overlapping part,
        and add to FILE1, but only if non-overlapping part is less than GAP (default 200ms).
        If GAP=inf, do not add any overlapping annotations from FILE2.
"""

def print_help():
    print(docstr)

def trim(eaf_fp: str, tier: str = 'default-lt', stopword: str = '') -> Elan.Eaf:
    """
    Remove all annotations of the given tier which contain only the given stopword.
    By default, remove all empty annotations from tier 'default-lt'
    """
    eaf_obj = Elan.Eaf(eaf_fp)
    annotations = eaf_obj.get_annotation_data_for_tier(tier)
    for a in annotations:
        a_start = a[0]
        a_end = a[1]
        a_mid = (a_start + a_end)/2
        a_value = a[2]
        if a_value == stopword:
            removed = eaf_obj.remove_annotation(tier, a_mid)
            assert removed >= 1
    return eaf_obj

def merge(file1: str, file2: str, tier: str = 'default-lt', gap: int = 200) -> Elan.Eaf:
    """
    For tier, add all annotations in file2 which are not already in file1.
    If annotation from file2 overlaps with file1, add only the non-overlapping part.
    If the non-overlapping part is shorter than the gap value, do not add.
    """
    file1_obj = Elan.Eaf(file1)
    file2_obj = Elan.Eaf(file2)

    file2_annotations = file2_obj.get_annotation_data_for_tier(tier)

    for start, end, value in file2_annotations:
        file1_overlap_file2_start = file1_obj.get_annotation_data_at_time(tier, start)
        f1_endtimes = [t[1] for t in file1_overlap_file2_start]
        start = max(f1_endtimes) if f1_endtimes else start
        
        file1_overlap_file2_end = file1_obj.get_annotation_data_at_time(tier, end)
        f1_startimes = [t[0] for t in file1_overlap_file2_end]
        end = min(f1_startimes) if f1_startimes else end

        if (end-start) < gap:
            continue

        file1_obj.add_annotation(tier, start, end, value)

    return file1_obj

def main():
    if len(sys.argv) == 1:
        print('Usage: python elan_tools.py COMMAND (ARG)')
        print('Run `elan_tools -h` for a list of available commands.')
    command = sys.argv[1]
    if command == '-h':
        print_help()
    elif command == 'trim':
        if len(sys.argv) <= 2:
            print('Include FILENAME argument pointing to .eaf file.')
            return
        in_fp = sys.argv[2]
        out_fp = sys.argv[3] if len(sys.argv) > 3 else in_fp
        tier = sys.argv[4] if len(sys.argv) > 4 else 'default-lt'
        stopword = sys.argv[5] if len(sys.argv) > 5 else ''
        eaf_obj = trim(in_fp, tier, stopword)
        if os.path.exists(out_fp):
            os.remove(out_fp)
        eaf_obj.to_file(out_fp)
    elif command == 'merge':
        if len(sys.argv) <= 3:
            print('Include FILE1 and FILE2 arguments pointing to .eaf files.')
            return
        file1 = sys.argv[2]
        file2 = sys.argv[3]
        out_fp = sys.argv[4] if len(sys.argv) > 4 else file1
        tier = sys.argv[5] if len(sys.argv) > 5 else 'default-lt'
        gap = sys.argv[6] if len(sys.argv) > 6 else 200
        try:
            gap = float(gap)
        except ValueError():
            print('Enter value for gap as a number of ms. Defaulting to 200ms')
            gap = 200
        eaf_obj = merge(file1, file2, tier, gap)
        if os.path.exists(out_fp):
            os.remove(out_fp)
        eaf_obj.to_file(out_fp)
    else:
        print('Unrecognized command')
        print_help()

if __name__ == '__main__':
    main()
