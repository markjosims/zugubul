import sys
import os
from pympi import Elan

docstr = """
Usage: elan_tools COMMAND...
Commands include:
- trim INFILE (OUTFILE) (TIER) (STOPWORD): remove all empty annotations from a .eaf file with name INFILE
        and writes to OUTFILE.
        If no OUTFILE argument supplied, overrwrites INFILE.
        TIER indicates which Elan tier to use, or 'default-lt' if no argument is passed.
        If a STOPWORD is passed after the TIER argument,
        remove all annotations with the indicated string value
        (e.g. elan_tools annotation.eaf annotation_out.eaf ipa x  removes all annotations that contain only the letter "x" on tier "ipa")
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
        if out_fp == in_fp:
            os.remove(in_fp)
        eaf_obj.to_file(out_fp)
    else:
        print('Unrecognized command')
        print_help()

if __name__ == '__main__':
    main()
