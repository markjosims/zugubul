import visedit
from visedit.utils import Levenshtein as Lev
import argparse
from typing import Optional, Sequence

def init_error_parser(error_parser: argparse.ArgumentParser) -> None:
    error_parser.add_argument('IN')
    error_parser.add_argument('OUT')

def main(argv: Optional[Sequence[str]]) -> int:
    parser = argparse.ArgumentParser('Generate visualization of error from preds.json file.')
    init_error_parser(parser)

    args = parser.parse_args(argv)

    # read input file
    # make html file with visualization of each individual error
    # make dictionary counting edits for each possible char>char edit

    return 0

if __name__ == '__main__':
    main()