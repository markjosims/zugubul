import os
import argparse
from pathlib import Path
from typing import Optional, Callable

def batch_funct(f: Callable,
                dir: str, suffix: str,
                file_arg: str,
                kwargs: Optional[dict] = None,
                save_f: Optional[Callable] = None,
                recursive: bool = False,
            ) -> dict:
    """
    Takes a function f, a string containing a directory path dir, a suffix string,
    a string indicating the argument name for the data file to be passed to f, and a dict of kwargs for f.
    Returns a dict mapping output of f to each file in directory with the specified suffix.
    If save_f is provided, instead of returning the output of f, return the output of save_f on f and the data file string.
    """
    if not kwargs:
        kwargs = {}
    if recursive:
        data_files = Path(dir).glob(f'**/*{suffix}')
    else:
        data_files = Path(dir).glob(f'*{suffix}')
    out = {}
    for data_file in data_files:
        data_file = str(data_file)
        kwargs[file_arg] = data_file
        if save_f:
            out[data_file] = save_f(data_file, f(**kwargs))
        else:
            out[data_file] = f(**kwargs)

    return out

def is_valid_file(parser: argparse.ArgumentParser, arg: str) -> str:
    """
    Return error if filepath not found, return filepath otherwise.
    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist" % arg)
    else:
        return arg
    
def is_valid_dir(parser: argparse.ArgumentParser, arg: str) -> str:
    """
    Return error if directory path not found, return filepath otherwise.
    """
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist" % arg)
    else:
        return arg
    
def file_in_valid_dir(parser: argparse.ArgumentParser, arg: str) -> str:
    """
    Return error if file's parent directory not found, return filepath otherwise.
    """
    parent_dir = os.path.split(arg)[0]
    if not os.path.isdir(parent_dir):
        parser.error(f"The file {arg} is invalid because the parent directory {parent_dir} does not exist")
    else:
        return arg