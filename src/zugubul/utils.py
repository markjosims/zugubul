import os
import argparse
from pathlib import Path
from typing import Optional, Callable, Union
from tqdm import tqdm
from pympi import Elan

def batch_funct(f: Callable,
                dir: str, suffix: str,
                file_arg: str,
                kwargs: Optional[dict] = None,
                out_path_f: Optional[Callable] = None,
                save_f: Optional[Callable] = None,
                recursive: bool = False,
                overwrite: bool = False,
                desc: str = 'Running batch process'
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
    data_files = list(data_files)
    for data_file in tqdm(data_files, total=len(data_files), desc=desc):
        out_path = out_path_f(data_file)
        if not overwrite and os.path.exists(out_path):
            tqdm.write(f'{out_path} already exists, skipping...')
            continue
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
    
def eaf_to_file_safe(eaf: Elan.Eaf, fp: Union[str, os.PathLike]):
    """
    Detects if backup version of eaf file already exists and, if so, deletes it before saving eaf to fp.
    """
    bak_fp = fp.replace('.eaf', '.bak')
    if os.path.exists(bak_fp):
        os.remove(bak_fp)
    eaf.to_file(fp)