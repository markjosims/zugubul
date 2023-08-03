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
