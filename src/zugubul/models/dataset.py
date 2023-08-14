from typing import Union
import os

from datasets import Dataset, load_dataset

def init_lid_dataset(dirpath: Union[str, os.PathLike]) -> Dataset:
    """
    dirpath indicates dataset directory
    directory must be of following structure:
    dirpath
        metadata.csv
        train
            recording1.wav
            recording2.wav
            ...
        val
            recording3.wav
            recording4.wav
            ...
        test
            recording5.wav
            recording6.wav
            ...
    """
    return load_dataset('audiofolder', data_dir=dirpath)
