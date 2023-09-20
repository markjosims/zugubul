#!/usr/bin/env python3
"""fabfile"""
import os
from fabric import Connection
from paramiko.ssh_exception import PasswordRequiredException
from typing import Sequence, List
from pathlib import Path
from time import time

def run_script_on_server(
        argv: Sequence[str],
        in_files: Sequence[str],
        out_files: Sequence[str],
        server: str,
        server_python: str
    ) -> int:
    with connect(server) as c:
        # replace local fps w server fps in arg str and put to server
        # TODO: dynamically check if input file is already present
        print(f'Uploading input files to server {server}...')
        #server_dir = Path(f'/tmp/annotate_{time():.0f}/')
        server_dir = Path('/tmp/annotate_1695240969/')
        #c.run(f'mkdir {server_dir}')
        for local_fp in in_files:
            filename = Path(local_fp).name
            server_fp = server_dir/filename
            
            print(f'Putting {local_fp} to {server_fp}')
            #c.put(str(local_fp), str(server_fp))
            argv = [arg if arg!=local_fp else str(server_fp) for arg in argv]

        # replace local fps with server fp but don't put to server
        server_out_files = []
        for local_fp in out_files:
            filename = Path(local_fp).name
            server_fp = server_dir/filename
            server_out_files.append(server_fp)
            argv = [arg if arg!=local_fp else str(server_fp) for arg in argv]

        print('Executing command on server...')
        argv = [server_python, '-m', 'zugubul.main'] + argv[1:]
        arg_str = make_arg_str(argv)
        c.run(arg_str)

        print('Downloading output files from server...')
        for local_fp, server_fp in zip(out_files, server_out_files):
            c.get(str(server_fp), str(local_fp))
            print('Output filed saved to', local_fp)

def make_arg_str(argv: Sequence[str]) -> str:
    """
    Wrap any arguments broken by whitespace with quotes,
    then join arguments into str and return.
    """
    has_whitespace = lambda s: any(c.isspace() for c in s)
    wrap_if_whitespace = lambda s: '"'+s+'"' if has_whitespace(s) else s
    argv = [wrap_if_whitespace(s) for s in argv]
    arg_str = ' '.join(argv)
    return arg_str


def connect(address: str) -> Connection:
    connect_kwargs = {
        'passphrase': os.getenv('SSH_PASSPHRASE')
    }
    r = Connection(address, connect_kwargs=connect_kwargs)
    try:
        r.open()
    except PasswordRequiredException:
        _passphrase = input(f'Type passphrase for connecting to {address}:')
        while not _passphrase:
            _passphrase = input(f'Type passphrase for connecting to {address}:')
        connect_kwargs['passphrase'] = _passphrase
        r = Connection(f'{address}', connect_kwargs=connect_kwargs)
        del _passphrase
    del connect_kwargs
    return r

if __name__ == '__main__':
    wav_file = '/Users/markjos/Google Drive/Shared drives/Tira/Audacity Recording 200.wav'
    eaf_file = '/Users/markjos/Google Drive/Shared drives/Tira/Audacity Recording 200-MODEL.eaf'
    etf_file = '/Users/markjos/Google Drive/Shared drives/Tira/Tira_template.etf'
    run_script_on_server(
        argv=[
            'python',
            'annotate',
            wav_file,
            'markjosims/wav2vec2-large-mms-1b-tira-lid'
            'markjosims/wav2vec2-large-xls-r-300m-tira-colab',
            'TIC',
            eaf_file,
            '--inference_method', 'local',
            '-t', 'IPA Transcription',
            etf_file,
        ],
        in_files=[wav_file],
        out_files=[eaf_file],
        server='mjsimmons@grice.ucsd.edu',
        server_python='zugubul/.venv/bin/python'
    )