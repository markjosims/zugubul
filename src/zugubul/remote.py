#!/usr/bin/env python3
"""fabfile"""
import os
from fabric import Connection
from paramiko.ssh_exception import PasswordRequiredException
from typing import Sequence
from pathlib import Path

def run_script_on_server(
        argv: Sequence[str],
        file_args: Sequence[str],
        server: str,
        server_python: str
    ) -> int:
    with connect(server) as c:
        for local_fp in file_args:
            filename = Path(local_fp).name
            server_fp = 'tmp'/filename
            c.put(local_fp, server_fp)
            argv.replace
            argv = [arg if arg!=local_fp else server_fp for arg in argv]
        argv = [server_python,] + argv[1:]
        c.run(' '.join(argv))

def connect(address: str) -> Connection:
    connect_kwargs = {
        'passphrase': os.getenv('SSH_PASSPHRASE')
    }
    try:
        r = Connection(address, connect_kwargs=connect_kwargs)
    except PasswordRequiredException:
        _passphrase = input(f'Type passphrase for connecting to {address}:')
        while not _passphrase:
            _passphrase = input(f'Type passphrase for connecting to {address}:')
        r = Connection(f'{address}', connect_kwargs=connect_kwargs)
        del _passphrase
    return r

if __name__ == '__main__':
    with connect('mjsimmons@grice.ucsd.edu') as c:
        print(c.run('zugubul/.venv/bin/python -m zugubul.main -h'))