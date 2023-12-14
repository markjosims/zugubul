#!/usr/bin/env python3
"""fabfile"""
import os
from fabric import Connection
from paramiko.ssh_exception import PasswordRequiredException
from typing import Sequence, Optional
from pathlib import Path, PurePosixPath
from tqdm import tqdm

GUI = os.environ.get('GUI')

def run_script_on_server(
        argv: Sequence[str],
        in_files: Sequence[str],
        out_files: Sequence[str],
        server: str,
        server_python: str,
        passphrase: str,
        files_on_server: bool = False,
    ) -> int:
    server_args = [
        '--remote',
        '--password', passphrase,
        '--server', server,
        '--server_python', server_python,
        '--files_on_server',
    ]
    argv = [x for x in argv if x not in server_args]

    with connect(server, passphrase) as c:
        # replace local fps w server fps in arg str and put to server
        # TODO: dynamically check if input file is already present
        if not files_on_server:
            print(f'Uploading input files to server {server}...')
            server_dir = PurePosixPath('/tmp/zugubul/')
            c.run(f'mkdir -p {server_dir}')
            for local_fp in in_files:
                filename = Path(local_fp).name
                server_fp = PurePosixPath(server_dir/filename)
                
                print(f'Putting {local_fp} to {server_fp}')
                if os.path.isdir(local_fp):
                    put_dir(local_fp, server_fp, c)
                else:
                    c.put(str(local_fp), str(server_fp))
                argv = [arg if arg!=local_fp else str(server_fp) for arg in argv]

            # replace local fps with server fp but don't put to server
            server_out_files = []
            for local_fp in out_files:
                filename = Path(local_fp).name
                server_fp = PurePosixPath(server_dir/filename)
                server_out_files.append(server_fp)
                argv = [arg if arg!=local_fp else str(server_fp) for arg in argv]

        print('Executing command on server...')
        argv = [server_python, '-m', 'zugubul.main'] + argv[1:]
        arg_str = make_arg_str(argv)
        c.run(arg_str)

        if not files_on_server:
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

def put_dir(local_dir: Path, server_dir: Path, connection: Connection) -> None:
    print('Making dir', server_dir)
    connection.run(f'mkdir -p {server_dir}')
    for dirpath, dirnames, filenames in tqdm(
        list(os.walk(local_dir)),
        desc=f'Putting dir {dirpath} to server'
    ):
        for fname in tqdm(filenames, desc=f'Putting files in dir {dirpath}'):
            local_fpath = Path(dirpath)/fname
            relpath = local_fpath.relative_to(local_dir)
            server_fpath = PurePosixPath(server_dir)/relpath
            #print(f'Putting {local_fpath} to {server_fpath}')
            connection.put(str(local_fpath), str(server_fpath))
        for dir in tqdm(dirnames, desc=f'Putting dirs in dir {dirpath}'):
            local_subdir = Path(dirpath)/dir
            relpath = local_subdir.relative_to(local_dir)
            server_subdir = server_dir/relpath
            #print('Making dir', server_subdir)
            connection.run(f'mkdir -p {server_subdir}')

        

def connect(address: str, passphrase: Optional[str]) -> Connection:
    connect_kwargs = {
        'passphrase': passphrase or os.getenv('SSH_PASSPHRASE')
    }
    r = Connection(address, connect_kwargs=connect_kwargs)
    try:
        r.open()
    except PasswordRequiredException as e:
        if GUI:
            # can't use input w/ GUI
            raise e
        _passphrase = input(f'Type passphrase for connecting to {address}:')
        while not _passphrase:
            _passphrase = input(f'Type passphrase for connecting to {address}:')
        connect_kwargs['passphrase'] = _passphrase
        r = Connection(f'{address}', connect_kwargs=connect_kwargs)
        del _passphrase
    del connect_kwargs
    return r

if __name__ == '__main__':
    dirpath = r'C:\projects\dendi-speechproc\dendi-asr'
    serverpath = '/tmp/zugubul/dendi-asr'
    server = 'mjsimmons@grice.ucsd.edu'
    password = os.environ.get('PASSWORD')
    with connect(server, password) as c:
        put_dir(dirpath, serverpath, c)