"""Single Python entrypoint: creates a venv, installs dependencies, runs setup.

On Windows: py -3 run.py
On macOS:  python3 run.py
Use '--setup-only' for unattended runs; '--camera' after the first setup.
"""
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parent
VENV=ROOT/'.venv'
PYTHON=VENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def call(*args):
    print('>',*args,flush=True)
    subprocess.run([str(PYTHON),*map(str,args)],cwd=ROOT,check=True)


def main():
    if sys.version_info<(3,11):
        raise SystemExit('Python 3.11+ required. Install python.org Python and rerun.')
    if not PYTHON.is_file():
        print('Creating local Python virtual environment...')
        venv.EnvBuilder(with_pip=True).create(VENV)
    # An editable local install allows every run to pick up new project Python files.
    try:
        call('-m','pip','install','--disable-pip-version-check','-e','.')
    except subprocess.CalledProcessError:
        raise SystemExit('Dependency installation failed. Check internet connection, then rerun run.py.')
    if '--camera' in sys.argv:
        call('-m','facecap.cli','camera');return
    if '--test' in sys.argv:
        call('-m','pytest','-q');return
    if '--image' in sys.argv:
        i=sys.argv.index('--image')
        if i+1>=len(sys.argv):raise SystemExit('Usage: python run.py --image /path/photo.jpg')
        call('-m','facecap.cli','image',sys.argv[i+1]);return
    call('-m','facecap.cli','setup')
    call('-m','pytest','-q')
    if '--setup-only' not in sys.argv and sys.stdin.isatty():
        answer=input('Setup and tests complete. Open local webcam application now? [y/N] ').strip().lower()
        if answer in ('y','yes'):
            call('-m','facecap.cli','camera')

if __name__=='__main__':main()
