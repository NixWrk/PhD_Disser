"""Launch the isolated FEM review without a visible helper window."""
from pathlib import Path
import subprocess
import os
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
run_tests = 'false' if '--skip-passed-tests' in sys.argv else 'true'
with (here / 'current_matlab.log').open('a', encoding='utf-8') as log:
    proc = subprocess.run(['matlab', '-batch',
        "addpath('notebooks/review_current_2026_09_08'); verify_current_fem(" + run_tests + ")"],
        cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
print('MATLAB exit', proc.returncode, flush=True)
raise SystemExit(proc.returncode)
