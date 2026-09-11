"""Finish the currently running local FEM job with verification and reporting.

This is a completion continuation of one authorized computation, not a recurring
schedule. No user messages, remote requests or new FEM levels are generated.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/resistivity_refinement_20260910'


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def process_running(pid):
    if sys.platform!='win32':
        raise RuntimeError('This completion controller requires the Windows launch process')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise OSError(ctypes.get_last_error(),'Cannot inspect the launched MATLAB process')
    try:
        status=ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(),'Cannot read MATLAB exit code')
        return status.value==259
    finally:
        kernel.CloseHandle(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matlab-pid',type=int,required=True)
    args=parser.parse_args()
    state_file=BASE/'completion_controller.json'
    state={'status':'waiting_for_current_fem','matlab_pid':args.matlab_pid,
           'started_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(state_file,state)
    try:
        while True:
            path=BASE/'progress.json'
            progress=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
            if progress.get('status')=='failed':
                raise RuntimeError('FEM failed: '+progress.get('message','See FEM logs'))
            if progress.get('status')=='fem_complete_analysis_pending': break
            if not process_running(args.matlab_pid):
                raise RuntimeError('MATLAB exited before full FEM completion; completed batches remain reusable')
            time.sleep(15)
        for script,stage in [('analyze_resistivity_refinement.py','verifying_and_analyzing'),('build_resistivity_refinement_report.py','building_notebook_and_html')]:
            state['status']=stage;state['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(state_file,state)
            with (BASE/f'{stage}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-X','utf8',str(TOOLS/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{script} failed with exit code {result.returncode}; see {stage}.log')
        state['status']='artifacts_built_final_review_pending'
        state['finished_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(state_file,state)
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);state['finished_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(state_file,state)
        raise


if __name__=='__main__':main()
