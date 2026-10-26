"""Wait for the own prepared mesh, then run the canonical MATLAB FEM profile."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run(subject,model,matlab):
    model=Path(model).resolve();root=Path(__file__).resolve().parents[1]
    logdir=model/'launch_logs';logdir.mkdir(exist_ok=False)
    status=model/'job_status.local.json'
    write(status,dict(subject_alias=subject,status='awaiting_prepared_own_mesh',pid=os.getpid(),FEM_was_run=False))
    start=time.monotonic()
    while not (model/'cem_contacts_input/preparation.local.json').is_file():
        failures=list(model.glob('*/preparation_failed.local.json'))
        if failures:
            failure=json.loads(failures[-1].read_text(encoding='utf8'))
            write(status,dict(subject_alias=subject,status='preparation_failed',failure_artifact=str(failures[-1]),
                message=failure['error'],FEM_was_run=False))
            raise RuntimeError(failure['error'])
        if time.monotonic()-start>12*3600:raise TimeoutError('Own preparation did not finish within the declared 12-hour launcher limit')
        time.sleep(5)
    def quote(p):return str(p).replace("'","''")
    script=(f"cd('{quote(root)}');run_pipeline('Profile','scorpion_session_fem','Subject','{subject}',"
        f"'ModelDir','{quote(model)}','OutputDir','{quote(model/'electrical')}');")
    with (logdir/'matlab.stdout.log').open('w',encoding='utf8') as log:
        process=subprocess.Popen([matlab,'-batch',script],stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(status,dict(subject_alias=subject,status='MATLAB_CEM_and_FEM_running',pid=os.getpid(),matlab_pid=process.pid,
            started_unix_s=time.time(),canonical_profile='scorpion_session_fem',FEM_completed=False))
        result=process.wait()
    complete=(model/'electrical/forward.local.json').exists() and result==0
    write(status,dict(subject_alias=subject,status='forward_completed' if complete else 'MATLAB_failed',
        exit_code=result,completed_unix_s=time.time(),log=str(logdir/'matlab.stdout.log'),
        FEM_completed=complete,quantitative_result_accepted=False))
    if not complete:raise RuntimeError('MATLAB FEM did not complete; inspect the pinned local log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',choices=['nik','yaroslav'],required=True);p.add_argument('--model',required=True)
    p.add_argument('--matlab',default='matlab')
    a=p.parse_args();run(a.subject,a.model,a.matlab)
