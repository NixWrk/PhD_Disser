"""One-shot final report export after the associated calculation job terminates."""
from pathlib import Path
import argparse
import json
import sys
import time


def run(subject,model):
    model=Path(model).resolve();start=time.monotonic()
    while time.monotonic()-start<13*3600:
        try:status=json.loads((model/'job_status.local.json').read_text(encoding='utf8'))
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
        if status['status'] in ('forward_completed','preparation_failed','MATLAB_failed'):break
        time.sleep(5)
    else:raise TimeoutError('No terminal calculation status for report export')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'Colab Notebooks/scorpion_v2'))
    import build_session_subject_reports as B
    path=B.execute(subject)
    (model/'report_refresh.local.json').write_text(json.dumps(dict(subject_alias=subject,
        report=str(path),calculation_status=status['status'],updated_unix_s=time.time(),
        committed_or_pushed_automatically=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Updated local report:',path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--model',required=True)
    a=p.parse_args();run(a.subject,a.model)
