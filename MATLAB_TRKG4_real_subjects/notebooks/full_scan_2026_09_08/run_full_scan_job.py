"""One local run: exhaustive MATLAB CEM, independent best QC, verified report."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matlab', required=True, type=Path)
    parser.add_argument('--tag', default='fullscan_20260909_v3')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--green-limit', type=int, default=10000)
    parser.add_argument('--centre-chunk', type=int, default=64)
    parser.add_argument('--u-min',type=int,default=-80)
    parser.add_argument('--u-max',type=int,default=80)
    parser.add_argument('--v-min',type=int,default=-120)
    parser.add_argument('--v-max',type=int,default=120)
    parser.add_argument('--max-blocks',type=int)
    parser.add_argument('--reuse-geometry-tag',default='fullscan_20260908_v2')
    args = parser.parse_args()
    if args.u_min>args.u_max or args.v_min>args.v_max or args.max_blocks is not None and args.max_blocks<1:
        raise ValueError('Invalid grid or block limit')
    if args.reuse_geometry_tag and not re.fullmatch(r'[A-Za-z0-9_-]+',args.reuse_geometry_tag):
        raise ValueError('Invalid geometry reuse tag')
    centres=(args.u_max-args.u_min+1)*(args.v_max-args.v_min+1)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('Invalid result tag')
    if min(args.workers, args.green_limit, args.centre_chunk) < 1:
        raise ValueError('Runtime limits must be positive')
    out = ROOT / 'output' / ('full_pose_scan_' + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'job.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = {'result_tag': args.tag, 'pid': os.getpid(),
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'grid': {'centres': centres, 'angles': 41, 'poses': centres*41},
             'workers': args.workers, 'green_node_limit': args.green_limit,
             'complete': False}
    def publish(stage, **extra):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        tmp = out / 'job_status.tmp.json'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(out / 'job_status.json')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        publish('starting')
        snapshot = out / 'source_snapshot.zip'
        if not snapshot.exists():
            sources = [ROOT / 'run_pipeline.m', Path(__file__).resolve(),
                       ROOT / 'notebooks/build_trkg4_full_scan_report.py',
                       ROOT / 'notebooks/full_scan_2026_09_08/verify_trkg4_full_scan_best.m']
            for folder, pattern in [('src', '*.m'), ('vendor_stl_eidors', '*.m'),
                    ('tools', 'trkg4_local*.m'), ('tools/full_pose_scan', 'full_scan*.m'),
                    ('tools/full_pose_scan', 'full_scan_project_mex.*')]:
                sources.extend((ROOT / folder).glob(pattern))
            sources.extend((ROOT / 'notebooks/full_scan_2026_09_08/patch_shape_reference').glob('*'))
            sources.extend((ROOT / 'notebooks/full_scan_2026_09_08').glob('patch_shape_*.json'))
            sources.append(ROOT / 'notebooks/full_scan_2026_09_08/verify_trkg4_patch_shape_compatibility.m')
            with zipfile.ZipFile(snapshot, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(set(sources)):
                    archive.write(file, file.relative_to(ROOT).as_posix())
        python_literal=str(Path(sys.executable).resolve()).replace("'","''")
        maximum='Inf' if args.max_blocks is None else str(args.max_blocks)
        batch = (
            "r=run_pipeline('Profile','full_pose_scan','SourceTag','auditfix_20260908',"
            f"'ResultTag','{args.tag}','Workers',{args.workers},"
            f"'GreenNodeLimit',{args.green_limit},'CentreChunk',{args.centre_chunk},"
            f"'UValues',{args.u_min}:{args.u_max},'VValues',{args.v_min}:{args.v_max},"
            f"'Streaming',true,'ReuseGeometryTag','{args.reuse_geometry_tag}',"
            f"'MaxBlocks',{maximum},'ReportPython','{python_literal}');"
            "clear full_scan_green;eidors_cache('clear_all');"
            "pool=gcp('nocreate');if ~isempty(pool),delete(pool);end;"
            "addpath('notebooks/full_scan_2026_09_08');"
            f"if r.complete,verify_trkg4_full_scan_best('{args.tag}');end;"
        )
        publish('matlab_full_scan_and_independent_qc')
        with (out / 'matlab_console.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run([str(args.matlab), '-batch', batch], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        if completed.returncode:
            raise RuntimeError(f'MATLAB exited {completed.returncode}; see matlab_console.log')
        progress=json.loads((out/'progress.json').read_text(encoding='utf-8'))
        if not progress.get('complete'):
            publish('paused_after_block_limit',complete=False,progress=progress,
                    report_html=str(out/'live/report/full_scan.html'))
            return
        qc = json.loads((out / 'independent_best_qc.json').read_text(encoding='utf-8'))
        if not qc.get('passed'):
            raise RuntimeError('Independent full CEM QC did not pass')
        publish('building_verified_report')
        with (out / 'report_build.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'notebooks/build_trkg4_full_scan_report.py'),
                '--input-dir', str(out)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        if completed.returncode:
            raise RuntimeError(f'Report builder exited {completed.returncode}; see report_build.log')
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        report = out / 'report/full_scan.html'
        if not report.is_file() or not summary.get('complete'):
            raise RuntimeError('Complete report is missing')
        publish('checking_3d_report')
        from playwright.sync_api import sync_playwright
        errors, external = [], []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1450, 'height': 1000}, device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: external.append(request.url)
                    if request.url.startswith(('http://', 'https://')) else None)
            page.goto(report.as_uri(), wait_until='load', timeout=120000)
            page.wait_for_function("!document.getElementById('angle').disabled", timeout=120000)
            layers = page.locator('#angle option').count()
            if layers != 42:
                raise RuntimeError(f'Expected best plus 41 angle layers, got {layers}')
            data = page.evaluate("""() => {const d=document.getElementById('body').data;
                return {valid:d[1].x.length,invalid:d[2].x.length,missing:d[3].x.length,
                        faces:d[0].i.length,outside:d[4].x.length};} """)
            if (data['valid'] != summary['counts']['centres_with_valid_angles'] or
                    data['valid'] + data['invalid'] + data['outside'] != centres or data['missing'] or data['faces'] != 155120):
                raise RuntimeError('3D report point counts differ from verified scan')
            page.screenshot(path=str(out / 'report/full_scan_3d_browser.png'), full_page=True)
            page.select_option('#angle', '20')
            page.wait_for_function("!document.getElementById('angle').disabled", timeout=120000)
            page.select_option('#angle', 'best')
            page.wait_for_function("!document.getElementById('angle').disabled", timeout=120000)
            browser.close()
        if errors or external:
            raise RuntimeError(f'Offline 3D report errors: {errors}; external requests: {external}')
        (out / 'report/browser_checks.json').write_text(json.dumps(
            {'passed': True, 'layers': layers, 'best_layer': data,
             'javascript_errors': errors, 'external_http_requests': external}, indent=2), encoding='utf-8')
        publish('complete', complete=True, counts=summary['counts'], best=summary['bestrecord'],
                independent_best_qc=qc, report_html=str(report),
                report_png=str(out / 'report/full_scan.png'))
    except BaseException as error:
        publish('failed', complete=False, error=str(error))
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
