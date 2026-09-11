"""Read stopped MATLAB geometry checkpoints; build a local diagnostic map, never run FEM."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "output/full_pose_scan_fullscan_20260908_v2"
LABELS = ["Не проверено", "Геометрия прошла", "Отказ контроля геометрии",
          "Не удалось набрать площадь", "Нет свободной начальной грани", "Общие узлы электродов"]
FAILURES = {"trkg4:electrodeDiagnostics": 2, "electrode_faces_by_area:patchTooSmall": 3,
            "electrode_faces_by_area:noSeed": 4, "trkg4:patchOverlap": 5}
COLOURS = [[0, "#a50026"], [.2, "#f46d43"], [.4, "#fee08b"], [.5, "#ffffbf"],
           [.6, "#d9ef8b"], [.8, "#66bd63"], [1, "#006837"]]
MISSING = "#cdd2d8"

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def text_cell(value):
    return "".join(str(x) for x in np.asarray(value).ravel())

def load_geometry(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "run_manifest.json")
    physics = manifest["physics"]
    u, v, phi = (np.atleast_1d(physics[k]).astype(float) for k in ("u_mm", "v_mm", "phi_deg"))
    assert all(np.array_equal(x, y) for x, y in
               [(u, np.arange(-120, 121)), (v, np.arange(-120, 121)), (phi, np.arange(-20, 21))])
    centres = pd.read_csv(directory / "centres.csv")
    n, a = len(centres), len(phi)
    assert n == len(u) * len(v) == 58081
    assert np.array_equal(centres.centre_index, np.arange(1, n + 1))
    assert np.array_equal(centres.u_mm, np.tile(u, len(v)))
    assert np.array_equal(centres.v_mm, np.repeat(v, len(u)))
    assert np.isfinite(centres[["x_mm", "y_mm", "z_mm"]]).all().all()
    state = np.zeros((n, a), dtype=np.uint8)
    inputs = []
    def remember(path):
        path = Path(path)
        inputs.append({"path":str(path.relative_to(directory)).replace("\\", "/"),
                       "sha256":digest(path)})
    for name in ("run_manifest.json", "centres.csv", "surface.mat", "stopped_by_user_20260909.json"):
        remember(directory / name)
    ignored = []
    blocks = sorted(p for p in (directory / "geometry").glob("block_*.mat")
                    if re.fullmatch(r"block_\d{5}\.mat", p.name))
    committed = 0
    for path in blocks:
        sidecar = Path(str(path) + ".json")
        if not sidecar.is_file():
            ignored.append(path.name)
            continue
        check = read_json(sidecar)
        assert check["fingerprint"] == manifest["fingerprint"], path.name
        assert digest(path) == check["sha256"], path.name
        b = loadmat(path, variable_names=["centre_ids", "phi_deg", "status_code", "failure_code", "fingerprint"])
        assert text_cell(b["fingerprint"]) == manifest["fingerprint"]
        raw_ci, raw_phi = b["centre_ids"].ravel(), b["phi_deg"].ravel()
        ci = raw_ci.astype(int) - 1
        ai = np.searchsorted(phi, raw_phi)
        assert np.array_equal(raw_ci, ci + 1) and np.all((ci >= 0) & (ci < n))
        assert np.all(ai < a) and np.array_equal(phi[ai], raw_phi)
        assert len(np.unique(ci * a + ai)) == len(ci) and not state[ci, ai].any()
        status = b["status_code"].ravel()
        assert set(status) <= {1, 2}
        failure = b["failure_code"].ravel()
        mapped = np.ones(len(ci), dtype=np.uint8)
        for k in np.flatnonzero(status == 2):
            mapped[k] = FAILURES[text_cell(failure[k])]
        assert all(not text_cell(failure[k]) for k in np.flatnonzero(status == 1))
        state[ci, ai] = mapped
        inputs.extend([{"path":"geometry/" + path.name, "sha256":check["sha256"]},
                       {"path":"geometry/" + sidecar.name, "sha256":digest(sidecar)}])
        committed += 1
    checked = (state > 0).sum(axis=1)
    passed = (state == 1).sum(axis=1)
    assert np.all((checked == 0) | (checked == a)), "A centre has incomplete angle coverage"
    progress = read_json(directory / "stopped_by_user_20260909.json")["last_progress"]
    assert int(checked.sum()) >= int(progress["poses_checked"])
    if int(checked.sum()) == int(progress["poses_checked"]):
        assert int(passed.sum()) == int(progress["geometry_valid"])
    rate = np.divide(100 * passed, checked, out=np.full(n, np.nan), where=checked > 0)
    centres = centres.assign(checked_angles=checked, passed_angles=passed,
                             rejected_angles=checked-passed, passed_percent=rate)
    for code in range(2, 6):
        centres[f"failure_code_{code}_angles"] = (state == code).sum(axis=1)
    surface = loadmat(directory / "surface.mat", variable_names=["nodes_mm", "boundary_faces"])
    nodes = surface["nodes_mm"]
    faces = surface["boundary_faces"].astype(np.int64) - 1
    assert nodes.shape == (77562, 3) and faces.shape == (155120, 3)
    assert faces.min() >= 0 and faces.max() < len(nodes)
    counts = {LABELS[k]:int((state == k).sum()) for k in range(6)}
    representative = np.flatnonzero(passed == passed.max())
    representative = int(representative[np.argmin(
        centres.u_mm.to_numpy()[representative]**2 + centres.v_mm.to_numpy()[representative]**2)])
    summary = {"kind":"partial_geometry_diagnostic_not_CEM_experiment_error",
               "result_tag":physics["result_tag"], "scan_stopped_by_user":True,
               "snapshot_created_utc":datetime.now(timezone.utc).isoformat(),
               "scan_fingerprint":manifest["fingerprint"], "committed_blocks":committed,
               "centres_total":n, "centres_checked":int((checked > 0).sum()),
               "poses_total":n*a, "poses_checked":int(checked.sum()),
               "geometry_passed":int(passed.sum()), "geometry_rejected":int((checked-passed).sum()),
               "centres_all_angles_passed":int(((checked == a) & (passed == a)).sum()),
               "centres_all_angles_rejected":int(((checked == a) & (passed == 0)).sum()),
               "centres_mixed":int(((passed > 0) & (passed < checked)).sum()),
               "centres_unchecked":int((checked == 0).sum()), "status_counts":counts,
               "thresholds":physics["geometry_thresholds"],
               "recorded_failure_detail":"Only the first failing identifier per pose is stored; size, electrode and QC subcondition are unavailable.",
               "default_centre_index":representative+1, "ignored_uncommitted_files":ignored}
    return dict(directory=directory, centres=centres, state=state, u=u, v=v, phi=phi,
                nodes=nodes, faces=faces, inputs=inputs, summary=summary)

def crop_view(scan, u_min, u_max):
    """Restrict displayed centres only; keep the immutable full-scan provenance."""
    assert np.isfinite([u_min, u_max]).all() and u_min <= u_max
    chosen = (scan["u"] >= u_min) & (scan["u"] <= u_max)
    assert chosen.any(), "The requested u window contains no grid centres"
    keep = scan["centres"].u_mm.between(u_min, u_max).to_numpy()
    c = scan["centres"].loc[keep].reset_index(drop=True)
    s = scan["state"][keep]
    checked, passed = c.checked_angles.to_numpy(), c.passed_angles.to_numpy()
    n, a = s.shape
    summary = dict(scan["summary"])
    summary["source_scan_totals"] = {k: summary[k] for k in (
        "centres_total", "centres_checked", "poses_total", "poses_checked",
        "geometry_passed", "geometry_rejected", "centres_unchecked")}
    representative = np.flatnonzero(passed == passed.max())
    representative = int(representative[np.argmin(
        c.u_mm.to_numpy()[representative]**2 + c.v_mm.to_numpy()[representative]**2)])
    summary.update(view_u_mm=[float(scan["u"][chosen][0]), float(scan["u"][chosen][-1])],
        source_u_mm=[float(scan["u"][0]), float(scan["u"][-1])],
        display_crop_only=True, centres_total=n, centres_checked=int((checked > 0).sum()),
        poses_total=n*a, poses_checked=int(checked.sum()), geometry_passed=int(passed.sum()),
        geometry_rejected=int((checked-passed).sum()),
        centres_all_angles_passed=int(((checked == a) & (passed == a)).sum()),
        centres_all_angles_rejected=int(((checked == a) & (passed == 0)).sum()),
        centres_mixed=int(((passed > 0) & (passed < checked)).sum()),
        centres_unchecked=int((checked == 0).sum()),
        status_counts={LABELS[k]:int((s == k).sum()) for k in range(6)},
        default_centre_index=int(c.centre_index.iloc[representative]),
        default_view_index=representative,
        author_boundary_explanation={"date":"2026-09-09", "status":"author_interpretation",
            "text":"Автор связывает граничные отказы с попаданием электродов на руки или нижний срез STL. По сохранённому общему коду нельзя назначить эту причину каждому отказу."})
    return dict(scan, centres=c, state=s, u=scan["u"][chosen], summary=summary)


def encode(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes(order="C")).decode("ascii")

def build_report(scan, output):
    from plotly.offline import get_plotlyjs
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    c, s, summary = scan["centres"], scan["state"], scan["summary"]
    payload = dict(u=scan["u"].tolist(), v=scan["v"].tolist(), phi=scan["phi"].tolist(),
                   centre_ids=c.centre_index.tolist(), state=encode(s, "u1"), xyz=encode(c[["x_mm","y_mm","z_mm"]], "<f8"),
                   nodes=encode(scan["nodes"], "<f4"), faces=encode(scan["faces"], "<u4"),
                   colours=COLOURS, missing=MISSING, labels=LABELS, summary=summary)
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    document = TEMPLATE.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", data)
    (output / "geometry_map.html").write_text(document, encoding="utf-8")
    c.to_csv(output / "centres_geometry_summary.csv", index=False)
    angles = pd.DataFrame({"phi_deg":scan["phi"], "checked_centres":(s > 0).sum(axis=0),
                           "passed_centres":(s == 1).sum(axis=0), "rejected_centres":(s > 1).sum(axis=0)})
    angles["passed_percent"] = 100 * angles.passed_centres / angles.checked_centres
    angles.to_csv(output / "angles_geometry_summary.csv", index=False)
    bands = c[c.checked_angles > 0].copy()
    bands["u_band_start_mm"] = np.floor((bands.u_mm + 120) / 40).clip(upper=5) * 40 - 120
    bands["v_band_start_mm"] = np.floor((bands.v_mm + 120) / 20).clip(upper=11) * 20 - 120
    bands = bands.groupby(["u_band_start_mm","v_band_start_mm"]).agg(
        centres=("centre_index","size"),checked=("checked_angles","sum"),passed=("passed_angles","sum")).reset_index()
    bands["passed_percent"] = 100 * bands.passed / bands.checked
    bands.to_csv(output / "spatial_bands.csv", index=False)
    summary["angle_highest_pass_fraction_in_checked_region"] = angles.loc[angles.passed_percent.idxmax()].to_dict()
    summary["angle_lowest_pass_fraction_in_checked_region"] = angles.loc[angles.passed_percent.idxmin()].to_dict()
    (output / "geometry_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    write_png(scan, output / "geometry_matrix.png")
    provenance = {"complete_scan":False,"kind":summary["kind"],"inputs":scan["inputs"],
                  "builder_sha256":digest(__file__), "scan_fingerprint":summary["scan_fingerprint"],
                  "outputs":[{"path":p.name,"sha256":digest(p)} for p in output.iterdir()
                             if p.is_file() and p.name in ["geometry_map.html","geometry_matrix.png",
                                "centres_geometry_summary.csv","angles_geometry_summary.csv","spatial_bands.csv","geometry_summary.json"]]}
    (output / "geometry_report_manifest.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def write_png(scan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch
    c, s = scan["centres"], scan["state"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("geometry_pass", [x[1] for x in COLOURS])
    grid = (len(scan["v"]), len(scan["u"]))
    zero_angle = int(np.flatnonzero(scan["phi"] == 0)[0])
    values = [c.passed_percent.to_numpy(), np.where(s[:,zero_angle]>0,100*(s[:,zero_angle]==1),np.nan)]
    titles = ["Доля допустимых поворотов в каждом центре", "Один поворот: φ = 0°"]
    extent = [scan["u"][0]-.5, scan["u"][-1]+.5, scan["v"][0]-.5, scan["v"][-1]+.5]
    for ax, value, title in zip(axes,values,titles):
        ax.imshow(np.zeros(grid),origin="lower",extent=extent,
                  cmap=ListedColormap([MISSING]),interpolation="none",rasterized=True)
        im=ax.imshow(np.ma.masked_invalid(value.reshape(grid)),origin="lower",
                     extent=extent,cmap=cmap,vmin=0,vmax=100,interpolation="none",rasterized=True)
        ax.set(title=title,xlabel="u, мм",ylabel="v, мм",xticks=np.arange(scan["u"][0],scan["u"][-1]+1,40),yticks=np.arange(scan["v"][0],scan["v"][-1]+1,40))
    fig.colorbar(im,ax=axes,label="Доля прошедших геометрию, %",shrink=.75,pad=.015)
    axes[0].legend(handles=[Patch(facecolor=MISSING,label="Не проверено")],loc="upper left")
    summary=scan["summary"]
    fig.suptitle("Остановленный перебор: допустимость электродов на расчётной сетке\n"
                 f"{summary['centres_checked']:,} из {summary['centres_total']:,} центров · "
                 "оценка совпадения с экспериментом ещё не выполнялась",fontsize=13)
    fig.savefig(path,dpi=150)
    plt.close(fig)

TEMPLATE=r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Карта допустимости электродов — остановленный перебор</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#18252e;font:16px system-ui,sans-serif}
main{max-width:1580px;margin:auto;padding:22px}h1{font-size:25px;margin:0 0 10px}
p{line-height:1.5;margin:8px 0}h2{font-size:17px;margin:0;padding:14px 16px 0}
.notice{background:#fff4db;border-left:5px solid #c17c00;padding:12px 16px;margin:15px 0}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.stat{background:white;border:1px solid #dce2e8;padding:12px 18px;min-width:185px;border-radius:6px}.stat strong{display:block;font-size:25px}
.controls{position:sticky;top:0;z-index:9;background:#f5f7faf5;padding:10px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button,input{font:inherit;padding:7px 10px;border:1px solid #aebac4;border-radius:4px;background:white}
button{cursor:pointer}.panel{background:white;border:1px solid #dce2e8;border-radius:6px;overflow:hidden}
.overview{display:grid;grid-template-columns:1fr 1fr;gap:14px}.plot{height:610px}
.detail{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}#angle-strip{height:205px}
#point-info{padding:10px 16px;font-size:15px;line-height:1.7}.sub{font-size:14px;color:#465865}
#angle-rate{height:310px}.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;padding:10px 16px}.dot{display:inline-block;width:13px;height:13px;vertical-align:middle;margin-right:5px}
footer{font-size:13px;color:#596b77;margin-top:20px;line-height:1.6}.error{color:#a00016;font-weight:600}
@media(max-width:900px){.overview,.detail{grid-template-columns:1fr}.plot{height:530px}}
</style><script>__PLOTLY__</script></head><body><main>
<h1>Где удаётся разместить электроды на расчётной сетке</h1>
<p id="subtitle"></p>
<div class="notice"><b>Прогон остановлен по запросу автора.</b> Цвет показывает результат проверки геометрии электродов.
Импедансы и ошибки относительно эксперимента в этом переборе ещё не рассчитаны.</div>
<div class="stats" id="stats"></div>
<p><b>Зелёный:</b> проходит больше поворотов. <b>Красный:</b> больше отказов. <b>Серый:</b> участок ещё не проверен.
На слое отдельного угла зелёный означает «прошёл», красный — «отклонён».</p>
<div class="controls"><label>Показать <select id="layer"><option value="all">Долю допустимых углов из 41</option></select></label>
<button id="zoom">Область центров в 3D</button><button id="whole">Вся модель в 3D</button>
<span class="sub">Вращение — мышью; масштаб — колёсиком. Нажмите точку для разбора углов.</span></div>
<div id="overview" class="overview">
<section class="panel"><h2>3D-модель и проверяемые центры</h2><div id="body" class="plot"></div></section>
<section class="panel"><h2 id="map-title">Полная матрица центров u × v</h2><div id="matrix" class="plot"></div></section>
</div>
<div class="detail">
<section class="panel"><h2>Углы в выбранном центре</h2><div id="point-info"></div><div id="angle-strip"></div>
<div class="legend" id="code-legend"></div></section>
<section class="panel"><h2>Проходимость каждого угла в проверенной области</h2><div id="angle-rate"></div>
<p class="sub" style="padding:0 16px 12px">Знаменатель — только проверенные центры. Это описание текущего участка, а не выбор лучшего угла по экспериментальным импедансам.</p></section>
</div>
<div class="notice"><b>Как понимать отказы.</b> Для принятия позы должны пройти все четыре электрода во всех девяти размерах сборки.
«Отказ контроля геометрии» объединяет нарушения площади, смещения центра площадки и расстояния до поверхности.
В сохранённых блоках не указаны конкретный размер сборки, электрод и нарушенный подпорог.
«Не удалось набрать площадь» и «Нет свободной начальной грани» сохранены отдельно.</div>
<p class="sub">Карта показывает пригодность представления контактов на текущей сетке с принятыми допусками.
По объяснению автора, граничные отказы связаны с попаданием электродов на руки или нижний срез STL. Сохранённый общий код не позволяет назначить эту причину каждому отказу. Карта не устанавливает анатомическую невозможность размещения электродов. Прямой край серой области связан с порядком перебора центров.</p>
<footer id="footer"></footer><p id="error" class="error"></p>
<script id="geometry-data" type="application/json">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('geometry-data').textContent),Q=D.summary,N=D.u.length*D.v.length,A=D.phi.length;
function decode(t,T){const s=atob(t),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);return new T(b.buffer);}
const S=decode(D.state,Uint8Array),X=decode(D.xyz,Float64Array),M=decode(D.nodes,Float32Array),F=decode(D.faces,Uint32Array);
const checked=new Uint8Array(N),passed=new Uint8Array(N),fail=Array.from({length:6},()=>new Uint8Array(N));
for(let c=0;c<N;c++)for(let a=0;a<A;a++){const k=S[c*A+a];if(k){checked[c]++;if(k===1)passed[c]++;else fail[k][c]++;}}
const fmt=x=>x.toLocaleString('ru-RU'),select=document.getElementById('layer');
D.phi.forEach((p,i)=>select.add(new Option('Угол φ = '+p+'°',String(i))));
document.getElementById('subtitle').textContent='Окно u = ['+D.u[0]+'; '+D.u[D.u.length-1]+'] мм: '+D.u.length+' × '+D.v.length+' центров × '+A+' угол. Проверено центров в окне: '+fmt(Q.centres_checked)+'. Исходные '+fmt(Q.committed_blocks)+' блоков и результаты вне окна сохранены.';
const cards=[['Проверено центров',Q.centres_checked],['Все 41 угла проходят',Q.centres_all_angles_passed],['Все 41 угла отклонены',Q.centres_all_angles_rejected],['Часть углов проходит',Q.centres_mixed],['Ещё не проверено центров',Q.centres_unchecked]];
document.getElementById('stats').innerHTML=cards.map(p=>'<div class="stat"><span>'+p[0]+'</span><strong>'+fmt(p[1])+'</strong></div>').join('');
document.getElementById('footer').textContent='Локальный диагностический снимок '+Q.result_tag+'. Проверено поз: '+fmt(Q.poses_checked)+' / '+fmt(Q.poses_total)+'. Допустимы: '+fmt(Q.geometry_passed)+', отклонены: '+fmt(Q.geometry_rejected)+'. Все загруженные блоки проверены по SHA-256. Интерполяция отсутствует; данные не передаются на сервер.';
const codeColours=[D.missing,'#008b48','#ce2538','#ed9c1b','#7757b2','#54278f'];
document.getElementById('code-legend').innerHTML=D.labels.map((l,i)=>'<span><i class="dot" style="background:'+codeColours[i]+'"></i>'+l+'</span>').join('');
const axis=(data,k)=>Array.from({length:data.length/3},(_,i)=>data[i*3+k]);
const xyz=[axis(X,0),axis(X,1),axis(X,2)];
const mesh={type:'mesh3d',x:axis(M,0),y:axis(M,1),z:axis(M,2),i:axis(F,0),j:axis(F,1),k:axis(F,2),
color:'#b8c4cd',opacity:.20,flatshading:false,hoverinfo:'skip',showscale:false,showlegend:false};
function limits(data){return [0,1,2].map(k=>{let lo=Infinity,hi=-Infinity;for(let i=k;i<data.length;i+=3){lo=Math.min(lo,data[i]);hi=Math.max(hi,data[i]);}const p=(hi-lo)*.06+3;return [lo-p,hi+p];});}
const region=limits(X),whole=limits(M);
function scene(r){return {xaxis:{title:{text:'x, мм'},range:r[0]},yaxis:{title:{text:'y, мм'},range:r[1]},zaxis:{title:{text:'z, мм'},range:r[2]},aspectmode:'data',camera:{eye:{x:-1.6,y:.7,z:.7}}};}
const config={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:'png',scale:2}};
function hover(c,value){
 const u=D.u[c%D.u.length],v=D.v[Math.floor(c/D.u.length)],a=value==='all'?-1:Number(value);
 let t='Центр '+D.centre_ids[c]+'<br>u='+u+'; v='+v+' мм<br>';
 t+=a<0?(checked[c]?'Проходит '+passed[c]+' из '+checked[c]+' углов ('+(100*passed[c]/checked[c]).toFixed(1)+'%)':'Не проверено'):
 ('φ='+D.phi[a]+'°: '+D.labels[S[c*A+a]]);
 t+='<br>XYZ: '+xyz.map(arr=>arr[c].toFixed(2)).join(', ')+' мм';
 if(checked[c])t+='<br>Общий контроль: '+fail[2][c]+'; площадь не набрана: '+fail[3][c]+'; нет начальной грани: '+fail[4][c];
 return t;
}
function layer(value){
 const known={x:[],y:[],z:[],ids:[],col:[],text:[]},missing={x:[],y:[],z:[],ids:[],text:[]};
 const z=Array.from({length:D.v.length},()=>Array(D.u.length).fill(null)),background=z.map(row=>row.map(()=>0)),ids=z.map((row,r)=>row.map((_,c)=>r*D.u.length+c)),texts=z.map(row=>row.slice());
 for(let c=0;c<N;c++){
  const a=value==='all'?-1:Number(value),done=a<0?checked[c]>0:S[c*A+a]>0;
  const r=done?(a<0?100*passed[c]/checked[c]:(S[c*A+a]===1?100:0)):null;
  const g=done?known:missing,t=hover(c,value);g.x.push(X[3*c]);g.y.push(X[3*c+1]);g.z.push(X[3*c+2]);g.ids.push(c);g.text.push(t);if(done)g.col.push(r);
  const row=Math.floor(c/D.u.length),col=c%D.u.length;z[row][col]=r;texts[row][col]=t;
 }
 return {known,missing,z,background,ids,texts};
}
function points(g,coloured){return {type:'scatter3d',mode:'markers',x:g.x,y:g.y,z:g.z,customdata:g.ids,text:g.text,hovertemplate:'%{text}<extra></extra>',
name:coloured?'Проверено':'Не проверено',showlegend:false,marker:{size:coloured?2.7:1.6,opacity:coloured?1:.23,color:coloured?g.col:D.missing,cmin:0,cmax:100,
colorscale:D.colours,showscale:false}};}
function error(e){document.getElementById('error').textContent=String(e);console.error(e);}
let selected=Q.default_view_index;
function point(c){
 selected=c;let text='<b>Центр '+D.centre_ids[c]+'</b> · u='+D.u[c%D.u.length]+' мм, v='+D.v[Math.floor(c/D.u.length)]+' мм<br>';
 text+='XYZ: '+xyz.map(arr=>arr[c].toFixed(2)).join(', ')+' мм<br>';
 text+=checked[c]?'Проверено '+checked[c]+' углов; прошло <b>'+passed[c]+'</b>; отклонено '+(checked[c]-passed[c])+'.':'Этот центр ещё не проверен.';
 document.getElementById('point-info').innerHTML=text;
 const row=Array.from(S.slice(c*A,(c+1)*A)),labels=row.map((s,a)=>'φ='+D.phi[a]+'°: '+D.labels[s]);
 return Plotly.react('angle-strip',[{type:'heatmap',x:D.phi,y:['Результат'],z:[row],zmin:0,zmax:5,
 colorscale:codeColours.flatMap((c,i)=>[[Math.max(0,(i-.5)/5),c],[Math.min(1,(i+.5)/5),c]]),showscale:false,
 text:[labels],hovertemplate:'%{text}<extra></extra>',zsmooth:false}],{margin:{l:85,r:20,t:10,b:55},
 xaxis:{title:{text:'φ, °'},dtick:5},yaxis:{fixedrange:true}},config);
}
select.disabled=true;
async function start(){
 const L=layer('all');
 await Promise.all([
 Plotly.newPlot('body',[mesh,points(L.missing,false),points(L.known,true)],{margin:{l:0,r:0,t:5,b:0},scene:scene(whole),uirevision:'geometry'},config),
 Plotly.newPlot('matrix',[
 {type:'heatmap',x:D.u,y:D.v,z:L.background,colorscale:[[0,D.missing],[1,D.missing]],showscale:false,customdata:L.ids,text:L.texts,hovertemplate:'%{text}<extra></extra>',zsmooth:false},
 {type:'heatmap',x:D.u,y:D.v,z:L.z,zmin:0,zmax:100,colorscale:D.colours,customdata:L.ids,text:L.texts,hovertemplate:'%{text}<extra></extra>',hoverongaps:false,zsmooth:false,connectgaps:false,
 colorbar:{title:{text:'Проходит, %'},thickness:14}}],
 {margin:{l:62,r:80,t:8,b:60},xaxis:{title:{text:'u, мм'},dtick:40,range:[D.u[0]-.5,D.u[D.u.length-1]+.5],constrain:'domain'},yaxis:{title:{text:'v, мм'},dtick:40,range:[D.v[0]-.5,D.v[D.v.length-1]+.5],scaleanchor:'x',constrain:'domain'},uirevision:'grid'},config),
 point(selected),
 Plotly.newPlot('angle-rate',[{type:'bar',x:D.phi,y:D.phi.map((_,a)=>{let good=0,done=0;for(let c=0;c<N;c++){const v=S[c*A+a];if(v){done++;if(v===1)good++;}}return done?100*good/done:0;}),marker:{color:'#33748e'},hovertemplate:'φ=%{x}°<br>Проходит %{y:.1f}% центров<extra></extra>'}],
 {margin:{l:55,r:20,t:15,b:55},xaxis:{title:{text:'φ, °'},dtick:5},yaxis:{title:{text:'Проходит, %'},range:[0,100]}},config)
 ]);
 for(const id of ['body','matrix'])document.getElementById(id).on('plotly_click',e=>{const c=e.points[0].customdata;if(Number.isInteger(c)&&c>=0&&c<N)point(c).catch(error);});
 document.getElementById('angle-strip').on('plotly_click',e=>{const a=D.phi.indexOf(e.points[0].x);if(a>=0){select.value=String(a);select.dispatchEvent(new Event('change'));}});
 select.disabled=false;
}
select.addEventListener('change',async()=>{
 select.disabled=true;
 try{const L=layer(select.value);
 await Promise.all([
 Plotly.restyle('body',{x:[L.missing.x,L.known.x],y:[L.missing.y,L.known.y],z:[L.missing.z,L.known.z],customdata:[L.missing.ids,L.known.ids],
 text:[L.missing.text,L.known.text],'marker.color':[D.missing,L.known.col]},[1,2]),
 Plotly.restyle('matrix',{z:[L.background,L.z],text:[L.texts,L.texts]},[0,1])]);
 document.getElementById('map-title').textContent=select.value==='all'?'Доля допустимых поворотов в каждом центре':'Геометрия для φ = '+D.phi[Number(select.value)]+'°';
 }catch(e){error(e);}finally{select.disabled=false;}
});
document.getElementById('zoom').onclick=()=>Plotly.relayout('body',{scene:scene(region)});
document.getElementById('whole').onclick=()=>Plotly.relayout('body',{scene:scene(whole)});
start().catch(error);
</script></main></body></html>"""

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=DEFAULT)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--u-min",type=float,default=-80)
    p.add_argument("--u-max",type=float,default=80)
    args=p.parse_args()
    scan=crop_view(load_geometry(args.input_dir),args.u_min,args.u_max)
    summary=build_report(scan,args.output_dir or args.input_dir/"geometry_report")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
