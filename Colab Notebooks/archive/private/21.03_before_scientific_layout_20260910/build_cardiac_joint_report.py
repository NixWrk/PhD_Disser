"""Build joint LV/RV diagnostics and localize selected changes on existing CT masks."""
from pathlib import Path
import argparse,hashlib,html,inspect,json
from datetime import datetime,timezone
import nibabel as nib
import numpy as np
import nbformat
from scipy.ndimage import binary_erosion
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs
from cardiac_function_analysis import VENTRICLES,CHAMBERS
from cardiac_joint_dynamics import joint_dynamics,boundary_change
from build_cardiac_function_report import sha256,table,write_csv

TAG="cardiac-joint-dynamics"
PREFIX="heart_rr_analysis/function_analysis/joint_dynamics/"
REGIONS={"inferior":"Нижняя треть","middle":"Средняя треть","superior":"Верхняя треть"}


def render_pair(derived,pair,phase_inputs,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subject=pair["subject"];first=pair["first_phase_id"];second=pair["second_phase_id"]
    render_hash=hashlib.sha256((inspect.getsource(render_pair)+inspect.getsource(boundary_change)).encode()).hexdigest()
    fingerprints={}
    for ch in VENTRICLES:
        fingerprints[ch]={"render_sha256":render_hash,"first_ct_sha256":phase_inputs[(subject,first)]["ct_sha256"],
                          "second_ct_sha256":phase_inputs[(subject,second)]["ct_sha256"],
                          "first_mask_sha256":phase_inputs[(subject,first)]["mask_sha256"][ch],
                          "second_mask_sha256":phase_inputs[(subject,second)]["mask_sha256"][ch]}
    cases=[]
    for ch in VENTRICLES:
        case_dir=output/"boundaries"/f"{subject}_{first}_{second}_{ch}"
        meta=case_dir/"case.json"
        if not meta.exists():break
        cached=json.loads(meta.read_text(encoding="utf-8"))
        if cached.get("inputs")!=fingerprints[ch] or not (case_dir/"changes.png").exists() or sha256(case_dir/"changes.png")!=cached["image_sha256"]:break
        cases.append(cached)
    if len(cases)==2:return cases
    images=[nib.as_closest_canonical(nib.load(derived/subject/"phases"/f"{pid}.nii.gz")) for pid in [first,second]]
    aff=images[0].affine
    if images[0].shape!=images[1].shape or not np.allclose(aff,images[1].affine,atol=1e-5):
        raise ValueError("Temporal comparison requires matching physical grids; no silent resampling")
    if not np.allclose(aff[:3,:3],np.diag(np.diag(aff[:3,:3])),atol=1e-5):raise ValueError("Oblique grid needs a separate viewer")
    ct=[np.asarray(im.dataobj,dtype=np.float32) for im in images];spacing=np.diag(aff[:3,:3]);cases=[]
    for ch in VENTRICLES:
        masks=[]
        for pid in [first,second]:
            im=nib.as_closest_canonical(nib.load(derived/subject/"automatic_masks_v2"/pid/"heartchambers_highres"/f"{ch}.nii.gz"))
            if im.shape!=images[0].shape or not np.allclose(im.affine,aff,atol=1e-5):raise ValueError("Mask/CT grid mismatch")
            masks.append(np.asarray(im.dataobj)>0)
        coords=np.column_stack(np.nonzero(masks[0]|masks[1]));lo=np.maximum(coords.min(0)-8,0);hi=np.minimum(coords.max(0)+9,masks[0].shape)
        sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        a,b=[m[sl] for m in masks];cropped_aff=aff.copy();cropped_aff[:3,3]+=aff[:3,:3]@lo
        metrics,added,removed=boundary_change(a,b,cropped_aff);changed=added|removed
        expected=pair["lv_delta_ml" if ch==VENTRICLES[0] else "rv_delta_ml"]
        if not np.isclose(metrics["net_ml"],expected,atol=1e-5):raise ValueError("Spatial difference does not match geometry volume delta")
        indices=[int(np.argmax(changed.sum(axis=tuple(j for j in range(3) if j!=axis)))) if changed.any() else int(a.shape[axis]//2) for axis in range(3)]
        cts=[x[sl] for x in ct]
        fig,axes=plt.subplots(3,3,figsize=(12,11),facecolor="white")
        plane_info=[]
        for row,(axis,label) in enumerate([(2,"Аксиальная"),(1,"Корональная"),(0,"Сагиттальная")]):
            idx=indices[axis];cut=[slice(None)]*3;cut[axis]=idx;cut=tuple(cut)
            slice_a=a[cut].T[::-1,::-1];slice_b=b[cut].T[::-1,::-1]
            ad=added[cut].T[::-1,::-1];rm=removed[cut].T[::-1,::-1]
            rest=[j for j in range(3) if j!=axis];aspect=spacing[rest[1]]/spacing[rest[0]]
            for col in range(3):
                im=cts[0 if col==0 else 1][cut].T[::-1,::-1]
                axes[row,col].imshow(im,cmap="gray",vmin=-25,vmax=225,aspect=aspect,interpolation="nearest")
                if col==2:
                    overlay=np.zeros((*ad.shape,4),dtype=float);overlay[ad]=[0.05,.8,.95,.6];overlay[rm]=[1,.55,.02,.6]
                    axes[row,col].imshow(overlay,aspect=aspect,interpolation="nearest")
                else:
                    mask=slice_a if col==0 else slice_b
                    edge=mask&~binary_erosion(mask);overlay=np.zeros((*mask.shape,4));overlay[edge]=[1,.55,.02,1] if col==0 else [.05,.8,.95,1]
                    axes[row,col].imshow(overlay,aspect=aspect,interpolation="nearest")
                title=(first[-2:]+" + контур") if col==0 else ((second[-2:]+" + контур") if col==1 else "Добавлено / удалено")
                axes[row,col].set_title(f"{label} · {title}",fontsize=10);axes[row,col].axis("off")
            plane_info.append({"axis":axis,"canonical_index":int(lo[axis]+idx),"world_coordinate_mm":float(aff[axis,3]+spacing[axis]*(lo[axis]+idx))})
        fig.suptitle(f"{subject.upper()} · {CHAMBERS[ch]} · {first[-2:]} → {second[-2:]} | L100/W250 HU\nГолубой: добавлено; оранжевый: удалено. Общая система координат КТ; без регистрации.",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94))
        target=output/"boundaries"/f"{subject}_{first}_{second}_{ch}";target.mkdir(parents=True,exist_ok=True)
        fig.savefig(target/"changes.png",dpi=115);plt.close(fig)
        case={"subject":subject,"cycle_index":pair["cycle_index"],"first_phase_id":first,"second_phase_id":second,"chamber":ch,
              "inputs":fingerprints[ch],**metrics,"planes":plane_info,"image_sha256":sha256(target/"changes.png"),
              "image":(target/"changes.png").relative_to(output).as_posix(),"page":(target/"index.html").relative_to(output).as_posix(),
              "selection_reasons":pair["selection_reasons"],"visual_review_status":"not_independently_validated"}
        zone_rows=[]
        for region in REGIONS:
            ad=next(x for x in metrics["zones"] if x["region"]==region and x["change"]=="added")
            rm=next(x for x in metrics["zones"] if x["region"]==region and x["change"]=="removed")
            zone_rows.append({"region":REGIONS[region],"added":ad["volume_ml"],"removed":rm["volume_ml"],"net":ad["volume_ml"]-rm["volume_ml"]})
        body=f'<h1>{subject.upper()} · {CHAMBERS[ch]} · фазы {first[-2:]} → {second[-2:]}</h1><p>Полный объём пространственной разности измерен на общей сетке КТ. Ни одна из двух масок не является эталоном. Изменения включают движение, деформацию и вариабельность сегментации; эти вклады здесь не разделены.</p>'
        body+=table([metrics],[("added_ml","Добавлено, мл"),("removed_ml","Удалено, мл"),("net_ml","Итоговое ΔV, мл"),("symmetric_difference_ml","Добавлено + удалено, мл"),("centroid_displacement_mm","Смещение центра, мм")])
        body+='<h2>Распределение изменений по высоте</h2><p>Три равных диапазона координаты «снизу вверх» в объединении двух масок. Эти геометрические области не являются разметкой основания, клапанов или выходного тракта.</p>'+table(zone_rows,[("region","Область"),("added","Добавлено, мл"),("removed","Удалено, мл"),("net","ΔV, мл")])
        body+='<h2>Срезы с наибольшей площадью различий</h2><p>В каждой плоскости выбран свой срез по максимальному числу различающихся вокселей. Окно 100/250 HU; контрастные различия между исследованиями сохраняются.</p><img src="changes.png" style="max-width:100%;height:auto" alt="КТ двух фаз и пространственные разности масок">'
        (target/"index.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Изменения границ</title><style>body{max-width:1150px;margin:24px auto;padding:15px;font:16px system-ui;line-height:1.55}.function-table{border-collapse:collapse;width:100%}.function-table td,.function-table th{border:1px solid #bbb;padding:8px}</style><p><a href="../../index.html">Совместная динамика ЛЖ и ПЖ</a></p>'+body+'</html>',encoding="utf-8")
        (target/"case.json").write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding="utf-8");cases.append(case)
        print(f"Boundary comparison: {subject} {first}/{second} {CHAMBERS[ch]}",flush=True)
    return cases


def figures(data):
    out=[]
    colors={VENTRICLES[0]:"#2864cb",VENTRICLES[1]:"#d84c39"}
    specs=[("observed_range_fraction","Объём, нормированный по наблюдаемому размаху","Доля размаха"),
           ("volume_area_length_mm","Отношение объёма к площади","V/A, мм"),
           ("elongation","Удлинённость эквивалентного эллипсоида","Безразмерная"),
           ("relative_shape_area","Площадь с поправкой на равномерный масштаб","Относительная площадь"),
           ("volume_rate_ml_s","Средняя скорость изменения объёма","мл/с"),
           ("fraction_rate_s","Средняя скорость нормированного изменения","1/с")]
    for subject in sorted({r["subject"] for r in data["points"]}):
        fig=make_subplots(rows=3,cols=2,subplot_titles=[r[1] for r in specs],vertical_spacing=.11,horizontal_spacing=.13)
        for cycle in sorted({r["cycle_index"] for r in data["points"] if r["subject"]==subject}):
            for ch in VENTRICLES:
                pts=[r for r in data["points"] if r["subject"]==subject and r["cycle_index"]==cycle and r["chamber"]==ch]
                seg=[r for r in data["intervals"] if r["subject"]==subject and r["cycle_index"]==cycle and r["chamber"]==ch]
                label=f"{CHAMBERS[ch]}, R–R {cycle+1}";group=f"{ch}_{cycle}"
                for i,(key,title,unit) in enumerate(specs):
                    row=i//2+1;col=i%2+1
                    if i<4:
                        x=[r["derived_time_from_R0_ms"] for r in pts];y=[r[key] for r in pts]
                        text=[f'Фаза {r["phase_id"][-2:]} · {r["phase_percent_within_cycle"]:.0f}% R–R · {len(pts)} кадров в интервале' for r in pts]
                        trace=go.Scatter(x=x,y=y,mode="lines+markers",text=text,name=label,legendgroup=group,showlegend=i==0,
                                         line={"color":colors[ch],"dash":"solid" if cycle==0 else "dot"},marker={"symbol":"circle" if cycle==0 else "diamond","size":7},
                                         hovertemplate="%{text}<br>t=%{x:.1f} мс<br>Значение=%{y:.4g}<extra>%{fullData.name}</extra>")
                    else:
                        trace=go.Scatter(x=[r["midpoint_ms"] for r in seg],y=[r[key] for r in seg],mode="markers",name=label,legendgroup=group,showlegend=False,
                                         marker={"color":colors[ch],"symbol":"circle" if cycle==0 else "diamond","size":7},
                                         error_x={"type":"data","array":[r["dt_ms"]/2 for r in seg],"visible":True,"thickness":1},
                                         text=[f'{r["first_phase_id"][-2:]} → {r["second_phase_id"][-2:]} · Δt={r["dt_ms"]:.1f} мс' for r in seg],
                                         hovertemplate="%{text}<br>Средняя скорость=%{y:.3f}<extra>%{fullData.name}</extra>")
                    fig.add_trace(trace,row=row,col=col);fig.update_yaxes(title_text=unit,row=row,col=col)
        fig.update_xaxes(title_text="Время от первой R-метки, мс")
        fig.update_layout(title=f"{subject.upper()}: объём, геометрия и средние межкадровые скорости",height=1050,template="plotly_white",
                          legend={"orientation":"h","y":1.06,"groupclick":"togglegroup"},margin={"l":75,"r":35,"t":110,"b":50})
        out.append(fig)
    return out


INTRO=r"""## Совместная динамика объёма и формы ЛЖ и ПЖ

Работа продолжается на исходных масках TotalSegmentator. По последнему решению
автора новые ручные маски не ожидаются из-за недостаточного контраста.
Цель этого раздела — определить участки расхождения двух желудочковых кривых
и локализовать изменения масок, существенные для УО/ФВ. Полученные различия
имеют диагностический статус; их нельзя разделить на истинное движение
и ошибку сегментации только по этим данным.

Для каждой камеры отдельно внутри каждого доступного R–R вычисляются

<div style="display:flex;flex-wrap:wrap;gap:32px;margin:20px 0;font-size:20px">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>V</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>−</mo><msub><mi>V</mi><mtext>min</mtext></msub></mrow><mrow><msub><mi>V</mi><mtext>max</mtext></msub><mo>−</mo><msub><mi>V</mi><mtext>min</mtext></msub></mrow></mfrac></mrow></math>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>L</mi><mtext>VA</mtext></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mn>1000</mn><mi>V</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mrow><mi>A</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mfrac></mrow></math>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>Q</mi><mi>A</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>A</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>/</mo><mi>A</mi><mo>(</mo><msub><mi>t</mi><mn>0</mn></msub><mo>)</mo></mrow><msup><mrow><mo>[</mo><mi>V</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>/</mo><mi>V</mi><mo>(</mo><msub><mi>t</mi><mn>0</mn></msub><mo>)</mo><mo>]</mo></mrow><mfrac><mn>2</mn><mn>3</mn></mfrac></msup></mfrac></mrow></math>
</div>

Здесь V — объём исходной маски в мл, A — её площадь в мм², минимум и максимум
берутся только среди кадров данного R–R; R — доля наблюдаемого размаха,
<i>L</i><sub>VA</sub> — геометрическая длина в мм, t₀ — первый доступный кадр интервала.
Отношение V/A не является толщиной миокарда. <i>Q</i><sub>A</sub> — безразмерное изменение
площади после учёта равномерного масштабирования объёма: при изотропном
увеличении одной формы оно равно единице. Этот показатель не измеряет
деформацию ткани и не выявляет источник изменения границы.

Средняя межкадровая скорость определяется как ΔV/Δt или ΔR/Δt. На графиках
она показана в середине интервала с горизонтальным отрезком между кадрами;
это положение маркера не означает мгновенного измерения скорости.
При нулевом размахе нормировка не определена. Нормировка короткого фрагмента
также растягивает его минимум и максимум
до 0 и 1. Поэтому для ранжирования нормированных различий применено явное
техническое правило: не менее четырёх кадров и ненулевой размах. Это не
критерий достаточности фаз для вычисления УО/ФВ.
"""

LIMITS=r"""### Интерпретация изменений без ручного референса

Постоянная добавка к объёму сохраняет разность двух фаз; постоянный множитель
сохраняет нормированную кривую и отношение размаха к максимуму. Различие
нормированных кривых показывает, что одной постоянной добавки или масштаба
недостаточно для их совмещения. Оно не доказывает ошибку конкретного желудочка:
две камеры не являются взаимными референсами, а механика и покрытие фаз могут
различаться. Корреляция V и A также служит только описанием геометрии.

В пространственном сравнении «добавлено» означает воксели второй маски,
отсутствующие в первой, «удалено» — обратную разность. Итоговое ΔV равно
разности этих двух объёмов; при почти неизменном V обе области могут быть
значительными. Сравнение выполнено в координатах сканера без регистрации,
поэтому оно включает перемещение сердца. Разбиение по высоте на три равные
геометрические части помогает локализовать изменения, но не выделяет
клапанные плоскости, основание желудочка или выходной тракт.

Малое ΔV при изменении геометрии рассматривается как кандидат для разбора.
Метки изоволюметрического сокращения/расслабления по этому правилу не
назначаются. Записанное временное разрешение составляет около 140 мс;
более частые реконструированные кадры и гладкие линии не создают независимых
измерений с более высоким разрешением. Точные ED/ES и независимая точность
УО/ФВ остаются нерешёнными. Новых поправок к объёмам этот раздел не вводит.
"""



def calculated_findings(data,cases):
    def f(x):return f"{x:.2f}".replace(".",",")
    parts=["<h3>Наблюдения по рассчитанным маскам</h3>"]
    for subject in sorted({r["subject"] for r in data["comparisons"]}):
        r=max((r for r in data["comparisons"] if r["subject"]==subject),key=lambda r:abs(r["rate_gap_ml_s"]))
        c=next(c for c in cases if c["subject"]==subject and c["first_phase_id"]==r["first_phase_id"] and c["second_phase_id"]==r["second_phase_id"] and c["chamber"]==VENTRICLES[1])
        shares=[100*sum(z["volume_ml"] for z in c["zones"] if z["region"]==region)/c["symmetric_difference_ml"] if c["symmetric_difference_ml"] else 0 for region in REGIONS]
        parts.append(f'<p><b>{subject.upper()}.</b> Наибольшее абсолютное расхождение средних скоростей находится между фазами {r["first_phase_id"][-2:]} и {r["second_phase_id"][-2:]}: ΔV ЛЖ = {f(r["lv_delta_ml"])} мл, ΔV ПЖ = {f(r["rv_delta_ml"])} мл за {f(r["dt_ms"])} мс. В пространственной разности ПЖ доли нижней, средней и верхней трети составляют соответственно {f(shares[0])}%, {f(shares[1])}% и {f(shares[2])}%. Это распределение изменений автоматической маски в координатах сканера.</p>')
    smallest=min((r for r in data["intervals"] if r["chamber"]==VENTRICLES[0]),key=lambda r:abs(r["delta_volume_pct"]))
    c=next(c for c in cases if c["subject"]==smallest["subject"] and c["first_phase_id"]==smallest["first_phase_id"] and c["second_phase_id"]==smallest["second_phase_id"] and c["chamber"]==VENTRICLES[0])
    parts.append(f'<p><b>Малое изменение общего объёма.</b> У {c["subject"].upper()} между фазами {c["first_phase_id"][-2:]} и {c["second_phase_id"][-2:]} ΔV ЛЖ = {f(c["net_ml"])} мл. Одновременно добавлено {f(c["added_ml"])} мл и удалено {f(c["removed_ml"])} мл: их сумма {f(c["symmetric_difference_ml"])} мл при указанном итоговом изменении объёма. Удлинённость меняется на {f(smallest["delta_elongation_pct"])}%, центр — на {f(c["centroid_displacement_mm"])} мм. Это наблюдение по маскам; изоволюметрическая фаза и достоверность небольших изменений формы не установлены.</p>')
    corr=[r["volume_area_correlation"] for r in data["coverage"] if r["normalization_rank_eligible"] and r["volume_area_correlation"] is not None]
    if corr:parts.append(f'<p><b>Объём и площадь.</b> В интервалах с не менее чем четырьмя кадрами описательная корреляция V и A составляет {min(corr):.3f}–{max(corr):.3f}. Она рассчитана по зависимым фазам без проверки статистической значимости и не позволяет выбрать между изменением полости и предполагаемой оболочечной ошибкой.</p>')
    opposite=[r for r in data["comparisons"] if r["opposite_volume_directions"]]
    if opposite:
        labels=", ".join(f'{r["subject"].upper()} {r["first_phase_id"][-2:]}→{r["second_phase_id"][-2:]}' for r in opposite)
        parts.append(f'<p><b>Противоположные знаки ΔV.</b> Они встречаются на интервалах {labels}. Знак здесь определён арифметически, без установленного порога ошибки; он не доказывает физиологическую противофазность желудочков.</p>')
    return "".join(parts)

def build_joint_report(rows,inputs,derived,work):
    output=Path(work)/"joint_dynamics";output.mkdir(exist_ok=True)
    data=joint_dynamics(rows)
    for name,records in data.items():write_csv(output/(name+".csv"),records)
    phase_inputs={(r["subject"],r["phase_id"]):r for r in inputs["phases"]}
    cases=[]
    for pair in data["selected"]:cases.extend(render_pair(Path(derived),pair,phase_inputs,output))
    zones=[{"subject":c["subject"],"chamber":c["chamber"],"first_phase_id":c["first_phase_id"],"second_phase_id":c["second_phase_id"],**z} for c in cases for z in c["zones"]]
    write_csv(output/"boundary_zones.csv",zones)
    flat=[{k:v for k,v in c.items() if k not in ["zones","inputs","planes","selection_reasons"]} for c in cases];write_csv(output/"boundary_summary.csv",flat)
    cov=[]
    for subject,cycle in sorted({(r["subject"],r["cycle_index"]) for r in data["coverage"]}):
        lv,rv=[next(r for r in data["coverage"] if r["subject"]==subject and r["cycle_index"]==cycle and r["chamber"]==ch) for ch in VENTRICLES]
        cov.append({"subject":subject.upper(),"cycle":str(cycle+1),"count":str(lv["phase_count"]),"coverage":f'{lv["phase_start_pct"]:.0f}–{lv["phase_end_pct"]:.0f}%',
                    "lv_range":lv["range_ml"],"rv_range":rv["range_ml"],"eligible":"да" if lv["normalization_rank_eligible"] else "нет: короткий фрагмент"})
    cover_html=table(cov,[("subject","Исследование"),("cycle","R–R"),("count","Кадры"),("coverage","Доступные фазы"),("lv_range","Размах ЛЖ, мл"),("rv_range","Размах ПЖ, мл"),("eligible","Сравнение нормированной динамики")])
    selected_rows=[]
    for r in sorted(data["selected"],key=lambda r:(r["subject"],r["start_ms"])):
        selected_rows.append({**r,"subject":r["subject"].upper(),"phase_pair":r["first_phase_id"][-2:]+" → "+r["second_phase_id"][-2:],"cycle":str(r["cycle_index"]+1),
                              "fraction_rate_gap_s":r["fraction_rate_gap_s"] if r["normalization_rank_eligible"] else None})
    selected_html=table(selected_rows,[("subject","Исследование"),("cycle","R–R"),("phase_pair","Фазы"),("dt_ms","Δt, мс"),("lv_delta_ml","ΔV ЛЖ, мл"),("rv_delta_ml","ΔV ПЖ, мл"),("rate_gap_ml_s","ПЖ − ЛЖ, мл/с"),("fraction_rate_gap_s","Δ скорости R, 1/с")])
    low=[{**r,"subject":r["subject"].upper(),"chamber":CHAMBERS[r["chamber"]],"pair":r["first_phase_id"][-2:]+" → "+r["second_phase_id"][-2:]} for r in data["low_volume_change"]]
    low_html=table(low,[("subject","Исследование"),("chamber","Камера"),("pair","Фазы"),("delta_volume_pct","ΔV, %"),("delta_area_pct","ΔA, %"),("shape_area_change_pct","Изменение площади сверх масштаба, %"),("delta_elongation_pct","Δ удлинённости, %"),("centroid_displacement_mm","Смещение центра, мм")])
    boundary_html='<div style="overflow-x:auto"><table class="function-table"><thead><tr><th>Исследование / фазы</th><th>Камера</th><th>Добавлено, мл</th><th>Удалено, мл</th><th>Итоговое ΔV, мл</th><th>Распределение по высоте</th><th>Срезы</th></tr></thead><tbody>'
    for c in cases:
        totals={region:sum(z["volume_ml"] for z in c["zones"] if z["region"]==region) for region in REGIONS}
        dominant=max(totals,key=totals.get);share=100*totals[dominant]/c["symmetric_difference_ml"] if c["symmetric_difference_ml"] else 0
        boundary_html+=f'<tr><td>{c["subject"].upper()} / {c["first_phase_id"][-2:]} → {c["second_phase_id"][-2:]}</td><td>{CHAMBERS[c["chamber"]]}</td><td>{c["added_ml"]:.2f}</td><td>{c["removed_ml"]:.2f}</td><td>{c["net_ml"]:.2f}</td><td>{REGIONS[dominant]}, {share:.1f}% суммы различий</td><td><a href="{PREFIX}{c["page"]}">КТ и разности</a></td></tr>'
    boundary_html+='</tbody></table></div>'
    fig_html=''.join(to_html(f,full_html=False,include_plotlyjs=False,config={"responsive":True,"displaylogo":False}) for f in figures(data))
    selected_explanation=f'<h3>Выбранные участки расхождения</h3><p>Отобраны максимум абсолютной разности средних скоростей для каждого исследования, максимум нормированной разности в каждом подходящем R–R и участок с минимальным относительным изменением объёма ЛЖ в каждом исследовании. Повторы объединены: {len(data["selected"])} интервалов. Это очередь разбора, без порога «патологичности» или доказанной ошибки.</p>'
    body='<style>.function-table{border-collapse:collapse;width:100%;font-size:13px}.function-table th,.function-table td{border:1px solid #ccd4de;padding:7px;text-align:left}.function-table th{background:#edf3f8}</style><h3>Покрытие каждого интервала R–R</h3>'+cover_html+calculated_findings(data,cases)+selected_explanation+selected_html+fig_html
    body+='<h3>Участки с наименьшим относительным изменением объёма</h3><p>Показаны два наименьших |ΔV/V первого кадра| для каждой камеры в каждом исследовании. Место в этом списке не означает постоянного объёма или установленной механической фазы.</p>'+low_html
    body+='<h3>Где меняется пространственная маска</h3><p>Таблица охватывает обе камеры на выбранных интервалах. Сумма добавленного и удалённого объёма описывает пространственное различие, а их разность — изменение общего объёма.</p>'+boundary_html
    body+=f'<p>Исходные таблицы: <a href="{PREFIX}points.csv">геометрия по фазам</a> · <a href="{PREFIX}intervals.csv">межкадровые изменения</a> · <a href="{PREFIX}comparisons.csv">сравнение ЛЖ/ПЖ</a> · <a href="{PREFIX}boundary_zones.csv">распределение разностей</a> · <a href="{PREFIX}manifest.json">происхождение и ограничения</a>.</p>'
    cells=[nbformat.v4.new_markdown_cell(INTRO),nbformat.v4.new_code_cell("# Совместная диагностика исходных масок; расчёт выполняет build_cardiac_joint_report.py.",outputs=[nbformat.v4.new_output("display_data",data={"text/html":body},metadata={})]),nbformat.v4.new_markdown_cell(LIMITS)]
    for cell in cells:cell.metadata["tags"]=[TAG]
    from nbconvert import HTMLExporter
    standalone=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(INTRO),nbformat.v4.new_code_cell("# Reader",outputs=[nbformat.v4.new_output("display_data",data={"text/html":"<script>"+get_plotlyjs()+"</script>"+body.replace(PREFIX,"")},metadata={})]),nbformat.v4.new_markdown_cell(LIMITS)])
    exported,_=HTMLExporter(exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(standalone)
    (output/"index.html").write_text(exported,encoding="utf-8")
    manifest={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"status":"exploratory_hypothesis_not_validated",
              "independent_reference_available":False,"manual_reference_expected":False,"primary_masks_changed":False,"sv_ef_corrections_applied":False,
              "inputs":inputs,"code_sha256":{name:sha256(Path(__file__).with_name(name)) for name in ["cardiac_joint_dynamics.py","build_cardiac_joint_report.py"]},
              "counts":{k:len(v) for k,v in data.items()},"boundary_cases":cases,
              "selection_rules":["largest absolute LV/RV secant rate gap per subject","largest observed-range normalized rate gap per RR with at least 4 frames and nonzero volume range","smallest absolute relative LV volume change per subject"],
              "normalization":"independently by observed within-RR min/max; not a complete-cycle normalization; constant volume yields null",
              "rates":"secants on acquired intervals; no crossing of RR boundaries; no instantaneous derivative or temporal-resolution improvement",
              "boundary_comparison":"scanner coordinates without registration; full-voxel differences; selected maximal-difference orthogonal slices",
              "limits":["motion deformation reconstruction and segmentation variation are not separated","geometric superior/inferior bands do not identify valves or RVOT","no fitted correction or Wiggers labels"],
              "output_sha256":{f.name:sha256(f) for f in output.glob("*.csv")},"reader_html_sha256":sha256(output/"index.html")}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Joint dynamics built:",len(data["comparisons"]),"paired intervals;",len(cases),"spatial comparisons",flush=True)
    return cells,manifest


if __name__=="__main__":
    from build_cardiac_function_report import collect_geometry
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--derived-root",type=Path,required=True);a=p.parse_args()
    root=Path(__file__).parent;work=root/"heart_rr_analysis/function_analysis"
    rows,inputs=collect_geometry(a.derived_root,root/"heart_rr_analysis/chamber_volumes_by_rr.csv",work,["adam","nix","georg"])
    build_joint_report(rows,inputs,a.derived_root,work)
