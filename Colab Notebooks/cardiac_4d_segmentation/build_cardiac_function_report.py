"""Build the ventricular-function section of notebook 21.03 from native masks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import nibabel as nib
import nbformat
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
from cardiac_function_analysis import CHAMBERS, VENTRICLES, geometry, verify_grid, estimates, changes
import cardiac_notebook_text as narrative
from cardiac_reference_indexing import build_reference_section

TAG = "cardiac-function-analysis"
NOTEBOOK_NAME = "21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb"

INTRO = r"""## Условные оценки ударного объёма и фракции выброса

УО и ФВ вычисляются из конечного диастолического и конечного систолического
объёмов. Для рассматриваемых масок соответствие выбранных экстремумов
этим состояниям остаётся предположением:

<div class="scientific-equations">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>SV</mi><mo>=</mo><mi>EDV</mi><mo>−</mo><mi>ESV</mi><mo>,</mo><mspace width="2em"/><mi>EF</mi><mo>=</mo><mn>100</mn><mfrac><mrow><mi>EDV</mi><mo>−</mo><mi>ESV</mi></mrow><mi>EDV</mi></mfrac><mo>,</mo></mrow></math>
</div>

где SV — УО в мл; EF — ФВ в процентах; EDV и ESV — кандидатные оценки
конечного диастолического и конечного систолического объёмов в мл.
В таблице показаны два разных основания для выбора этих значений.

**Экстремумы доступных масок** определяются среди всех реконструированных
кадров исследования. Их разность — наблюдаемый размах, а отношение размаха
к максимуму — относительный размах. Если экстремумы принадлежат разным
сокращениям или конечные состояния не попали в реконструкцию,
интерпретация этих величин как УО и ФВ не обоснована.

**Условный полный цикл** получается из периодической модели с двумя
гармониками и регуляризованными аддитивными поправками для отдельных
сокращений. Поправки центрированы относительно нуля. Модельный максимум
и минимум дают условные EDV, ESV, УО и ФВ при допущении сопоставимости
сокращений. Условия периодичности следуют из базиса Фурье; точки между
кадрами являются значениями аппроксимации. Подробности временной привязки
и первоначального выбора модели сохранены в приложении А.

У Adam и Nix модельное объединение удовлетворяет рабочему порогу различия
ЧСС 10%. У Georg оно отклонено из-за коротко-длинной пары R–R.
Совпадение ЧСС в пределах порога само по себе не подтверждает равенство
объёмов соседних сокращений.

**Пояснение к таблице и рисунку.** Столбцы EDV/ESV содержат выбранные
экстремумы, а обозначения «УО / размах» и «ФВ / относительный размах»
следует читать согласно указанному способу оценки. Порядок ED → ES
обозначает максимум и минимум, а не обязательно хронологический порядок
кадров. На рисунке точки принадлежат реальным реконструкциям, линии —
периодической модели, а метки ED? и ES? обозначают модельных кандидатов.
Отсутствие линии у Georg соответствует отклонению объединённой модели.
Номера кадров отсчитываются от 00; проценты обозначают положение внутри R–R.
"""
GEOMETRY_TEXT = r"""## Геометрические показатели исходных масок

Объём рассчитывается как число вокселей маски, умноженное на физический
объём вокселя; мм³ переводятся в мл делением на 1000. Пространственная сетка
сверена с исходной КТ и параметрами конвертации. Площадь поверхности
получена на исходной сетке без сглаживания и сокращения числа треугольников. Главные размеры
определены через вторые моменты маски как диаметры равномерно заполненного
эквивалентного эллипсоида; удлинённость — отношение наибольшего диаметра к
наименьшему. Она характеризует форму, но не измеряет продольную деформацию
миокарда. Перемещение центра сохранено отдельно от изменения формы.

Графики показывают исходные отсчёты в миллисекундах от первой R-метки.
Линии соединяют только точки одного R–R-интервала. Изменения площади,
удлинённости и объёма между кадрами сохранены в таблице диагностики; их
отношение к интервалу времени является средней межкадровой скоростью,
а не разрешённой во времени мгновенной производной. Клапанные события,
изоволюметрическое сокращение и расслабление автоматически не размечаются.

Для кадров, определяющих наблюдаемые экстремумы, ниже даны ссылки на
КТ-просмотр. Он показывает исходные маски на срезах и предназначен для
проверки границ; создание страницы просмотра не означает ручного принятия.

Предварительное визуальное наблюдение: на трёх ортогональных сечениях
кадров Nix `phase_03` и `phase_09` граница крови и миокарда местами различима
слабо, в том числе в узком окне 60/160 HU. Этот ограниченный просмотр
не обосновывает исправление всей камеры равномерной оболочкой. Маски
оставлены исходными; полный посрезовой разбор и клапанные плоскости
ещё требуют проверки. Временное разрешение реконструкций — 140 мс;
шаг модельной сетки 1% R–R не повышает это разрешение.

**Пояснение к рисунку.** Столбцы соответствуют исследованиям, строки —
объёму, площади и удлинённости; показаны все четыре камеры. Предсердия
сохранены как геометрический контекст, тогда как итоговые УО/ФВ относятся
к желудочкам. Ненулевые изменения формы между кадрами рассматриваются
совместно с объёмом, без автоматической классификации механических фаз.
"""
SCENARIO_TEXT = r"""## Влияние границ масок и сценарных поправок на УО и ФВ

Сравнение с пересечением камер и маски всего сердца показывает влияние
реальной операции над имеющимися масками. Это диагностический вариант:
расхождение двух автоматических моделей не устанавливает, какой контур
точнее. Остальные варианты изменяют только таблицу объёмов и не создают
исправленных анатомических масок.

Постоянное вычитание 10 мл, множители 0,90 и 1,10 и толщины 0,5 и 1,0 мм выбраны
как иллюстративные вычислительные сценарии. Они не являются измеренными
ошибками, анатомическими толщинами, рекомендуемыми поправками или границами
неопределённости. Их нельзя подбирать ради совпадения ЛЖ и ПЖ или нормы ФВ.
Для модели тонкой оболочки используется

<div class="scientific-equations">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>V</mi><mtext>сцен</mtext></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><msub><mi>V</mi><mtext>исх</mtext></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>−</mo><mfrac><mrow><mi>h</mi><mi>A</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mn>1000</mn></mfrac><mo>,</mo></mrow></math>
</div>

где V<sub>исх</sub> и V<sub>сцен</sub> — исходный и сценарный объёмы в мл,
h — заданная толщина в мм, A — площадь поверхности исходной маски в мм²,
t — время реконструированного кадра. Приближение
не учитывает кривизну, клапанные плоскости и локальные различия стенки.
Постоянная толщина создаёт меняющуюся по фазам поправку, если меняется площадь.
Вычитание постоянного объёма сохраняет размах; умножение всех объёмов на
постоянный коэффициент сохраняет отношение размаха к максимуму.

Для каждой строки показаны изменения УО в мл и ФВ в процентных пунктах
относительно исходных масок при том же способе оценки. Для Georg сценарии
относятся только к наблюдаемому размаху, поскольку полный цикл не восстановлен.

**Пояснение к таблице и рисунку.** В таблице указаны абсолютные значения
и отклонения от исходного варианта. Столбчатый график показывает отклонения
по УО и ФВ; нулевое отклонение означает инвариантность данного показателя
к конкретной операции, без утверждения о правильности анатомической границы.
"""
LIMITS = r"""## Обоснование точности и следующая проверка

Параметры периодической модели проверены отдельной сеткой: одна, две и три
гармоники при трёх коэффициентах регуляризации. Диапазон результатов этой
сетки показывает чувствительность к модели; он не является доверительным
интервалом. Для расчёта интервала неопределённости пока недостаточно сведений
об ошибках масок, временной реконструкции и воспроизводимости разметки.

Текущий практический приоритет — совместный анализ объёма, формы и границ
имеющихся масок, включая чувствительность УО/ФВ к выбору фаз. Новые ручные
маски не ожидаются из-за недостаточного контраста. Протоколы 21.05–21.07
сохраняют требования к независимой проверке, но такая проверка сейчас не
выполнена и не является условием продолжения диагностических расчётов.
Ни согласие моделей, ни правдоподобная форма кривой не определяют точность.

Правила включения папиллярных мышц, трабекул и выходных трактов, положение
клапанных плоскостей и выбранные ED/ES должны быть зафиксированы до сравнения
результатов. [Рекомендации SCMR по постобработке](https://events.scmr.org/wp-content/uploads/2024/08/Post-processing.pdf)
используются как источник методических определений; они относятся к МРТ и
не валидируют этот КТ-расчёт. Числа из текущих масок не используются для
клинической классификации или перехода к функциям сердца по ТТРКГ.
"""


def sha256(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, columns); writer.writeheader(); writer.writerows(rows)


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def read_numeric_csv(path):
    """Restore scalar types and absent values from saved analysis tables."""
    def scalar(value):
        if value in ("True", "False"): return value == "True"
        try: return json.loads(value)
        except (ValueError, TypeError): return value
    return [{key:scalar(value) for key,value in row.items() if value != ""} for row in load_csv(path)]


def collect_geometry(derived_root, base_csv, output_dir, subjects):
    base_rows = [r for r in load_csv(base_csv) if r["subject"] in subjects]
    inputs = {"base_table": sha256(base_csv), "geometry_code": sha256(Path(__file__).with_name("cardiac_function_analysis.py")), "phases": []}
    plan = []
    for subject in subjects:
        conversion_path = derived_root / subject / "phases/conversion_manifest.json"
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        for info in conversion["phases"]:
            pid = info["phase_id"]
            source = next(r for r in base_rows if r["subject"] == subject and r["phase_id"] == pid)
            ct = derived_root / subject / "phases" / info["nifti"]
            whole = derived_root / subject / "products_v2" / pid / "whole_heart.nii.gz"
            masks = {ch: derived_root / subject / "automatic_masks_v2" / pid / "heartchambers_highres" / f"{ch}.nii.gz" for ch in CHAMBERS}
            entry = {"subject": subject, "phase_id": pid, "conversion_sha256": sha256(conversion_path),
                     "ct_sha256": sha256(ct), "whole_sha256": sha256(whole),
                     "mask_sha256": {ch: sha256(p) for ch, p in masks.items()}}
            inputs["phases"].append(entry); plan.append((subject, info, source, ct, whole, masks))
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache = output_dir / "geometry_cache.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("input_fingerprint") == fingerprint:
            print("Validated geometry cache reused", flush=True)
            return cached["rows"], inputs
    rows = []
    numeric = ("cycle_index", "phase_percent_within_cycle", "rr_percent", "derived_time_from_R0_ms", "rr_interval_ms", "image_heart_rate_bpm", "temporal_resolution_ms")
    for subject, info, source, ct_path, whole_path, masks in plan:
        ct = nib.load(str(ct_path)); whole_image = nib.load(str(whole_path))
        verify_grid(ct, info); verify_grid(whole_image, info, ct)
        whole = np.asanyarray(whole_image.dataobj) > 0
        for chamber, path in masks.items():
            img = nib.load(str(path)); verify_grid(img, info, ct)
            raw = np.asanyarray(img.dataobj)
            metrics = geometry(raw, img.affine)
            if not np.isclose(metrics["volume_ml"], float(source[chamber + "_ml"]), rtol=1e-7, atol=1e-6):
                raise ValueError(f"Stale input table: {subject}/{info['phase_id']}/{chamber}")
            mask = raw > 0
            clip_ml = int(np.count_nonzero(mask & whole)) * abs(float(np.linalg.det(img.affine[:3, :3]))) / 1000
            row = {"subject": subject, "phase_id": info["phase_id"], "chamber": chamber,
                   **{key: float(source[key]) for key in numeric}, **metrics,
                   "clipped_volume_ml": clip_ml, "outside_whole_ml": metrics["volume_ml"] - clip_ml,
                   "units_basis": "source_CT_grid_and_conversion_spacing_mm", "mask_variant": "automatic_raw"}
            row["cycle_index"] = int(row["cycle_index"]); rows.append(row)
        print(f"Geometry: {subject}/{info['phase_id']} (4 chambers)", flush=True)
    cache.write_text(json.dumps({"input_fingerprint": fingerprint, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, inputs


def number(value):
    if value is None or value == "": return "—"
    if isinstance(value, (float, int)):
        value = 0.0 if round(value, 2) == 0 else value
        return f"{value:.2f}".replace(".", ",")
    return html.escape(str(value))


def table(rows, columns):
    return '<div style="overflow-x:auto"><table class="function-table"><thead><tr>' + ''.join('<th>'+html.escape(label)+'</th>' for _, label in columns) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+number(row.get(key))+'</td>' for key,_ in columns)+'</tr>' for row in rows) + '</tbody></table></div>'


def function_table(result):
    display=[]
    for row in result:
        if row.get("scenario") != "raw" or row["estimator"] not in ("observed_global_range", "periodic_composite"): continue
        observed=row["estimator"]=="observed_global_range"
        text="Экстремумы доступных масок" if observed else "Условный полный цикл"
        if "sv_ml" not in row: text += ": объединение отклонено"
        phase = (f"{row.get('ed_phase_id','—').removeprefix('phase_')} → {row.get('es_phase_id','—').removeprefix('phase_')}" if observed else
                 f"{number(row.get('ed_coordinate'))}% → {number(row.get('es_coordinate'))}%")
        cycles = f"{row.get('ed_cycle_index',0)+1} → {row.get('es_cycle_index',0)+1}" if observed else "композит"
        if "sv_ml" not in row: phase = cycles = "—"
        display.append({**row,"method_label":text,"selected_phases":phase,"selected_cycles":cycles})
    return table(display,[("subject","Испытуемый"),("chamber_label","Камера"),("method_label","Способ оценки"),("edv_ml","EDV, мл"),("esv_ml","ESV, мл"),("sv_ml","УО / размах, мл"),("ef_percent","ФВ / относительный размах, %"),("selected_phases","Выбранные фазы ED → ES"),("selected_cycles","Циклы ED → ES")])


def scenario_table(result):
    chosen=[]
    for row in result:
        method = "periodic_composite" if row.get("cycle_merge_allowed") else "observed_global_range"
        if row["estimator"] != method or "sv_ml" not in row: continue
        chosen.append({**row,"method_label":"Модельный цикл" if method=="periodic_composite" else "Доступные кадры"})
    return table(chosen,[("subject","Испытуемый"),("chamber_label","Камера"),("scenario_label","Вариант"),("method_label","Основание"),("edv_ml","EDV, мл"),("esv_ml","ESV, мл"),("sv_ml","УО / размах, мл"),("ef_percent","ФВ / относительный размах, %"),("delta_sv_ml","Изменение УО, мл"),("delta_ef_pp","Изменение ФВ, п.п.")])


def plots(rows, result, profiles, subjects):
    colours={"heart_ventricle_left":"#125bb5","heart_ventricle_right":"#d64b32","heart_atrium_left":"#49a6d8","heart_atrium_right":"#e8a13f"}
    fig=make_subplots(rows=3,cols=len(subjects),subplot_titles=[s.upper() for s in subjects],vertical_spacing=.10)
    for col, subject in enumerate(subjects,1):
        for chamber in CHAMBERS:
            selected=sorted([r for r in rows if r['subject']==subject and r['chamber']==chamber],key=lambda r:r['derived_time_from_R0_ms'])
            for cycle in sorted({r['cycle_index'] for r in selected}):
                part=[r for r in selected if r['cycle_index']==cycle]
                for row_index,field in [(1,'volume_ml'),(2,'surface_area_mm2'),(3,'elongation')]:
                    fig.add_trace(go.Scatter(x=[r['derived_time_from_R0_ms'] for r in part],y=[r[field] for r in part],mode='lines+markers',name=CHAMBERS[chamber],legendgroup=chamber,showlegend=col==1 and row_index==1 and cycle==0,line={'color':colours[chamber]},text=[r['phase_id'] for r in part],hovertemplate='%{text}<br>%{x:.1f} мс<br>%{y:.3f}<extra>%{fullData.name}</extra>'),row=row_index,col=col)
    for row,label in [(1,'Объём, мл'),(2,'Площадь, мм²'),(3,'Удлинённость')]:fig.update_yaxes(title_text=label,row=row,col=1)
    fig.update_xaxes(title_text='Время от первой R-метки, мс',row=3)
    fig.update_layout(height=980,title='Геометрия исходных масок по реконструированным кадрам',template='plotly_white')
    impact=make_subplots(rows=2,cols=len(subjects),subplot_titles=[s.upper() for s in subjects],vertical_spacing=.22)
    for col,subject in enumerate(subjects,1):
        for chamber in VENTRICLES:
            selected=[r for r in result if r['subject']==subject and r['chamber']==chamber and 'delta_sv_ml' in r and r.get('scenario')!='raw' and r['estimator']==('periodic_composite' if r.get('cycle_merge_allowed') else 'observed_global_range')]
            for row,field in [(1,'delta_sv_ml'),(2,'delta_ef_pp')]:
                impact.add_trace(go.Bar(x=[r['scenario_label'] for r in selected],y=[r[field] for r in selected],name=CHAMBERS[chamber],legendgroup=chamber,showlegend=col==1 and row==1,marker_color=colours[chamber]),row=row,col=col)
    impact.update_yaxes(title_text='Изменение УО, мл',row=1,col=1);impact.update_yaxes(title_text='Изменение ФВ, п.п.',row=2,col=1)
    impact.update_layout(height=900,title='Влияние сценариев: Georg — размах доступных кадров',template='plotly_white',barmode='group')
    profile_fig=make_subplots(rows=1,cols=len(subjects),subplot_titles=[s.upper() for s in subjects])
    for col,subject in enumerate(subjects,1):
        for chamber in VENTRICLES:
            samples=[r for r in rows if r['subject']==subject and r['chamber']==chamber]
            for cycle in sorted({r['cycle_index'] for r in samples}):
                part=[r for r in samples if r['cycle_index']==cycle]
                profile_fig.add_trace(go.Scatter(x=[r['phase_percent_within_cycle'] for r in part],y=[r['volume_ml'] for r in part],mode='markers',name=CHAMBERS[chamber]+f' · цикл {cycle+1}',showlegend=col==1,marker={'color':colours[chamber],'symbol':'circle' if cycle==0 else 'diamond','size':9},text=[r['phase_id'] for r in part],hovertemplate='%{text}<br>%{x}% R–R<br>%{y:.2f} мл<extra>%{fullData.name}</extra>'),row=1,col=col)
            curve=[r for r in profiles if r['subject']==subject and r['chamber']==chamber and r['scenario']=='raw']
            if curve:
                profile_fig.add_trace(go.Scatter(x=[r['phase_percent'] for r in curve],y=[r['volume_ml'] for r in curve],mode='lines',name=CHAMBERS[chamber]+' · модель',showlegend=col==1,line={'color':colours[chamber]}),row=1,col=col)
                e=next(r for r in result if r['subject']==subject and r['chamber']==chamber and r['scenario']=='raw' and r['estimator']=='periodic_composite')
                profile_fig.add_trace(go.Scatter(x=[e['ed_coordinate'],e['es_coordinate']],y=[e['edv_ml'],e['esv_ml']],mode='markers+text',text=['ED?','ES?'],textposition='top center',showlegend=False,marker={'color':colours[chamber],'size':12,'symbol':'x'}),row=1,col=col)
    profile_fig.update_xaxes(title_text='Фаза своего R–R, %');profile_fig.update_yaxes(title_text='Объём, мл',col=1)
    profile_fig.update_layout(height=530,title='Кандидатные ED/ES; Georg — точки разных циклов без объединения',template='plotly_white')
    return profile_fig,fig,impact


def sensitivity_table(result):
    rows=[]
    for subject in sorted({r['subject'] for r in result}):
        for chamber in VENTRICLES:
            selected=[r for r in result if r['subject']==subject and r['chamber']==chamber and r['estimator']=='model_sensitivity']
            if not selected: continue
            rows.append({'subject':subject,'chamber':CHAMBERS[chamber],
                         'sv_range':f"{number(min(r['sv_ml'] for r in selected))}–{number(max(r['sv_ml'] for r in selected))}",
                         'ef_range':f"{number(min(r['ef_percent'] for r in selected))}–{number(max(r['ef_percent'] for r in selected))}"})
    return table(rows,[('subject','Испытуемый'),('chamber','Камера'),('sv_range','УО на сетке моделей, мл'),('ef_range','ФВ на сетке моделей, %')])


def ventricular_difference_table(result):
    pairs=[]
    for subject in sorted({r['subject'] for r in result}):
        for estimator in ('observed_global_range','periodic_composite'):
            matched={r['chamber']:r for r in result if r['subject']==subject and r.get('scenario')=='raw' and r['estimator']==estimator and 'sv_ml' in r}
            if len(matched)!=2:continue
            lv=matched['heart_ventricle_left']['sv_ml'];rv=matched['heart_ventricle_right']['sv_ml']
            pairs.append({'subject':subject,'method':'Модельный цикл' if estimator=='periodic_composite' else 'Доступные кадры','lv':lv,'rv':rv,'difference':rv-lv})
    return '<h3>Расхождение ЛЖ и ПЖ</h3><p class="scientific-caption"><b>Таблица 2.</b> Разность желудочковых оценок при одинаковом способе расчёта.</p><p>Разность служит поводом проверить фазы и границы камер. Она не определяет причину расхождения и не используется для подгонки масок или принудительного равенства УО.</p>'+table(pairs,[('subject','Испытуемый'),('method','Основание'),('lv','ЛЖ: УО / размах, мл'),('rv','ПЖ: УО / размах, мл'),('difference','ПЖ − ЛЖ, мл')])


def refresh_function_report(derived_root, output_dir, subjects, presentation_only=False):
    from analyze_subtraction_and_rr import periodic_cycle_profile, assess_cycle_merge
    derived_root=Path(derived_root) if derived_root is not None else None
    output_dir=Path(output_dir);work=output_dir/'function_analysis';work.mkdir(exist_ok=True)
    if presentation_only:
        previous=json.loads((work/'function_analysis_manifest.json').read_text(encoding='utf-8'))
        for name,expected in previous['output_sha256'].items():
            if sha256(work/name)!=expected: raise ValueError(f"Saved numeric output changed: {name}")
        for name in ['cardiac_function_analysis.py','analyze_subtraction_and_rr.py']:
            if sha256(Path(__file__).with_name(name))!=previous['code_sha256'][name]:
                raise ValueError(f"Computation changed; presentation-only refresh refused: {name}")
        if sha256(output_dir/'chamber_volumes_by_rr.csv')!=previous['inputs']['base_table']:
            raise ValueError("Original volume table changed")
        cache=json.loads((work/'geometry_cache.json').read_text(encoding='utf-8'))
        inputs=previous['inputs']
        expected_fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        if cache['input_fingerprint']!=expected_fingerprint: raise ValueError("Geometry cache provenance differs")
        rows=cache['rows']
        if set(subjects)!={r['subject'] for r in rows}: raise ValueError("Saved subject set differs")
        result=read_numeric_csv(work/'ventricular_function_estimates.csv')
        profiles=read_numeric_csv(work/'volume_profiles.csv')
        review_count=previous['ct_review_phase_count']
        print("Presentation-only refresh: saved calculations verified; numeric tables are read-only",flush=True)
    else:
        rows,inputs=collect_geometry(derived_root,output_dir/'chamber_volumes_by_rr.csv',work,subjects)
        result,profiles=estimates(rows,periodic_cycle_profile,assess_cycle_merge);delta=changes(rows)
        write_csv(work/'chamber_geometry.csv',rows);write_csv(work/'ventricular_function_estimates.csv',result)
        write_csv(work/'volume_profiles.csv',profiles);write_csv(work/'geometry_changes.csv',delta)
        from build_cardiac_ct_review import build_review
        review_count=build_review(derived_root,output_dir,result)
    figures=plots(rows,result,profiles,subjects)
    figure_html=[to_html(f,full_html=False,include_plotlyjs=(True if i==0 else False),config={'responsive':True,'displaylogo':False}) for i,f in enumerate(figures)]
    style='<style>.function-table{border-collapse:collapse;font-size:13px;width:100%}.function-table td,.function-table th{border:1px solid #cad1db;padding:7px;text-align:left}.function-table th{background:#edf2f7}</style>'
    data_links='<p>Таблицы: <a href="heart_rr_analysis/function_analysis/ventricular_function_estimates.csv">УО и ФВ</a> · <a href="heart_rr_analysis/function_analysis/chamber_geometry.csv">геометрия масок</a> · <a href="heart_rr_analysis/function_analysis/geometry_changes.csv">межкадровые изменения</a></p>'
    qc_link='<p><a href="heart_rr_analysis/function_analysis/ct_review/index.html">КТ и маски в кадрах наблюдаемых экстремумов</a></p>'
    # Keep code in modules; these cells retain the established report-only notebook contract.
    cells=[]
    def md(s):
        cell=nbformat.v4.new_markdown_cell(s);cell.metadata['tags']=[TAG];cells.append(cell)
    def shown(s):
        cell=nbformat.v4.new_code_cell('# Расчёт и построение выполняются в build_cardiac_function_report.py.',outputs=[nbformat.v4.new_output('display_data',data={'text/html':s},metadata={})]);cell.metadata['tags']=[TAG];cells.append(cell)
    md(narrative.STYLE+"\n\n"+narrative.FRONT)
    md(INTRO)
    shown(style+'<p class="scientific-caption"><b>Таблица 1.</b> Экстремумы доступных масок и условные оценки полного цикла.</p>'+function_table(result)+data_links+'<p class="scientific-caption"><b>Рисунок 1.</b> Наблюдаемые объёмы и условные периодические профили желудочков.</p>'+figure_html[0]+ventricular_difference_table(result)+narrative.estimate_interpretation(result))
    md(GEOMETRY_TEXT)
    shown(qc_link+'<p class="scientific-caption"><b>Рисунок 2.</b> Объём, площадь и удлинённость масок четырёх камер.</p>'+figure_html[1]+'<p><b>Вывод раздела.</b> Геометрия позволяет описать динамику масок несколькими признаками. Поскольку все признаки получены из одной сегментации, их согласованность не определяет анатомическую точность. Далее сопоставляются форма кривых ЛЖ и ПЖ и локализация межкадровых изменений.</p>')
    joint_insertion=len(cells)
    md(SCENARIO_TEXT)
    shown('<p class="scientific-caption"><b>Таблица 7 и рисунок 6.</b> Сценарные значения и их отклонения от исходного варианта; у Georg — наблюдаемый размах.</p>'+scenario_table(result)+figure_html[2]+'<p><b>Вывод раздела.</b> Постоянная добавка может изменить относительный размах при сохранении абсолютного; постоянный множитель — абсолютный размах при сохранении относительного. Поправка по площади зависит от фазы. Ни один вариант не выбран как анатомическая коррекция.</p>')
    md(narrative.SENSITIVITY)
    shown('<p class="scientific-caption"><b>Таблица 8.</b> Минимальные и максимальные оценки на исследованной сетке периодических моделей.</p>'+sensitivity_table(result)+'<p><b>Вывод раздела.</b> Зависимость оценки от параметров аппроксимации сохраняет модельную неопределённость. Представленный диапазон нельзя использовать как интервал точности измерения.</p>')
    reference_method,reference_results,reference_limits,reference_manifest=build_reference_section(result,work)
    md(reference_method)
    shown(reference_results)
    md(reference_limits)
    md(narrative.DISCUSSION)
    md(narrative.LIMITATIONS)
    md(narrative.CONCLUSIONS)
    md(narrative.REPRODUCIBILITY)
    notebook_path=output_dir.parent/NOTEBOOK_NAME
    nb=nbformat.read(notebook_path,as_version=4)
    from summarize_georg_mask_pilot import notebook_cells
    pilot_cells=notebook_cells(output_dir.parent)
    from build_cardiac_joint_report import build_joint_report
    joint_cells,joint_manifest=build_joint_report(rows,inputs,derived_root,work,presentation_only=presentation_only)
    if pilot_cells:
        pilot_cells[0].source="## Приложение Б. Пилот автоматической коррекции масок Георга\n\n**Исторический вычислительный эксперимент.** Улучшение не подтверждено. После него автор решил продолжать на исходных масках; новый ручной референс не ожидается. Предложения дальнейшей разметки ниже не являются текущим планом.\n\n"+pilot_cells[0].source.replace("## Пилот уточнения", "### Пилот уточнения",1)
    context=[cell for cell in nb.cells if not {TAG,'georg-pilot-analysis','cardiac-joint-dynamics'}.intersection(cell.metadata.get('tags',[]))]
    for cell in context:
        cell.metadata['tags']=list(dict.fromkeys(cell.metadata.get('tags',[])+['cardiac-rr-context']))
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('# Проверка автоматической сегментации сердца в 4D-КТ','## Приложение А. Кардиосинхронизация и исходные представления')
            cell.source=cell.source.replace('**Статус:** исследовательский вычислительный тест; автоматические маски ожидают\nпокадровой экспертной проверки.','**Статус приложения:** сохранённые методические сведения и результаты первичного разбора. Они дополняют основной анализ; независимая анатомическая проверка масок не выполнена.')
            cell.source=cell.source.replace('нормальная\nразличимость крови и миокарда есть только у Georg благодаря остаточному\nконтрасту; для Adam и Nix она не подтверждена.','остаточный\nконтраст сохранялся у Georg. По последующему решению автора различимость\nграниц недостаточна для подготовки новых ручных масок; они не ожидаются.')
            cell.source=cell.source.replace('Доверительный интервал точности сегментации пока не рассчитан. Нужны независимые\nразметки двух экспертов, повторная разметка части кадров одним экспертом и','Независимая точность сегментации не установлена. Первоначальный план её\nпроверки предусматривал разметки двух экспертов, повторную разметку части\nкадров одним экспертом и')
            cell.source=cell.source.replace('Фазы одного сердца зависимы, поэтому\nинтервалы следует строить кластерным бутстрэпом по испытуемым. При трёх\nиспытуемых популяционный интервал будет только пилотным.','Этот план сейчас не выполняется. Фазы одного сердца зависимы;\nпервоначально рассматривался кластерный бутстрэп по испытуемым. При трёх\nиспытуемых обоснованность популяционного интервала не установлена.')
            lines=cell.source.splitlines()
            cell.source='\n'.join(('### '+line[3:] if line.startswith('## ') and not line.startswith('## Приложение А.') else line) for line in lines)
    nb.cells=cells[:joint_insertion]+joint_cells+cells[joint_insertion:]+context+pilot_cells
    nb.metadata['scientific_notebook']={'genre':'scientific-notebook','language':'ru','structure':'context_problem_objective_method_evidence_interpretation_limitations_conclusion','presentation_only':presentation_only}
    nb.metadata['cardiac_joint_dynamics']={'status':'exploratory_hypothesis_not_validated','paired_intervals':joint_manifest['counts']['comparisons'],'manual_reference_expected':False}
    nb.metadata['georg_mask_pilot']={'included':bool(pilot_cells),'accepted_candidates':0,'primary_masks_replaced':False}
    nb.metadata['cardiac_function_analysis']={'schema_version':1,'status':'exploratory_hypothesis_not_validated','primary_outputs':['sv_ml','ef_percent'],'manual_masks_added':False}
    nb.metadata['cardiac_reference_indexing']={'status':reference_manifest['status'],'record_count':reference_manifest['record_count'],'prior_estimates_recomputed':not presentation_only,'new_indices_computed':True}
    nbformat.write(nb,notebook_path)
    # Canonical reader export: remove source code, retain Markdown and stored outputs.
    from nbconvert import HTMLExporter
    exporter=HTMLExporter(exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True)
    html_body,_=exporter.from_notebook_node(nb,resources={'metadata':{'name':'21.03 — Геометрия камер и условные оценки УО/ФВ'}})
    html_path=notebook_path.with_suffix('.html')
    html_path.write_text(html_body,encoding='utf-8')
    manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'status':'exploratory_hypothesis_not_validated','inputs':inputs,
              'scientific_status':'conditional_estimates_not_validated','new_manual_segmentation':False,'independent_reference_available':False,
              'units_basis':'source_CT_grid_and_conversion_manifest_spacing_mm','scenario_parameters':'illustrative_not_anatomical_not_confidence_bounds',
              'ct_review_phase_count':review_count,'geometry_rows':len(rows),'estimate_rows':len(result),'code_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ['cardiac_function_analysis.py','build_cardiac_function_report.py','analyze_subtraction_and_rr.py','build_cardiac_ct_review.py','summarize_georg_mask_pilot.py','cardiac_joint_dynamics.py','build_cardiac_joint_report.py','cardiac_notebook_text.py']},
              'output_sha256':{p.name:sha256(p) for p in work.glob('*.csv')},'notebook_sha256':sha256(notebook_path),'reader_html_sha256':sha256(html_path)}
    manifest['presentation_only_refresh']=presentation_only
    manifest['reference_indexing_manifest_sha256']=sha256(work/'reference_indexing/manifest.json')
    manifest['code_sha256']['cardiac_reference_indexing.py']=sha256(Path(__file__).with_name('cardiac_reference_indexing.py'))
    manifest['joint_dynamics_manifest_sha256']=sha256(work/'joint_dynamics/manifest.json')
    if pilot_cells:
        manifest['georg_pilot_reader_sha256']=sha256(work/'georg_pilot_20260910/final_reader.json')
    (work/'function_analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Function report built: {len(rows)} chamber-phases; {len(result)} estimates/scenarios',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--derived-root',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'heart_rr_analysis');parser.add_argument('--subjects',nargs='+',default=['adam','nix','georg'])
    parser.add_argument('--presentation-only',action='store_true',help='Rebuild text and figures from verified saved calculations without recomputing numerical results')
    args=parser.parse_args()
    if not args.presentation_only and args.derived_root is None: parser.error('--derived-root is required for calculation')
    refresh_function_report(args.derived_root,args.output_dir,args.subjects,presentation_only=args.presentation_only)

if __name__=='__main__':main()
