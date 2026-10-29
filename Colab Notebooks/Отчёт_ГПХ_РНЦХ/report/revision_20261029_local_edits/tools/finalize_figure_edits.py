"""Write only local figure mapping and audit; verify frozen inputs first."""
from pathlib import Path
import hashlib
import json
import sys
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
REV=Path(__file__).resolve().parents[1]
ROOT=REV.parents[3]
BASE=REV.parent/'revision_20261029_author_review'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rel=lambda p:p.relative_to(ROOT).as_posix()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def main():
    frozen=read(REV/'tools/figure_edits_inputs.json')
    assert all((BASE/p).is_file() and sha(BASE/p)==h for p,h in frozen.items()),'Base revision changed'
    manifest=read(BASE/'build_manifest.json')
    assert len(manifest['figures'])==19
    for f in manifest['figures']:assert sha(Path(f['path']))==f['sha256']
    ecg=read(REV/'tools/ecg_display_check.json')
    material=read(REV/'tools/material_label_checks.json')
    anatomy=read(REV/'tools/materials_anatomy_check.json')
    embedded=read(REV/'tools/embedded_figure_source_check.json')
    assert embedded['passes']==1 and embedded['potential_saved_data_files']==0
    assert sha(ROOT/anatomy['source_svg']['path'])==anatomy['source_svg']['sha256']
    for r in material['inputs']:assert sha(ROOT/r['path'])==r['sha256']
    assert sha(ROOT/ecg['source']['path'])==ecg['source']['sha256']
    captions={
        1:manifest['figures'][0]['caption'],
        10:manifest['figures'][9]['caption']+' ЭКГ и ординаты меток QRS показаны с множителем −1 только при отображении; времена QRS-кандидатов сохранены.',
        12:manifest['figures'][11]['caption'].replace('изменение объёма от опоры','изменение объёма относительно исходного состояния')+' Под «опорой» в названиях панелей понимается исходное состояние — первая доступная фаза данного R–R.',
        13:'Объёмная чувствительность STL, G1 и L1 при изменении объёмного шага и числа квадратурных точек на одной сетке. Столбцы соответствуют двум наборам опорных параметров тканей: слева для сердечного компартмента задано ρ_H = 4 Ом·м, справа используются условные параметры подгонки STL. Здесь ρ_H — эффективное удельное сопротивление сердечного компартмента. Цвет соответствует трём альтернативным монтажам. Горизонтальная шкала общая для всех панелей; q обозначает число квадратурных точек. На горизонтальной оси |ΔV| — полусумма модулей фактических отрицательной и положительной объёмных проб, V₀ — исходный номинальный объём STL; номинальные пробы составляют ±1; ±2,5 и ±5% от V₀.',
        14:manifest['figures'][13]['caption']+' В левой панели для сердечного компартмента задано ρ_H = 4 Ом·м; ρ_H — его эффективное удельное сопротивление.',
        15:manifest['figures'][14]['caption'].replace('эффективного сопротивления сердца','эффективного удельного сопротивления сердечного компартмента')+' Столбцы соответствуют двум наборам опорных параметров тканей: слева для сердечного компартмента задано ρ_H = 4 Ом·м, справа используются условные параметры подгонки STL. Здесь ρ_H — эффективное удельное сопротивление сердечного компартмента.',
        19:manifest['figures'][18]['caption']+' Под «одной опорой» на рисунке понимается эта известная исходная фаза.',
    }
    replacements={}
    for number,record in [(1,anatomy),(10,ecg),(13,material['figures'][0]),(15,material['figures'][1])]:
        f=manifest['figures'][number-1]
        old=rel(Path(f['path']));dest=ROOT/record['new_path']
        assert dest.exists() and sha(dest)==record['sha256']
        assert Image.open(dest).size==Image.open(f['path']).size
        replacements[old]=dict(new_path=record['new_path'],figure_number=number,key=f['key'],
            caption_text=captions[number],source_asset_sha256=f['sha256'],new_sha256=sha(dest),
            dimensions_px=list(Image.open(dest).size),word_width_cm=f['width_cm'],
            word_visible_height_cm=f['visible_height_cm'],
            basis=(
                'По прямому уточнению автора неоднозначная фраза «Анатомическая опора» заменена на '
                '«Индивидуальная анатомия по КТ» в одном text-элементе исходного SVG (render_visuals.cjs:28). '
                'Все атрибуты и остальные элементы SVG неизменны; PNG заново отрисован из SVG. '
                'Исходный SVG воспроизвёл базовый PNG попиксельно, вне целевой строки пиксели новой схемы совпали.'
                if number==1 else
                'В сохранённой HTML-сцене 4 ординаты ЭКГ и всех пяти QRS-маркеров отрицательны в точках QRS; '
                'render_visuals.cjs:100–105 переносит их в верхнюю панель без смены знака. По прямому указанию автора '
                'применён display multiplier −1 к y ЭКГ и y маркеров; все x и оба импедансных ряда неизменны. '
                'Это соглашение отображения, а не установленная физическая полярность отведения.'
                if number==10 else
                'heart_montage_figures.py и полные CSV из report/results/montage_sensitivity_v1 задают те же '
                'кривые/столбцы для diagnostic4 и commonSTLfit. В diagnostic4 параметр сердечного компартмента '
                'rho_heart_ohm_m равен 4 Ом·м; это условный параметр модели. Надпись «ρ сердца» уточнена до ρ_H; '
                'полное определение «сердечный компартмент, ρ_H = 4 Ом·м» дано в подписи. '
                '«Условные ρ из подгонки STL» заменено на «опорные ρ из подгонки STL»; условность сохранена в подписи.'),
            display_multiplier=-1 if number==10 else None,
            reproduction_check='tools/materials_anatomy_check.json' if number==1 else 'tools/ecg_display_check.json' if number==10 else 'tools/material_label_checks.json')
        if number==1:replacements[old]['new_svg_path']=anatomy['new_svg_path']
        if number==13:
            replacements[old]['axis_label_change']=material['figures'][0]['volume_probe_axis']
            replacements[old]['basis']+=' Подпись горизонтальной оси заменена на «Объёмная проба |ΔV| / V₀, %». CSV подтверждает номинальные ±1/±2,5/±5%; фактические x сохранены как полусумма модулей парных проб, без округления к делениям.'
        if number==15:
            replacements[old]['title_label_change']={'before':'изменением ρ сердца','after':'изменением ρ_H'}
            replacements[old]['caption_change']={'before':'эффективного сопротивления сердца','after':'эффективного удельного сопротивления сердечного компартмента'}

    pending=[]
    definitions=[
        (12,[['Изменение объёма от опоры R–R','Изменение объёма от исходного состояния'],
             ['Ошибка перемещения от опоры','Ошибка перемещения от исходного состояния']],
         'Colab Notebooks/cardiac_4d_segmentation/summarize_nik_whole_heart_geometry.py:667–691',
         'Нормализованные rows из summary.local.json, записываемого run() этого скрипта: первый R–R, '
         'T0/I1/I2/I3/G1/G2/G3a; volume_ml, anchor_volume_change_ml, anchor_displacement_error_mm, dice, '
         'optimization_acceptable, phase_id; список исходных фаз groups/manifest.phases.',
         'Код явно строит приращения и ошибку перемещения относительно первой доступной фазы данного R–R. '
         'Исходная нормализованная таблица этого прогона не найдена; проверен код и сохранённый PNG.'),
        (14,[['Одинаковое ΔV; ρ сердца = 4 Ом·м','Одинаковое ΔV; ρ_H = 4 Ом·м'],
             ['Заданное ρ целого сердца, Ом·м','ρ_H сердечного компартмента, Ом·м']],
         'Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261028/tools/prepare_reader_figures.py:347–372',
         'Colab Notebooks/Отчёт_ГПХ_РНЦХ/pulmovista_check_20261028/local/stl_electrical_comparison/summary.local.json: '
         'shape_comparison, reference_volume_probe_ml, heart_material_scenarios.',
         'Для левой панели сохранены отдельные пробы семи форм, для правой — три отдельные точки STL. '
         'Нужная сводка отсутствует. Публичная серия трёх форм STL/G1/L1 не заменяет эти семь форм и три сценарные точки.'),
        (19,[['Одна опора: 14%; временной зазор: 140 мс','Исходная фаза: 14%; временной зазор: 140 мс']],
         'Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261028/tools/prepare_reader_chamber_figure.py:50–132',
         'Colab Notebooks/Отчёт_ГПХ_РНЦХ/execution_20261026/single_phase/nix_only/exploratory/predictions.csv; '
         'single_phase/ct_dataset/dataset.json; single_phase/nix_only/exploratory/window_indices.csv.',
         'Код и provenance chamber_figure_edit.json фиксируют одну известную фазу 14% R–R и восемь '
         'отложенных фаз. Таблицы предсказаний и КТ для этого графика отсутствуют; доступных PNG недостаточно для воспроизведения.'),
    ]
    for number,changes,code,missing,basis in definitions:
        f=manifest['figures'][number-1]
        pending.append(dict(old_asset=rel(Path(f['path'])),figure_number=number,key=f['key'],new_path=None,
            status='not_rendered_missing_saved_arrays',caption_text=captions[number],
            proposed_label_changes=changes,source_code_locator=code,missing_inputs=missing,basis=basis,
            source_asset_sha256=f['sha256']))
        pending[-1]['embedded_data_check']='tools/embedded_figure_source_check.json'
        pending[-1]['basis']+=' Один дополнительный проход по сохранённым HTML/Plotly JSON/выводам notebook и сопроводительному архиву не выявил достаточных встроенных численных данных. Уточнение термина дано в подписи; PNG не оцифровывался.'

    inspection=[]
    for f in manifest['figures']:
        number=f['number']
        status='replacement_rendered' if number in [1,10,13,15] else 'needs_edit_missing_data' if number in [12,14,19] else 'keep'
        note={1:'Неоднозначная фраза заменена по уточнению автора на «Индивидуальная анатомия по КТ»; остальные элементы неизменны.',
              10:'Единственная ЭКГ среди 19 использованных рисунков; QRS направлены вниз в исходнике.',
              12:'Два названия панелей содержат неоднозначное «от опоры».',
              13:'Три названия левого столбца: «ρ сердца = 4 Ом·м»; правый столбец — условные параметры STL.',
              14:'Левый заголовок «ρ сердца = 4 Ом·м» и название оси ρ справа требуют определения компартмента.',
              15:'Три названия левого столбца: «ρ сердца = 4 Ом·м»; правый столбец — условные параметры STL.',
              16:'«Опорный участок 36–44 с» однозначно обозначает временной интервал базовой инверсии; сохранён.',
              18:'Д4/О-STL — идентификаторы сценариев; численный смысл и подпись сохранены.',
              19:'«Одна опора: 14%» следует назвать исходной фазой.'}.get(number,'ЭКГ и целевые неоднозначные надписи отсутствуют; замена не требуется.')
        inspection.append(dict(number=number,key=f['key'],old_asset=rel(Path(f['path'])),sha256=f['sha256'],
                               status=status,note=note,visually_inspected=True))

    output=dict(schema='local_figure_edits_v1',base_revision=rel(BASE),figure_count=19,
        asset_mapping=replacements,unrendered_edits=pending,inspection=inspection,
        embedded_source_check=dict(path='tools/embedded_figure_source_check.json',passes=1,
            repository_files=embedded['repo_files_examined'],archive_entries=len(embedded['archive']['entries_examined']),
            potential_saved_data_files=0,conclusion=embedded['conclusion']),
        input_freeze=dict(file='tools/figure_edits_inputs.json',protected_files=len(frozen),all_hashes_unchanged=True),
        scope=dict(figure_mode='with_figures',formatting_mode='existing_report_figure_layout',
            scientific_sources='existing report assets, saved HTML/CSV and plotting code only',
            Word_written=False,shared_manifest_written=False,drafts_written=False,
            raw_signal_written=False,detector_run=False,FEM_run=False,TEPS_used=False,
            raster_plot_reflected=False,raster_plot_repainted=False))
    (REV/'figures/figure_edits_mapping.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Локальная правка рисунков — 29.10.2026','',
        'База — `revision_20261029_author_review/build_manifest.json`: 19 использованных рисунков. '
        'Основной текст и структура отчёта не редактировались. Новые рисунки не интегрированы в Word.','',
        'Применены требования `docs/RUSSIAN_SCIENTIFIC_WRITING.md`, scientific-evidence-workflow и '
        'avoid-ai-writing-russian с контекстом `vak`; отдельный голос не назначен. '
        'Научные основания проверялись по существующим данным до языковой правки.','',
        '## Результат','',
        'Подготовлены четыре замены: рисунки 1, 10, 13 и 15. Для рисунков 12, 14 и 19 сохранены предложения '
        'подписей и указаны отсутствующие массивы; новые изображения для них не создавались. '
        'Машиночитаемая карта: `../figures/figure_edits_mapping.json`, поля `asset_mapping` и `unrendered_edits`.','',
        '### Рисунок 1: индивидуальная анатомия по КТ','',replacements[anatomy['old_asset']]['basis'],'',
        'Фраза изменена внутри исходного SVG; шрифт, размер, координаты текста, блоки, стрелки, '
        'цвета и прочие надписи сохранены. Новая строка помещается в исходный холст. '
        'Изменение проверено визуально и попиксельно вне целевой строки; исходные SVG/PNG не перезаписывались.','',
        '### Рисунок 10: полярность ЭКГ','',replacements[ecg['old_asset']]['basis'],'',
        'Операция: `y_display = −1 × y_saved` для ЭКГ и ординат сохранённых QRS-маркеров. '
        'Множитель −1 не включён в сигнал, детектор или вычисленные результаты. '
        'Автомасштаб верхней оси соответствует изменённым ординатам. Времена меток и импеданс не менялись.','',
        'Сохранённые времена QRS, с: `'+json.dumps(ecg['QRS_time_s'])+'`. '
        'Исходные ординаты меток, мВ: `'+json.dumps(ecg['QRS_original_mV'])+'`.','',
        'Краткое дополнение к подписи: «ЭКГ и ординаты меток QRS показаны с множителем −1 только при отображении; '
        'времена QRS-кандидатов сохранены».','',
        'Проверки: исходный код до правки воспроизвёл базовый PNG попиксельно; '
        'после смены знака оба импедансных ряда и проверенная внутренняя область нижней панели совпадают попиксельно. '
        'Массивы x совпадают точно; y ЭКГ и меток точно равны отрицанию исходных y. '
        'Смена знака выполнена над сохранёнными числами, отражение растра не применялось.','',
        '### Рисунки 13 и 15: параметры компартмента','',
        'Использованы полные CSV `report/results/montage_sensitivity_v1/tables`: derivatives, '
        'volume_refinement, electrical_states, finite_compensation_initial и finite_compensation_newton. '
        'Проверены SHA-256 этих файлов по собственному манифесту результата. Взяты только графические блоки '
        '`heart_montage_figures.py:62–91,117–138` через AST; расчётные процедуры не выполнялись.','',
        'В кратких заголовках введено ρ_H; в предлагаемых полных подписях оно определено как эффективное '
        'удельное сопротивление сердечного компартмента, в левом столбце ρ_H = 4 Ом·м. '
        'В правом столбце «опорные ρ из подгонки STL» обозначают набор условных параметров; '
        'это не измеренные свойства тканей. Числа, единицы, пределы осей, цвета, маркеры, '
        'все три монтажа и оба столбца сохранены.','',
        'Последнее уточнение автора: общий заголовок рисунка 15 теперь содержит «изменением ρ_H». '
        'В caption15 термин заменён на «эффективного удельного сопротивления сердечного компартмента»; '
        'статус условных параметров сохранён. Горизонтальная ось рисунка 13 подписана '
        '«Объёмная проба |ΔV| / V₀, %». Поле probe_signed_step в electrical_states.csv подтверждает '
        'номинальные ±1; ±2,5; ±5%. При этом сохранённые x учитывают фактические объёмные пробы: '
        'x = 100 × (minus_step + plus_step) / (2V₀). Например, для STL/diagnostic4/q64/top_gap100 '
        'они равны 0,9932184991762946; 2,493425449977883; 4,99506754755036%. '
        'Они не заменялись номинальными значениями. Определение |ΔV| дано в подписи.','',
        'Размеры PNG: рисунок 10 — 3060 × 1700; рисунок 13 — 2160 × 1500; рисунок 15 — 2460 × 1710. '
        'При интеграции следует сохранить width_cm и visible_height_cm из базового манифеста; '
        'они также записаны в карте. Рисунки 13/15 имеют 180 dpi, поскольку использован исходный научный рендерер; '
        'физический размер размещения задаётся указанной шириной Word.','',
        'Проверки: координаты всех графических объектов и пределы осей совпали до и после локальной правки. '
        'Кроме того, исходный рендерер воспроизвёл все 12 внутренних областей панелей исходных кэшированных '
        'рисунков попиксельно. Эти же области новых PNG совпали с исходными попиксельно после целочисленного '
        'переноса в существующую компоновку. Растровые панели не копировались: все кривые и столбцы отрисованы кодом. '
        'Масштаб и общий дизайн сохранены. Итоговые PNG просмотрены.','',
        '## Невыполненные замены из-за отсутствия данных','']
    for p in pending:
        lines += [f"### Рисунок {p['figure_number']}",'',p['basis'],'',
                  'Код: `'+p['source_code_locator']+'`.','',
                  'Требуемые данные: `'+p['missing_inputs']+'`.','',
                  'Предлагаемые надписи:']
        lines += ['','\n'.join('- «'+a+'» → «'+b+'».' for a,b in p['proposed_label_changes']),'']
    lines += ['Диск D: недоступен в текущей среде. Проверены доступные файлы репозитория, сохранённые '
              'сводки Scorpion, публичный численный набор montage_sensitivity_v1 и состав приложенного '
              'архива материалов научной редакции. Недостающие массивы не подменялись оцифровкой PNG, '
              'интерполяцией точек или результатами другого расчёта.','',
              'Дополнительная проверка встроенных данных выполнена одним проходом: '
              f"{embedded['repo_files_examined']} файлов HTML/HTM/IPYNB/JSON репозитория и "
              f"{len(embedded['archive']['entries_examined'])} записей HTML/HTM/IPYNB/JSON/CSV сопроводительного архива. "
              'Найденные совпадения — описания, ссылки на расчётные файлы, манифесты или растровые представления; '
              'достаточных численных рядов или Plotly-сцен этих трёх фигур не выявлено. '
              'Архив содержит обработанные таблицы КТ и агрегаты отчёта, но не требуемые фазовые предсказания, '
              'нормализованные ряды семи геометрических методов или полную сводку семи электрических форм. '
              'Перечень совпадений и SHA-256 архива записаны в `tools/embedded_figure_source_check.json`. '
              'Повторный широкий поиск не выполнялся.','',
              '## Осмотр всех 19 рисунков','',
              '| № | Asset | Решение | Основание |','|---|---|---|---|']
    for r in inspection:lines.append(f"| {r['number']} | {Path(r['old_asset']).name} | {r['status']} | {r['note']} |")
    lines += ['', '## Полные подписи для интеграции','']
    for n,c in captions.items():lines += [f'**Рисунок {n}.** '+c,'']
    lines += ['## Сохранность и воспроизведение','',
        f'Все {len(frozen)} файлов базовой ревизии сохранили SHA-256, включая Word, рисунки, общие манифесты и исходные тексты. '
        'Хэши сохранённых HTML/CSV и использованного кода также проверены после выполнения. '
        'Записи, численные результаты, исходный код детекторов и тексты авторской рецензии не изменялись.','',
        'Из корня репозитория:','',
        '```powershell',
        'node "Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261029_local_edits/tools/rebuild_materials_anatomy.cjs"',
        'node "Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261029_local_edits/tools/rebuild_ecg_display.cjs"',
        'python "Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261029_local_edits/tools/rebuild_material_labels.py"',
        'python "Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261029_local_edits/tools/finalize_figure_edits.py"',
        '```','',
        'Рендеринг ЭКГ использует локальные Playwright/Sharp и встроенную в исходный HTML библиотеку Plotly; '
        'сетевые запросы блокируются. Для материальных графиков использованы Python, Matplotlib 3.10.9, NumPy и Pillow. '
        'Проверки сохранности чисел и пикселей включены в скрипты. Скрипт внешнего детектора ИИ и ТЕПС не использовались. '
        'Языковая проверка подписей выполнена вручную; автоматический evidence-bundle validator не запускался. '
        'Исследовательская и клиническая валидация по этой редакционной правке не заявляется.','']
    (REV/'audit').mkdir(exist_ok=True)
    (REV/'audit/figure_edits.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(rendered=[1,10,13,15],not_rendered=[12,14,19],inspected=19,protected_files_unchanged=len(frozen)),ensure_ascii=False))

if __name__=='__main__':main()
