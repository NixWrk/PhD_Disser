"""Summarize an automatic historical cardiac sequence; never estimate accuracy without references."""
from __future__ import annotations
import argparse, base64, csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import nibabel as nib
import numpy as np

MASKS = {
    'LV': ('heartchambers_highres/heart_ventricle_left.nii.gz', 'Левый желудочек'),
    'RV': ('heartchambers_highres/heart_ventricle_right.nii.gz', 'Правый желудочек'),
    'LA': ('heartchambers_highres/heart_atrium_left.nii.gz', 'Левое предсердие'),
    'RA': ('heartchambers_highres/heart_atrium_right.nii.gz', 'Правое предсердие'),
    'whole_heart': ('total_heart/heart.nii.gz', 'Целое сердце'),
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def mask_summary(image, ct):
    if image.shape != ct.shape or not np.allclose(image.affine,ct.affine,atol=1e-5,rtol=0):
        raise ValueError('Mask and CT grids differ')
    if ct.header.get_xyzt_units()[0]!='mm': raise ValueError('CT units must be mm')
    data=np.asanyarray(image.dataobj)
    if not np.isfinite(data).all() or not np.all(np.isin(data,[0,1])):
        raise ValueError('Expected finite binary mask')
    mask=data>0; count=int(mask.sum())
    voxel_ml=abs(float(np.linalg.det(ct.affine[:3,:3])))/1000
    if not np.isfinite(voxel_ml) or voxel_ml<=0: raise ValueError('Invalid voxel volume')
    indices=np.argwhere(mask)
    center=nib.affines.apply_affine(ct.affine,indices.mean(axis=0)).tolist() if count else [None]*3
    touches=bool(any(mask.take(i,axis=axis).any() for axis in range(3) for i in (0,-1)))
    return {'volume_ml':count*voxel_ml,'voxel_count':count,'center_ras_mm':center,'touches_image_boundary':touches,'empty':count==0}

def analyze(sequence, masks, out, allow_partial=False):
    manifest_path=sequence/'phases'/'conversion_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    phases=sorted(manifest['phases'],key=lambda p:p['phase_percent'])
    rows=[]; missing=[]; provenance=[]
    for p in phases:
        pid=p['phase_id']; ct_path=sequence/'phases'/f'{pid}.nii.gz'
        paths={key:masks/pid/rel for key,(rel,_) in MASKS.items()}
        if not all(path.is_file() for path in paths.values()):missing.append(pid);continue
        ct_hash=sha(ct_path)
        if ct_hash!=p['sha256']: raise ValueError('CT hash differs from conversion manifest')
        ct=nib.load(ct_path)
        for key,path in paths.items():
            metrics=mask_summary(nib.load(path),ct)
            rows.append({'phase_id':pid,'phase_percent':p['phase_percent'],'structure':key,**metrics})
            provenance.append({'phase_id':pid,'structure':key,'mask_sha256':sha(path),'ct_sha256':ct_hash})
    if missing and not allow_partial:raise ValueError(f'Missing phases: {missing}')
    if not rows:raise ValueError('No complete automatic phase outputs')
    for key in MASKS:
        group=[r for r in rows if r['structure']==key]
        v0=group[0]['volume_ml']
        for i,r in enumerate(group):
            r['reference_phase_percent']=group[0]['phase_percent']
            r['delta_from_reference_ml']=r['volume_ml']-v0
            r['previous_phase_percent']=group[i-1]['phase_percent'] if i else None
            r['delta_from_previous_ml']=r['volume_ml']-group[i-1]['volume_ml'] if i else None
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'automatic_results_without_reference','created_at':datetime.now(timezone.utc).isoformat(),'case_id':manifest['case_id'],'sequence_id':manifest['sequence_id'],'expected_phases':len(phases),'completed_phases':len(rows)//len(MASKS),'missing_phases':missing,'conversion_manifest_sha256':sha(manifest_path),'rows':rows,'provenance':provenance,'accuracy_evaluated':False,'ED_ES_accepted':False,'temporal_unit':'percent_of_reconstructed_cycle','anatomical_visual_review':'not_performed'}
    (out/'automatic_sequence_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (out/'automatic_volumes.csv').open('w',newline='',encoding='utf8') as f:
        fields=[k for k in rows[0] if k!='center_ras_mm'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v for k,v in r.items() if k in fields} for r in rows)
    queue={'purpose':'Future independent reference annotations; not a validation manifest','reference_available':False,'entries':[{'phase_id':p['phase_id'],'phase_percent':p['phase_percent'],'structure':key,'reference_path':None,'status':'awaiting_independent_reference'} for p in phases for key in ['LV','RV','LA','RA']]}
    (out/'reference_annotation_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report

def notebook(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nbformat
    from nbconvert import HTMLExporter
    import io
    rows=report['rows']; cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
    md('''# 21.08. Изменения объёмов камер на исторической контрастной КТ

## Цель и статус исследования

Задача — подготовить проверку точности TotalSegmentator при оценке межфазных изменений объёмов камер сердца. Сейчас представлены автоматические результаты и технический контроль. Независимые референсные маски отсутствуют; ошибка алгоритма, точность ударного объёма и фракции выброса ещё не установлены. По решению автора исследование продолжается без ожидания ручной разметки.

Используются те же задачи TotalSegmentator, что в [21.03](21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb): `total` для целого сердца и `heartchambers_highres` для камер. Существующие результаты 21.03 не заменяются. Исторические серии образуют отдельную проверочную ветку.''')
    md(f'''## Данные и метод расчёта

Для первого прогона выбрана одна полная последовательность из 20 реконструированных фаз: 0–95% через 5%. Выбор сделан по полноте последовательности до получения автоматических масок. В текущем отчёте обработано **{report['completed_phases']} из {report['expected_phases']} фаз**. Неполный набор позволяет оценить только изменения между представленными фазами; его размах не является размахом полного цикла.

Исходный формат — многокадровый Enhanced CT. В выбранной последовательности 99 пространственных срезов на фазу, размер 512 × 512, толщина реконструированного среза 5 мм и шаг между его положениями 2,5 мм. Эти величины имеют разный смысл: перекрытие срезов не увеличивает исходное пространственное разрешение до 2,5 мм. В DICOM отмечено введение контраста; это поле не устанавливает качество различимости каждой анатомической границы.

Положение срезов, ориентация, размеры пикселя и преобразование интенсивности читаются из функциональных групп DICOM. Координаты преобразуются из LPS в RAS; единицы NIfTI — миллиметры. Маски анализируются на сетке исходной преобразованной КТ, без сглаживания и удаления компонент. Объём равен числу включённых вокселей, умноженному на модуль определителя пространственной части affine-матрицы и делённому на 1000; результат выражен в миллилитрах.

Фазовые проценты не преобразуются в секунды. Фазы 0% и 50% не объявляются автоматически конечной диастолой и систолой. Кровь каждого желудочка и предсердия рассматривается отдельно; изменение объёма наружной маски сердца не подменяет ударный объём желудочка.''')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for ax,key in zip(axes.flat,['LV','RV','LA','RA']):
        group=[r for r in rows if r['structure']==key]
        ax.plot([r['phase_percent'] for r in group],[r['volume_ml'] for r in group],'-o',ms=4)
        ax.set_title(MASKS[key][1]);ax.set_ylabel('Объём, мл');ax.set_xlabel('Фаза реконструкции, %');ax.grid(alpha=.25)
    fig.suptitle('Объёмы автоматических масок; точность относительно анатомии не установлена');fig.tight_layout()
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);plt.close(fig)
    cell=nbformat.v4.new_markdown_cell('## Автоматические объёмы камер\n\n![Объёмы четырёх камер](attachment:volumes.png)\n\n**Рисунок 1.** Точки соответствуют рассчитанным маскам. Линии соединяют соседние доступные фазы и не являются физиологической моделью или оценкой промежуточных состояний. Различие высот точек показывает изменение автоматического объёма; оно включает движение анатомических границ, особенности реконструкции и ошибки сегментации. Без референса эти составляющие не разделены.')
    cell['attachments']={'volumes.png':{'image/png':base64.b64encode(buf.getvalue()).decode()}};cells.append(cell)
    table=['| Камера | Минимум, мл | Максимум, мл | Размах доступных фаз, мл |','|---|---:|---:|---:|']
    for key in ['LV','RV','LA','RA']:
        g=[r['volume_ml'] for r in rows if r['structure']==key];table.append(f'| {MASKS[key][1]} | {min(g):.2f} | {max(g):.2f} | {max(g)-min(g):.2f} |')
    md('## Численный результат и его трактовка\n\n'+'\n'.join(table)+'\n\nТаблица описывает доступные автоматические маски. Приведённый размах пока не назван ударным объёмом: для этого требуется обосновать конечные состояния и проверить границы камер. При частичном прогоне добавление фаз может увеличить найденный размах. Даже полная последовательность не устраняет ошибку сегментации.')
    flagged=[r for r in rows if r['empty'] or r['touches_image_boundary']]
    md(f'''## Технический контроль и ограничения

Совпадение размеров и affine-матриц масок с КТ проверено для каждого включённого результата. Пустые маски и касания границы изображения проверены отдельно; найдено записей с такими признаками: **{len(flagged)}**. Отсутствие этих признаков не доказывает правильность клапанных и эндокардиальных границ. Анатомический визуальный просмотр в этом прогоне не выполнен.

В этой выборке один человек и одна последовательность. Фазы не являются независимыми участниками. Доверительные интервалы популяционной точности не рассчитываются. Полученные кривые нельзя переносить как нормальные физиологические законы на других людей. Будущая проверка на контрастных КТ отдельно от проверки применимости к нынешним слабоконтрастным сериям.

## Будущая проверка изменений объёма

Для каждой камеры и одинаковой пары фаз основной показатель — разность автоматического и референсного изменения объёма:

$$\\varepsilon_{{\\Delta V}}=[V_{{TS}}(t_2)-V_{{TS}}(t_1)]-[V_{{ref}}(t_2)-V_{{ref}}(t_1)] = e(t_2)-e(t_1).$$

Здесь $V_{{TS}}$ — объём автоматической маски, $V_{{ref}}$ — объём независимой референсной маски, $e(t)=V_{{TS}}(t)-V_{{ref}}(t)$ — ошибка объёма в конкретной фазе; все величины объёма выражены в миллилитрах. Постоянная по фазам составляющая ошибки сокращается при вычитании. Изменяющаяся составляющая непосредственно искажает оценку $\\Delta V$. Поэтому одного Dice или средней ошибки абсолютного объёма недостаточно.

Ручные маски должны соответствовать тем же фазам и правилам выделения камер, включая клапанные плоскости, папиллярные мышцы и трабекулы. Чтобы не подгонять проверочный результат к алгоритму, независимую разметку следует создавать без использования его контуров как начального референса; доступ разметчика к автоматическим результатам фиксируется. Пилот включает 0, 25, 50 и 75%; после технической проверки требуется вся последовательность. Для поверхностных метрик допуск задаётся до сравнения с учётом разрешения и воспроизводимости разметки.

Последующий анализ включает подписанную и абсолютную ошибку $\\Delta V$ в мл, ошибки объёмных кривых, ошибки отдельных объёмов и поверхностные метрики. Относительная ошибка изменения не интерпретируется при близком к нулю референсном изменении. УО и ФВ сравниваются только после независимого принятия конечных состояний. Протокол анатомического сравнения описан в [21.05](21.05_Валидация_анатомической_сегментации_сердца_4D_КТ.md).

## Вывод

Подготовлены автоматические пофазные объёмы и их изменения для последующего парного сравнения с ручными масками. Числа и графики показывают поведение алгоритма на выбранной контрастной последовательности. Пока они не позволяют утверждать, насколько точно TotalSegmentator измеряет изменения объёмов камер. Следующий содержательный результат — оценка ошибки этих изменений после независимой разметки; до её появления продолжаются автоматическая обработка, контроль геометрии и условные вычислительные эксперименты.''')
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'research_status':'automatic_results_without_reference','summary_phase_count':report['completed_phases']})
    nbformat.validate(nb);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(nbformat.writes(nb),encoding='utf8',newline='\n')
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    body,_=exporter.from_notebook_node(nb);path.with_suffix('.html').write_text(body,encoding='utf8',newline='\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence-dir',type=Path,required=True);p.add_argument('--mask-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--notebook',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    r=analyze(a.sequence_dir,a.mask_dir,a.output_dir,a.allow_partial)
    if a.notebook:notebook(r,a.notebook)
    print(json.dumps({'completed_phases':r['completed_phases'],'expected_phases':r['expected_phases'],'accuracy_evaluated':False}))
if __name__=='__main__':main()
