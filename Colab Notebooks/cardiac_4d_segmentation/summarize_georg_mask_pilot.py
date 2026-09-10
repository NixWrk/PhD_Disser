"""Summarize the two exploratory Georg pilots without promoting masks to references."""
from pathlib import Path
import argparse,csv,json,html
import nibabel as nib
import numpy as np
from pilot_blood_segmentation_candidates import sha256_file

CHAMBERS={'heart_ventricle_left':'ЛЖ','heart_ventricle_right':'ПЖ'}


def phase_pair_rows(rows):
    out=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        for chamber in CHAMBERS:
            group={r['phase_id']:r for r in rows if r['method']==method and r['chamber']==chamber}
            for first,second in [('phase_08','phase_10'),('phase_14','phase_10')]:
                if first not in group or second not in group:continue
                a,b=group[first],group[second];raw=a['raw_ml']-b['raw_ml'];proposal=a['candidate_ml']-b['candidate_ml']
                out.append({'method':method,'chamber':chamber,'first_phase_id':first,'second_phase_id':second,'raw_difference_ml':raw,'candidate_difference_ml':proposal,'change_in_difference_ml':proposal-raw,'status':'within_long_RR_phase_difference_not_accepted_SV'})
    return out


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def table(rows,columns):
    def value(v):return f'{v:.2f}'.replace('.',',') if isinstance(v,float) else html.escape(CHAMBERS.get(v,str(v)))
    return '<div style="overflow-x:auto"><table class="function-table"><thead><tr>'+''.join('<th>'+t+'</th>' for _,t in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+value(r.get(k,''))+'</td>' for k,_ in columns)+'</tr>' for r in rows)+'</tbody></table></div>'


def summarize(derived,first,second,output):
    output.mkdir(parents=True,exist_ok=True);rows=[];provenance={}
    for label,pilot in [('Без точек',first),('Последовательные подсказки',second)]:
        m=json.loads((pilot/'pilot_manifest.json').read_text(encoding='utf-8'))
        if m.get('run_status')!='completed_proposals_not_accepted':raise ValueError('Only completed, nonexcluded runs can be summarized')
        provenance[label]={'manifest_sha256':sha256_file(pilot/'pilot_manifest.json'),'method':m['method'],'model_provenance':m['model_provenance']}
        for phase in m['phases_processed']:
            pid=phase['phase_id'];source=derived/'georg/automatic_masks_v2'/pid/'heartchambers_highres'
            structures={name:np.asarray(nib.load(source/f'{name}.nii.gz').dataobj)>0 for name in ['heart_myocardium','aorta','pulmonary_artery']}
            for c in phase['candidates']:
                ch=c['chamber'];image=nib.load(pilot/c['candidate_file']);mask=np.asarray(image.dataobj)>0;voxel=abs(float(np.linalg.det(image.affine[:3,:3])))/1000
                if sha256_file(pilot/c['candidate_file'])!=c['candidate_sha256']:raise ValueError('Candidate changed')
                if sha256_file(source/f'{ch}.nii.gz')!=c['raw_sha256']:raise ValueError('Original mask changed')
                if not np.isclose(mask.sum()*voxel,c['metrics']['candidate_ml']):raise ValueError('Candidate measurement mismatch')
                row={**c['metrics'],'method':label,'proposal_sha256':c['candidate_sha256'],'source_sha256':c['raw_sha256']}
                for name,s in structures.items():row['overlap_'+name+'_ml']=int((mask&s).sum())*voxel
                if 'prompts' in m:
                    spec=next(s for s in m['prompts']['items'] if s['phase_id']==pid and s['chamber']==ch)
                    row['positive_points_satisfied']=all(mask[tuple(p)] for p in spec['positive_points'])
                    row['negative_points_satisfied']=all(not mask[tuple(p)] for p in spec['negative_points'])
                row['reader_page']=('guided/' if label=='Последовательные подсказки' else '')+f'{pid}_{ch}/index.html'
                rows.append(row)
    pairs=phase_pair_rows(rows);write_csv(output/'pilot_comparison.csv',rows);write_csv(output/'phase_pair_sensitivity.csv',pairs)
    result={'schema_version':1,'status':'exploratory_hypothesis_not_validated','primary_masks_replaced':False,'independent_manual_reference':False,'scientifically_accepted_candidates':0,'rows':rows,'phase_pairs':pairs,'provenance':provenance,'excluded_trial':'Batched point trial excluded after installed API warning; only sequential point trial retained here','interpretation':'Mask differences and cross-model overlaps are diagnostics, not accuracy. ED/ES remain unaccepted, short and long cycles are not merged.'}
    (output/'pilot_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result



def build_reader(result, output):
    # Measurements and explicit visual decisions remain separate inputs.
    decision_path=output/'visual_review_decisions.json'
    decisions=json.loads(decision_path.read_text(encoding='utf-8'))
    by_key={(d['method'],d['phase_id'],d['chamber']):d for d in decisions['items']}
    for row in result['rows']:
        decision=by_key[(row['method'],row['phase_id'],row['chamber'])]
        if row['proposal_sha256']!=decision['proposal_sha256']:
            raise ValueError('Visual decision belongs to a different candidate')
        if sha256_file(output/decision['evidence'])!=decision['evidence_sha256']:
            raise ValueError('Reviewed image changed')
    methods=list(dict.fromkeys(row['method'] for row in result['rows']))
    if len(methods)!=2:raise ValueError('This reader compares exactly two included methods')
    lookup={(r['method'],r['phase_id'],r['chamber']):r for r in result['rows']}
    volume_rows=[]
    for pid in ['phase_04','phase_08','phase_10','phase_14']:
        for ch in CHAMBERS:
            a,b=[lookup[(m,pid,ch)] for m in methods]
            if not np.isclose(a['raw_ml'],b['raw_ml']):raise ValueError('Methods used different baselines')
            volume_rows.append({'phase':pid[-2:],'cycle':'короткий' if pid=='phase_04' else 'длинный','chamber':ch,'raw':a['raw_ml'],'first':a['candidate_ml'],'second':b['candidate_ml']})
    volume_table=table(volume_rows,[('phase','Фаза'),('cycle','R–R'),('chamber','Камера'),('raw','Исходная маска, мл'),('first','Без точек, мл'),('second','С подсказками, мл')])
    pair_rows=[]
    for ch in CHAMBERS:
        for first in ['phase_08','phase_14']:
            a,b=[next(r for r in result['phase_pairs'] if r['method']==m and r['chamber']==ch and r['first_phase_id']==first) for m in methods]
            pair_rows.append({'pair':first[-2:]+' − 10','chamber':ch,'raw':a['raw_difference_ml'],'first':a['candidate_difference_ml'],'second':b['candidate_difference_ml']})
    pair_table=table(pair_rows,[('pair','Разность фаз'),('chamber','Камера'),('raw','Исходные маски, мл'),('first','Без точек, мл'),('second','С подсказками, мл')])
    intro='''## Пилот уточнения масок Георга — 10.09.2026

**Результат вычислительного пилота:** получено 16 кандидатных масок ЛЖ/ПЖ
на фазах 04, 08, 10 и 14 двумя способами. Ни одна не принята для замены
исходных масок в основном расчёте. Это автоматическая сегментация с отдельным
визуальным просмотром, а не выполненная вручную разметка.

**Подтверждено автором:** нормальная различимость крови и миокарда в этой серии
есть только у Георга благодаря остаточному контрасту. Это основание начать с
него; точность масок из этого не следует. Фаза 04 относится к короткому
интервалу 491 мс, фазы 08/10/14 — к длинному 1159 мс. Интервалы не объединены.

**Метод и допущение:** первый вариант — локальный nnInteractive с исходной
маской TotalSegmentator. Во втором добавлены последовательные точечные
подсказки внутри камеры и соседних структур, выбранные алгоритмом из
автоматических масок; HU использованы только для выбора положительной точки.
Это не независимые ручные подсказки. Рабочая цель — оболочка полости с включением
папиллярных мышц и трабекул в объём; клапанные плоскости пока не приняты.
Предсердия в этом пилоте не исправлялись.

**Наблюдение при ограниченном просмотре:** без точек часть контуров ЛЖ
расширяется в стенку, а ПЖ — в лёгочный ствол. Подсказки уменьшают некоторые
расхождения, но в отдельных срезах ЛЖ теряет часть полости, а захват ствола
сохраняется. Все заданные точки выполнены — это не подтверждает точность всей
границы. Просмотрены по три выбранных ортогональных среза каждого кандидата;
полная экспертная проверка всех срезов не выполнена.
'''
    style='<style>.function-table{border-collapse:collapse;width:100%;font-size:14px}.function-table td,.function-table th{border:1px solid #cad1db;padding:8px;text-align:left}.function-table th{background:#edf2f7}.pilot-evidence{max-width:800px;width:100%;height:auto}</style>'
    prefix='heart_rr_analysis/function_analysis/georg_pilot_20260910/'
    links='<h3>Просмотр каждого кандидата</h3><ul>'
    for v in volume_rows:
        pid='phase_'+v['phase'];ch=v['chamber']
        a,b=[lookup[(m,pid,ch)] for m in methods]
        links+=f'<li>Фаза {v["phase"]}, {CHAMBERS[ch]}: <a href="{prefix}{a["reader_page"]}">без точек</a> · <a href="{prefix}{b["reader_page"]}">с подсказками</a></li>'
    links+='</ul>'
    observations='<details><summary>Наблюдения по выбранным срезам всех 16 кандидатов</summary><ul>'
    for row in result['rows']:
        d=by_key[(row['method'],row['phase_id'],row['chamber'])]
        observations+=f'<li><a href="{prefix}{d["evidence"]}">{html.escape(row["method"])} · {row["phase_id"][-2:]} · {CHAMBERS[row["chamber"]]}</a>: {html.escape(d["visual_observation"])}</li>'
    observations+='</ul></details>'
    body=style+'<h3>Как меняются объёмы</h3>'+volume_table+'<h3>Чувствительность межфазной разности</h3><p>Только фазы длинного R–R. Эти разности не являются принятым УО: ED/ES ещё не установлены. Фаза 14 расположена позже фазы 10; запись «14 − 10» означает только порядок вычитания.</p>'+pair_table
    body+='<p>Например, разность ЛЖ 14 − 10 меняется с 49,34 до 25,32 мл без точек. Близость второго варианта к исходной разности также не доказывает его правильности: ошибки двух масок могут частично компенсироваться.</p>'
    body+=links+observations+'<h3>Проблемные участки</h3><p>Оранжевый — исходный контур; голубой — кандидат. Слева на каждой панели КТ без контура. Здесь показаны срезы с наибольшим расхождением масок.</p>'
    for rel,caption in [('guided/phase_04_heart_ventricle_left/largest_changes.png','ЛЖ, фаза 04: потеря части полости после подсказок.'),('guided/phase_10_heart_ventricle_right/largest_changes.png','ПЖ, фаза 10: распространение в лёгочный ствол после подсказок.')]:
        body+=f'<figure><img class="pilot-evidence" src="{prefix}{rel}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
    body+='<h3>Что следует из пилота</h3><p>Улучшение масок не подтверждено. Следующий этап — задать и проверить эндокардиальные контуры и клапанные границы на этих же четырёх фазах. После сохранения отдельных исправленных масок повторить сравнение объёмов, затем отдельно принять ED/ES. Только после этих проверок возможен переход к УО/ФВ.</p>'
    body+='<p>Исходные маски и основная таблица объёмов сохранены. Dice между моделями и перекрытия с автоматическими масками миокарда/сосудов — диагностические расхождения, а не показатели точности. Пакетный технический запуск точек исключён из сравнения после предупреждения установленного API; показан только запуск с отдельным пересчётом после каждой точки.</p>'
    body+=f'<p>Данные: <a href="{prefix}pilot_comparison.csv">16 кандидатов CSV</a> · <a href="{prefix}phase_pair_sensitivity.csv">разности CSV</a> · <a href="{prefix}visual_review_decisions.json">объём и решения просмотра</a> · <a href="{prefix}pilot_comparison.json">происхождение и статусы</a>.</p>'
    reader={'schema_version':1,'status':'exploratory_hypothesis_not_validated','markdown':intro,'html':body,'comparison_sha256':sha256_file(output/'pilot_comparison.json'),'visual_decisions_sha256':sha256_file(decision_path),'raw_volume_csv_sha256':decisions['raw_volume_csv_sha256'],'accepted_candidates':0}
    (output/'final_reader.json').write_text(json.dumps(reader,ensure_ascii=False,indent=2),encoding='utf-8')
    import nbformat
    from nbconvert import HTMLExporter
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell('# Reader output',outputs=[nbformat.v4.new_output('display_data',data={'text/html':body.replace(prefix,'')},metadata={})])])
    exported,_=HTMLExporter(exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(nb)
    (output/'index.html').write_text(exported,encoding='utf-8')
    return reader


def notebook_cells(root):
    import nbformat
    path=root/'heart_rr_analysis/function_analysis/georg_pilot_20260910/final_reader.json'
    if not path.exists():return []
    reader=json.loads(path.read_text(encoding='utf-8'))
    for name,key in [('pilot_comparison.json','comparison_sha256'),('visual_review_decisions.json','visual_decisions_sha256')]:
        if sha256_file(path.parent/name)!=reader[key]:raise ValueError('Pilot reader provenance changed')
    if sha256_file(root/'heart_rr_analysis/chamber_volumes_by_rr.csv')!=reader['raw_volume_csv_sha256']:
        raise ValueError('Pilot baseline differs from current primary volume table')
    cells=[nbformat.v4.new_markdown_cell(reader['markdown']),nbformat.v4.new_code_cell('# Результаты отдельного пилота масок Георга; исходные объёмы не заменяются.',outputs=[nbformat.v4.new_output('display_data',data={'text/html':reader['html']},metadata={})])]
    for cell in cells:cell.metadata['tags']=['georg-pilot-analysis']
    return cells


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--derived-root',type=Path,required=True);p.add_argument('--first-pilot',type=Path,required=True);p.add_argument('--second-pilot',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();r=summarize(a.derived_root,a.first_pilot,a.second_pilot,a.output_dir);build_reader(r,a.output_dir);print('Compared',len(r['rows']),'candidates; phase differences',len(r['phase_pairs']))
