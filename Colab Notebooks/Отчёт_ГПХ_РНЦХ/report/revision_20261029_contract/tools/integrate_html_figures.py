"""Insert verified HTML figures into the scientific argument and renumber figures."""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[3]
DATA=ROOT/'figures/html_exports'
# Insert immediately before the following section, within the relevant topic.
PLACEMENTS={
 '01_n_ct_tissue_masks.png':('06_anatomy.md','## 6.2'),
 '02_n_candidate_lateral_montage.png':('05_experiments.md','## 5.5'),
 '03_g_ct_contour_regularization.png':('06_anatomy.md','## 6.4'),
 '04_tt_ecg_four_cycles.png':('05_experiments.md','## 5.9'),
 '05_tissue_identifiability_singular_values.png':('08_tissue.md','## 8.4'),
 '06_measured_pulse_vs_volume_probe.png':('09_heart_dynamics.md','## 9.14'),
 '07_ct_whole_heart_and_chambers.png':('10_chambers.md','## 10.4'),
 '08_respiratory_contrast_observation_model.png':('08_tissue.md','## 8.4'),
}

def main():
    receipt=ROOT/'reviews/html_figure_integration.json'
    if receipt.exists():raise RuntimeError('Figures already integrated')
    d=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    integrated=[]
    for i,f in enumerate(d['figures'],1):
        name=f['filename']; chapter,anchor=PLACEMENTS[name]
        picture=DATA/name
        if hashlib.sha256(picture.read_bytes()).hexdigest()!=f['output_sha256']:raise ValueError(name)
        source=REPO/f['source_path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=f['source_html_sha256']:raise ValueError('HTML changed: '+name)
        p=ROOT/'chapters'/chapter;t=p.read_text(encoding='utf-8')
        if anchor not in t:raise ValueError('Missing insertion anchor '+anchor)
        n=int(chapter[:2]);label=f'{n}.{90+i}'
        source_head=re.search(r'^## Источники к главе[^\n]*$',t,re.M)
        if not source_head:raise ValueError('Missing bibliography')
        used=[int(x) for x in re.findall(r'^(\d+)\.\s',t[source_head.end():],re.M)]
        ref=max(used,default=0)+1
        url=Path(os.path.relpath(source,p.parent)).as_posix()
        t+='\n'+f'{ref}. [Сохранённый HTML-отчёт: {f["source_figure"]}]({url}). Локатор: элемент изображения {f["image_index_zero_based"]}, индексация от нуля. Условия и ограничения сохранены в подписи рисунка.\n'
        intro=f'На рисунке {label} приведён дополнительный результат из сохранённого расчётного отчёта [{ref}].'
        block=f'\n{intro}\n\n![{f["title"]}](../figures/html_exports/{name})\n\nРисунок {label} — {f["title"]}\n\n{f["caption"]}\n\n'
        extra=' '.join(f.get('limitations',[]))
        if extra:block+=extra+'\n\n'
        t=t.replace(anchor,block+anchor,1)
        p.write_text(t,encoding='utf-8')
        integrated.append({'id':f['id'],'file':name,'chapter_before_permutation':chapter,'temporary_label':label,'source_id':f['source_id'],'source_sha256':f['source_html_sha256'],'output_sha256':f['output_sha256']})
    figure_map={}
    for p in sorted((ROOT/'chapters').glob('*.md')):
        t=p.read_text(encoding='utf-8')
        labels=re.findall(r'^Рисунок\s+(\d+\.\d+)\s+[—–-]',t,re.M)
        for i,old in enumerate(labels,1):figure_map[old]=old.split('.')[0]+'.'+str(i)
    # Replace only references explicitly designated as figures, in one pass.
    paths=list((ROOT/'chapters').glob('*.md'))+[ROOT/'front_matter.md',ROOT/'conclusion.md',ROOT/'appendices.md']
    for p in paths:
        t=p.read_text(encoding='utf-8')
        t=re.sub(r'(?i)((?:рисунок|рисунке|рисунка|рисунки|рисунков|рис\.)\s+)(\d+\.\d+)',lambda m:m[1]+figure_map.get(m[2],m[2]),t)
        p.write_text(t,encoding='utf-8')
    for f in integrated:f['label_before_chapter_permutation']=figure_map[f['temporary_label']]
    receipt.write_text(json.dumps({'integrated':integrated,'figure_number_mapping_before_chapter_permutation':figure_map},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'added_figures':len(integrated),'total_figure_captions':len(figure_map)},ensure_ascii=False))

if __name__=='__main__':main()
