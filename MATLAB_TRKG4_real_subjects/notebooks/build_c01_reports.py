"""Single producer for C01 geometry, validation, comparison and half-space appendices."""
from pathlib import Path
import argparse,json,hashlib,sys
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from c01_report_text import cells_for,NAMES
from build_c01_anatomy_overview import build as build_anatomy_overview, FIGURE as OVERVIEW_FIGURE, MANIFEST as OVERVIEW_MANIFEST
ROOT=Path(__file__).resolve().parents[2]
PIPE=ROOT/'MATLAB_TRKG4_real_subjects'
OUT=PIPE/'output/exploratory/c01_baseline_models_20260914'
SOURCE=PIPE/'output/exploratory/transverse_consistency_20260914'
HALF=PIPE/'output/exploratory/halfspace_bem_20260914'
REVIEW=PIPE/'output/exploratory/baseline_model_comparison_20260914'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(which=None):
    files=['geometry.json','geometry_volumes.csv','geometry_mask_qc.csv','fem_acceptance.json',
      'analytic_identity.json','analysis_summary.json','inverse_recovery.csv','forward_errors.csv',
      'experimental_fits.csv','experimental_perturbations.csv','conditional_holdouts.csv','cem_fit_summary.json','cem_baseline.csv','library_timings.csv',
      'analytic_quadrature_qc.csv','pem_library.csv','reference_holdouts.csv','sensitivity_atlas.csv','sensitivity_atlas_summary.json']
    expected={p:sha(OUT/p) for p in files}
    source_files=[SOURCE/'figures/03_four_models_3d.json',SOURCE/'cem_inverse.csv',SOURCE/'geometry_control_recovery.csv',
      SOURCE/'cem_baseline.csv',SOURCE/'geometry_volumes.csv',HALF/'fem_benchmark.csv',HALF/'direct_timings.csv']
    source_expected={p.relative_to(PIPE).as_posix():sha(p) for p in source_files}
    visual_record=None
    visual_expected={}
    if which is None or which=='20.10':
        visual_record=build_anatomy_overview()
        visual_expected={
            OVERVIEW_FIGURE.relative_to(PIPE).as_posix():sha(OVERVIEW_FIGURE),
            OVERVIEW_MANIFEST.relative_to(PIPE).as_posix():sha(OVERVIEW_MANIFEST),
        }
    setup="""from pathlib import Path
import sys,json,hashlib
project = next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'MATLAB_TRKG4_real_subjects').is_dir())
sys.path.insert(0,str(project/'MATLAB_TRKG4_real_subjects/tools'))
from c01_report_tables import *
expected = """+repr(expected)+"""
source_expected = """+repr(source_expected)+"""
for name,digest in expected.items():
    with (OUT/name).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, 'Изменён расчётный вход: '+name
for name,digest in source_expected.items():
    with (ROOT/name).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, 'Изменён внешний расчётный вход: '+name
geometry = checked()
assert geometry['candidate']=='C01'
"""
    for key in [which] if which else NAMES:
        key_setup=setup
        study={'candidate':'C01','scope':'baseline_resistivity_comparison','status':'exploratory_hypothesis_not_validated',
          'source_directory':OUT.relative_to(PIPE).as_posix(),'input_sha256':expected,'source_input_sha256':source_expected,'generator_sha256':sha(Path(__file__)),
          'reader_source_sha256':sha(Path(__file__).with_name('c01_report_text.py')),'table_source_sha256':sha(PIPE/'tools/c01_report_tables.py'),
          'sensitivity_atlas_source_sha256':sha(PIPE/'tools/c01_sensitivity_atlas.py'),'sensitivity_figure_source_sha256':sha(PIPE/'tools/c01_sensitivity_figure.py'),
          'model_catalog_source_sha256':sha(PIPE/'tools/c01_model_catalog.py')}
        if key=='20.10':
            assert visual_record is not None and visual_record['new_FEM_solutions']==0
            key_setup+='\noverview_expected = '+repr(visual_expected)+'\n'
            study['visualization_input_sha256']=visual_expected
            study['visualization_new_FEM_solutions']=0
        nb=nbf.v4.new_notebook(cells=cells_for(key,key_setup),metadata={
          'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'study':study})
        NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        dest=ROOT/'Colab Notebooks'/(NAMES[key]+'.ipynb');nbf.write(nb,dest)
        ex=HTMLExporter();ex.exclude_input=True;ex.exclude_input_prompt=True;ex.exclude_output_prompt=True
        html,_=ex.from_notebook_node(nb)
        html_tmp=dest.with_suffix('.html.pending')
        html_tmp.write_text(html,encoding='utf-8');html_tmp.replace(dest.with_suffix('.html'))
        heading=nb.cells[0].source.splitlines()[0]
        passport=heading+'\n\n[Выполненный ноутбук]('+dest.name+') · [HTML без кода]('+dest.with_suffix('.html').name+')\n\n'
        roles={'20.10':'Геометрия C01, происхождение выбора и текущий прямой контроль.', '20.13':'Геометрические правила и новые маски C01.','20.14':'Две проверки: известные синтетические сопротивления и реальные измерения.',
               '20.16':'Основное сравнение шести заданных моделей восстановления базовых сопротивлений относительно КТ/FEM-референса. Двенадцать дополнительных реализаций отделены как диагностические проверки допущений; приведены результаты на 65 состояниях и ограничения выбора методики.',
               '20.17':'Математическое обоснование и сходимость двух включений в полупространстве.'}
        passport+=roles[key]+' Все текущие базовые сравнения используют единый набор C01. h фиксирована по КТ; пульсовая задача отложена. Результаты являются этапом разработки методики.\n'
        passport+='\n[Общее сравнение](20.16_Поперечная_форма_лёгкого_и_точность_обратной_модели.ipynb). Прежние полные редакции сохранены в archive/private/20260914_c01_migration_before.\n'
        dest.with_suffix('.md').write_text(passport.replace('сравнение19','сравнение 19').replace('на65','на 65'),encoding='utf-8')
        REVIEW.mkdir(exist_ok=True)
        record={'notebook':dest.name,'executions':[c.execution_count for c in nb.cells if c.cell_type=='code'],
           'input_sha256':expected,'artifact_sha256':{p.name:sha(p) for p in [dest,dest.with_suffix('.html'),dest.with_suffix('.md')]}}
        (REVIEW/f'execution_{key}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(key,'executed',record['executions'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--notebook',choices=list(NAMES));a=p.parse_args();build(a.notebook)
