"""Execute and export the canonical 20.17 report from verified result tables."""
from pathlib import Path
import argparse
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from halfspace_report_text import cells,passport_text

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914'
NAME='20.17_Полупространство_и_точность_восстановления_тканевых_параметров'
REQUIRED=['experiment_contract.json','inverse_recovery.csv','jacobian_comparison.csv',
          'linked_scenarios.csv','placement_inverse.csv','pulse_nonlinear_inverse.csv',
          'direct_timings.csv','fem_benchmark.csv','cem_baseline_check.csv']
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--no-execute',action='store_true')
    args=ap.parse_args()
    for name in REQUIRED:
        if not (OUT/name).is_file():raise FileNotFoundError(OUT/name)
    nb=nbf.v4.new_notebook(cells=cells(),metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}})
    if not args.no_execute:
        NotebookClient(nb,timeout=180,kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
    dest=ROOT/'Colab Notebooks'
    nbf.write(nb,dest/(NAME+'.ipynb'))
    exporter=HTMLExporter()
    exporter.exclude_input=True
    exporter.exclude_input_prompt=True
    exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    (dest/(NAME+'.html')).write_text(html,encoding='utf-8')
    (dest/(NAME+'.md')).write_text(passport_text(),encoding='utf-8')
    print({'notebook':str(dest/(NAME+'.ipynb')),'executions':
           [c.execution_count for c in nb.cells if c.cell_type=='code']})
if __name__=='__main__':main()
