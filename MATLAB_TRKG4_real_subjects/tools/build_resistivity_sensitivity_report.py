"""Build, execute, and export scientific report 40.07 from sealed results."""
from pathlib import Path
import argparse,re,json,sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[2]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/resistivity_sensitivity_20260910'
TARGET=ROOT/'Colab Notebooks/40.07_Чувствительность_сборок_по_удельному_сопротивлению.ipynb'
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--export-only',action='store_true');args=parser.parse_args()
    if args.export_only:nb=nbformat.read(TARGET,as_version=4)
    else:
        nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'scientific_status':'exploratory_hypothesis_not_validated'})
        nb.cells=[nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT=Path.cwd()\nif ROOT.name=='Colab Notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/arm_sigma_20260908/python_solver_deps'))\nsys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))\nimport resistivity_report_figures as report\nr=report.load_report(ROOT)")]
        pieces=re.split(r'<!-- CODE:(\w+) -->',TOOLS.joinpath('resistivity_report_ru.md').read_text(encoding='utf-8'))
        for i,piece in enumerate(pieces):
            if not piece.strip():continue
            nb.cells.append(nbformat.v4.new_code_cell(f'report.{piece}(r)') if i%2 else nbformat.v4.new_markdown_cell(piece.strip()))
        NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(nb,TARGET)
    assert not [o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    css='''<style>
body { background:#fff; color:#20242a; }
.jp-Notebook { max-width:1160px; margin:auto; padding:30px 38px 70px; }
.jp-MarkdownCell p, .jp-RenderedMarkdown p { font-size:16px; line-height:1.72; }
.jp-RenderedMarkdown h1 { font-size:30px; line-height:1.3; }
.jp-RenderedMarkdown h2 { font-size:24px; margin-top:40px; }
.jp-RenderedMarkdown h3 { font-size:19px; margin-top:25px; }
.jp-RenderedHTMLCommon table { font-size:13px; width:100%; border-collapse:collapse; }
.jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { padding:9px 10px; border-bottom:1px solid #dae0e6; white-space:normal; }
.jp-RenderedHTMLCommon th { background:#eaf0f5; text-align:left; }
.jp-OutputArea-output { overflow-x:auto; }
.jp-RenderedImage img { max-width:100%; height:auto; }
.jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt { display:none; }
@media print { .jp-Notebook {padding:0;} h2,h3 {break-after:avoid;} img,table {break-inside:avoid;} }
</style>'''
    html=html.replace('</head>',css+'</head>')
    from bs4 import BeautifulSoup
    cleaned=BeautifulSoup(html,'html.parser')
    for node in cleaned.select('.jp-InputArea'):
        if node.select_one('.jp-RenderedMarkdown'):node.unwrap()
        else:node.decompose()
    for node in cleaned.select('.jp-InputPrompt,.jp-OutputPrompt'):node.decompose()
    html=str(cleaned)
    TARGET.with_suffix('.html').write_text(html,encoding='utf-8')
    report=BASE/'report';report.mkdir(exist_ok=True)
    from bs4 import BeautifulSoup
    reader=[]
    for c in nb.cells:
        if c.cell_type=='markdown':reader.append(c.source)
        else:
            for o in c.get('outputs',[]):
                data=o.get('data',{})
                if 'text/markdown' in data:reader.append(data['text/markdown'])
                elif 'text/html' in data:reader.append(BeautifulSoup(data['text/html'],'html.parser').get_text(' ',strip=True))
    (report/'reader_text.md').write_text('\n\n'.join(reader)+'\n',encoding='utf-8')
    soup=BeautifulSoup(html,'html.parser')
    validation={'notebook_cells':len(nb.cells),'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),'errors':0,'figures':len(soup.select('.jp-RenderedImage img')),'tables':len(soup.select('table')),'input_nodes':len(soup.select('.jp-InputArea')),'reader_words':len(' '.join(reader).split()),'html_bytes':len(html.encode())}
    assert len(soup.get_text(' ',strip=True))>35000, 'Reader narrative missing from HTML'
    assert validation['figures']==14 and validation['tables']==7 and validation['input_nodes']==0,validation
    (report/'render_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
