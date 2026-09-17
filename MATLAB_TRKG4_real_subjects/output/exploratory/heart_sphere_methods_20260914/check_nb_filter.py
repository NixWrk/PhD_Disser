import subprocess,json
from pathlib import Path
p='Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'
a=Path(p).read_bytes(); b=subprocess.check_output(['git','show','HEAD:'+p])
x=json.loads(a); y=json.loads(b)
print(json.dumps({'working_bytes':len(a),'committed_bytes':len(b),'working_cells':len(x['cells']),'committed_cells':len(y['cells']),'working_attachments':sum(len(c.get('attachments',{})) for c in x['cells']),'committed_attachments':sum(len(c.get('attachments',{})) for c in y['cells']),'metadata_working':x['metadata'],'metadata_committed':y['metadata'],'reader_sources_identical':[c['source'] for c in x['cells']]==[c['source'] for c in y['cells']]},ensure_ascii=False,indent=2))
