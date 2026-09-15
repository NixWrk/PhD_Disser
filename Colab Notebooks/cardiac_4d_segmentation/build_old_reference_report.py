"""Build the OLD annotation checkpoint from saved local drafts, without inference.

This report cannot accept a reference or evaluate TotalSegmentator accuracy.
The result directory contains local medical-derived data and stays out of Git.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import nbformat as nbf
from nbconvert import HTMLExporter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEM = '21.09_Референсная_разметка_исторической_КТ'


def build(root, output_dir):
    root, output_dir = Path(root), Path(output_dir)
    source = Path(__file__).with_name(STEM+'.md')
    review = json.loads((root/'body_review/anatomical_review.json').read_text(encoding='utf-8'))
    if review['expert_accepted'] or review['TS_accuracy_estimated']:
        raise ValueError('This checkpoint is only for unaccepted partial drafts')
    folders = {'Левый желудочек':'sequence02_phase00_LV_body_v03',
               'Правый желудочек':'sequence02_phase00_RV_body_v02'}
    rows = []
    for label, folder in folders.items():
        directory = root/'agent_drafts'/folder
        receipt = json.loads((directory/'draft_provenance.json').read_text(encoding='utf-8'))
        if receipt['accepted'] or receipt['rules'].get('whole_chamber') is not False:
            raise ValueError('Expected explicitly partial, unaccepted chamber drafts')
        for threshold, mask in receipt['masks'].items():
            if hashlib.sha256((directory/mask['file']).read_bytes()).hexdigest() != mask['sha256']:
                raise ValueError('Mask hash mismatch')
            rows.append({'chamber':label,'lower_hu':float(threshold),'partial_volume_ml':mask['volume_ml'],
                         'accepted':False,'whole_chamber':False})
    output_dir.mkdir(parents=True,exist_ok=True)
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def picture(relative, caption):
        path=root/relative
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        cells.append(nbf.v4.new_markdown_cell('![Иллюстрация](attachment:figure.jpg)',
            attachments={'figure.jpg':{'image/jpeg':encoded}}))
        md(caption)
    protocol = source.read_text(encoding='utf-8')
    for section in protocol.split('\n## '):
        md(section if section.startswith('# ') else '## '+section)
        if section.startswith('Первые сохранённые контуры'):
            picture('body_review/oblique_partial_boundaries.jpg',
                '**Рисунок 1. Границы текущей разметки в двух косых продольных плоскостях.** '
                'Зелёный контур показывает размеченную часть левого желудочка, голубой — правого. '
                'Прямой верхний край особенно ясно виден справа: разметка заканчивается внутри '
                'продолжающейся полости. Это граница выполненной работы, а не найденная клапанная плоскость. '
                'Шкалы заданы в миллиметрах вдоль выбранных плоскостей. Порог 190 HU служит одним '
                'из исследованных вариантов; его оптимальность не установлена.')
            picture('thin_coarse_same_physical_planes.jpg',
                '**Рисунок 2. Толстосрезовая и тонкосрезовая реконструкции на одинаковых физических плоскостях.** '
                'Слева показана реконструкция толщиной 5 мм с шагом 2,5 мм; справа — 0,5 мм с шагом '
                '0,25 мм. Тонкие срезы позволяют различать более мелкую структуру, но изображение '
                'не является проверенным пространственным эталоном для левой колонки. Совмещение '
                'анатомии отдельно не выполнялось. Эта иллюстрация не обосновывает перенос масок между реконструкциями.')
            md('### Как зависит частичный объём от порога интенсивности\n\n'
               'Таблица содержит объёмы **только размеченных частей**, в одной фазе. '
               'Ни одно из этих чисел не является полным объёмом камеры.\n\n'
               '| Нижний порог, HU | Часть ЛЖ, мл | Часть ПЖ, мл |\n|---:|---:|---:|\n'+
               '\n'.join(f'| {threshold} | '+ ' | '.join(f'{next(r["partial_volume_ml"] for r in rows if r["chamber"]==label and r["lower_hu"]==threshold):.2f}' for label in folders)+' |' for threshold in [170,190,210]))
            fig, ax = plt.subplots(figsize=(8,4.6),layout='constrained')
            differences=[]
            for label,color in zip(folders,['#257c3f','#0879b5']):
                subset=sorted([r for r in rows if r['chamber']==label],key=lambda r:r['lower_hu'])
                xs=[r['lower_hu'] for r in subset];ys=[r['partial_volume_ml'] for r in subset]
                ax.plot(xs,ys,'o-',label=label+' — часть',color=color)
                differences.append((ys[0]-ys[-1],100*(ys[0]-ys[-1])/ys[1]))
            ax.set_xlabel('Нижний порог интенсивности, HU');ax.set_ylabel('Объём размеченной части, мл')
            ax.set_xticks([170,190,210]);ax.grid(alpha=.25);ax.legend()
            buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140);plt.close(fig)
            cells.append(nbf.v4.new_markdown_cell('![Зависимость частичного объёма](attachment:volume.png)',
                attachments={'volume.png':{'image/png':base64.b64encode(buffer.getvalue()).decode('ascii')}}))
            md('**Рисунок 3. Зависимость черновой разметки от нижнего порога HU.** '
               f'При повышении порога от 170 до 210 HU размеченный объём ЛЖ уменьшается на {differences[0][0]:.2f} мл '
               f'({differences[0][1]:.2f}% от варианта 190 HU), ПЖ — на {differences[1][0]:.2f} мл '
               f'({differences[1][1]:.2f}%). Сравнивается один и тот же снимок и один набор опорных контуров: '
               'изменение вызвано способом выделения вокселей. Оно показывает, почему хорошая яркость '
               'крови не устраняет неопределённость разметки. Эти проценты нельзя переносить на полный '
               'объём камеры, на другие фазы или считать ошибкой TotalSegmentator.')
    md('## Послойный контроль сохранённых масок\n\n'
       'Ниже приведены все аксиальные плоскости текущей разметки и соседние плоскости выше её границы. '
       'Индекс k обозначает номер плоскости исходного массива КТ, начиная с нуля. '
       'Левый желудочек отмечен зелёным, правый — голубым. Внутренние пробелы отражают результат '
       'пороговой обработки; их нельзя автоматически считать доказанными границами конкретных мышечных структур.')
    for page, extent in [(1,'4–11'),(2,'12–19'),(3,'20–27')]:
        picture(f'body_review/axial_page{page}.jpg',
            f'**Рисунок {page+3}. Аксиальные плоскости {extent}.** '+
            ('На плоскостях 25–27 маски отсутствуют, хотя анатомия продолжается: полная камера ещё не размечена.'
             if page==3 else 'Контуры позволяют проверить отделение размеченных областей от стенок и межжелудочковой перегородки. Независимая экспертная оценка границ ещё не выполнена.'))
    md('## Практический итог\n\n'
       'Получены редактируемые частичные маски двух желудочков и проверен их экспорт без изменения '
       'вокселей и физических координат. Ошибка TotalSegmentator пока не измерена. Следующий '
       'содержательный шаг — завершить анатомические границы камер у клапанов и выходных трактов, '
       'после чего перейти к другим фазам и независимому принятию разметки.\n\n'
       'Локальные данные находятся в папке результатов исследования: проекты четырёх фаз — в `pilot_02`, '
       'первые редактируемые черновики — в `body_review/slicer`, исходные точки и варианты масок — '
       'в `agent_drafts`. Файл `ОТКРЫТЬ_ЧЕРНОВИКИ.cmd` в `body_review` открывает подготовленную сцену. '
       'Эта сцена предназначена для просмотра и исправлений; сохранение в ней не означает принятия референса.')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'language_info':{'name':'python'},
        'old_reference':{'accepted':False,'whole_chamber_reference_available':False,
            'protocol_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    nbf.validate(notebook)
    path=output_dir/(STEM+'.ipynb');nbf.write(notebook,path)
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(notebook)
    (output_dir/(STEM+'.html')).write_text(html,encoding='utf-8')
    (root/'body_review/report_numbers.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Built {len(cells)} cells; partial annotation checkpoint, no accepted reference')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();build(args.results,args.output_dir)
