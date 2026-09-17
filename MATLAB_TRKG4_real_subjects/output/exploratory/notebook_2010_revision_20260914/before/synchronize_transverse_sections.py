"""Insert shared current section while retaining explicitly historical calculations."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def apply_current_section(cells):
    if any(c.metadata.get('shared_transverse_current') for c in cells):return cells
    sys.path.insert(0,str(ROOT/'tools'))
    from shared_transverse_section import shared_cells
    start=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and '### 11.3.' in c.source)
    end=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## 12.'))
    # Only the old comparison retains the old namespace, montage and saved outputs.
    for c in cells[start+1:end]:
        c.source=c.source.replace('11.3.1','11.5.1').replace('### 11.4.','### 11.5.2.')
    additions=shared_cells()
    for c in additions:c.metadata['shared_transverse_current']=True
    legacy=nbf.v4.new_markdown_cell('''### 11.5. Историческое сравнение другого положения сборки

**Сохранено для прослеживаемости прежних расчётов.** Следующие срезы, толщины и подгонки относятся к раннему локальному результату, описанному в разделах 1–10. Центр отличается от C01 на 64,27 мм; направление глубины также строилось иначе. Центральная глубина около 30,9 мм не используется в текущем согласованном сравнении 11.3.1–11.4 или в 20.16. Исходная многотканная FEM-задача и её отдельный двухтканный контроль сохраняют свои собственные сопротивления и контакты.

Этот исторический материал объясняет, как возникли проверяемые геометрические гипотезы. Количественные результаты разных положений не объединяются в один опыт.''')
    legacy.metadata['shared_transverse_current']=True
    cells[start+1:start+1]=additions+[legacy]
    cells[0].source+='\n\n**Согласование с 20.16 от 14.09.2026.** Текущие срезы и прямое сравнение в 11.3.1–11.4 используют выбранный C01 и тот же повторно рассчитанный двухтканный набор, что и 20.16. Ранний пример другого положения сохранён отдельно в 11.5.'
    return cells
