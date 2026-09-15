"""Remove redundant scientific views without changing numerical inputs."""
import re

def apply(nb):
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            cell.source=cell.source.replace('Текущие срезы и прямое сравнение в 11.3.1–11.4', 'Определения геометрии и прямое сравнение в 11.3.1–11.4')
            cell.source=cell.source.replace('Следующие срезы, толщины и подгонки', 'Следующие профили, толщины и подгонки')
        src=cell.source
        if cell.cell_type == 'code' and 'surface_fig = go.Figure()' in src:
            start=src.index('# A separate exterior-only view')
            end=src.index('# A deliberately simple local 3-D diagram',start)
            cell.source=src[:start]+src[end:]
            cell.outputs=[o for o in cell.get('outputs',[]) if 'отдельный контрастный вид без внутренних органов' not in str(o.get('data',{}))]
        elif cell.cell_type == 'code' and 'from shared_transverse_section import display_geometry, display_direct' in src:
            cell.source=src.replace('display_geometry, display_direct','OUT as _shared_out, checked as _shared_checked, display_direct').replace('display_geometry()', """import json as _shared_json
_shared_manifest = _shared_checked()
_shared_contract = _shared_json.loads((_shared_out/'input_contract.json').read_text(encoding='utf8'))
display(Markdown(f"Центр: **({_shared_contract['centre_mm'][0]:.3f}; {_shared_contract['centre_mm'][1]:.3f}; {_shared_contract['centre_mm'][2]:.3f}) мм**. Направленная центральная толщина: **{_shared_manifest['h_centre_mm']:.2f} мм**. Эти значения читаются из того же расчётного набора, что и 20.16."))""")
            cell.outputs=[o for o in cell.get('outputs',[]) if 'image/png' not in o.get('data',{})]
        elif cell.cell_type == 'markdown' and '**Рисунок 11.3а.' in src:
            cell.source=r"""Геометрическая иллюстрация текущего положения приведена один раз в разделе 4; объёмные варианты упрощения сравниваются в 20.16. В настоящем разделе существенны определения границ и электрических операторов. Направленная центральная толщина в расчётном базисе C01 равна 67,00 мм. Она не является кратчайшим расстоянием до лёгкого и не обозначает толщину одних мышц.

При отсутствии пересечения луча с лёгким сохраняется однородный предел. Дополнительные пересечения замкнутого лёгочного объёма не входят в эвристическое ядро первой границы. Следовательно, более подробный профиль сам по себе ещё не гарантирует правильного электрического поля. Это ограничение проверяется следующим прямым сравнением при одинаковых заданных сопротивлениях."""
        elif cell.cell_type == 'code' and '# Two-column visual comparison requested:' in src:
            a=src.index('display(pd.DataFrame(');b=src.index('# Row 1:',a)
            cell.source=("# Численные определения исторического примера; повторная схема среза исключена.\n"
                "x_visual = np.linspace(-80, 80, 401)\nh_constant = float(h_centre_surface_mm)\n"
                +src[a:b]+"\nh_flat_visual = np.polynomial.polynomial.polyval(x_visual / 70.0, coefficients)\n"
                "lung_flat_branch_profiles = lung_branch_profiles - skin_profile[:, None]\n")
            cell.outputs=[o for o in cell.get('outputs',[]) if 'text/html' in o.get('data',{})]
    return nb
