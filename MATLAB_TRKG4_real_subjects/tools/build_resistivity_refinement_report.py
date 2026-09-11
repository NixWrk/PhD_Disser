"""Execute and export report 40.08; a running snapshot requires an explicit flag."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/resistivity_refinement_20260910'
TARGET=ROOT/'Colab Notebooks/40.08_Сходимость_чувствительности_по_удельному_сопротивлению.ipynb'
PROTOCOL=ROOT/'Colab Notebooks/40.08_Протокол_проверки_плотности_перебора_ρ.md'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    complete=(BASE/'analysis/refinement_summary.json').exists()
    if not complete and not args.allow_incomplete:
        raise RuntimeError('Full result missing; use --allow-incomplete only for an explicitly labelled progress snapshot')
    protocol=PROTOCOL.read_text(encoding='utf-8-sig')
    if complete:
        protocol=protocol.replace('Текущий статус — расчёт выполняется; достаточность перебора ещё не установлена.', 'Полный расчёт завершён; решение о достаточности приведено ниже по результатам зафиксированных проверок.')
        protocol=protocol.replace('Полученные результаты будут характеризовать', 'Полученные результаты характеризуют')
        protocol=protocol.replace('После завершения полного перебора научный отчёт должен содержать', 'Научный отчёт содержит').replace('До этого момента промежуточный журнал отражает только ход расчёта.', 'Журнал расчёта сохраняется отдельно от научных результатов.')
        protocol=protocol.replace(
            'Программа зафиксирована до получения новых результатов. Полный расчёт завершён; решение о достаточности приведено ниже по результатам зафиксированных проверок.',
            'Критерии проверки определены до анализа новых результатов. Расчёт завершён; решение о достаточности основано на этих критериях.'
        )
        protocol=protocol.replace(
            'Для каждой сборки и цели сохраняются знаки и экстремумы S, диапазоны B и Q, а также минимальные найденные B и Q. Два последних показателя используются как самостоятельные критерии предпочтения: наименьшая целевая чувствительность в исследованных состояниях и наименьшая избирательность. Произвольная взвешенная сумма критериев не вводится.',
            'Для каждой сборки и компартмента анализируются знак и диапазон S, а также диапазоны B и Q. Отдельно определяются минимальные B и Q по всем рассчитанным состояниям. Эти два показателя используются как самостоятельные критерии; их произвольная взвешенная сумма не вводится.'
        )
        protocol=protocol.replace('## Правило достаточности первой попытки', '## Критерии достаточности параметрической сетки')
        protocol=protocol.replace('## Воспроизводимость и последующий отчёт', '## Воспроизводимость вычислений')
        protocol=protocol.replace(
            'FEM-решения выполняются через каноническую точку входа `run_pipeline` с профилем `electrode_sensitivity`. Каждая завершённая порция содержит параметры, исходный код, контрольные суммы и результаты. Старые результаты сохраняются. Незавершённая порция не принимается как полноценный набор данных; повторная попытка получает отдельный каталог.',
            'Конечноэлементные решения выполняются функцией `run_pipeline` с профилем `electrode_sensitivity`. Вместе с результатами сохраняются параметры расчёта, версия исходного кода и контрольные суммы. В анализ включаются только полностью завершённые наборы решений.'
        )
        protocol=protocol.replace(
            'Научный отчёт содержит структуру параметрической сетки, зависимости чувствительности, карты ошибок в контрольных состояниях, парные сравнения, устойчивость минимумов и предпочтительных множеств, а также явное решение по каждому критерию. Журнал расчёта сохраняется отдельно от научных результатов.',
            'Воспроизводимый отчёт включает параметры сетки, значения чувствительности, ошибки в контрольных состояниях, парные сравнения и проверку устойчивости минимумов. Технический журнал расчёта хранится отдельно.'
        )
    nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'scientific_status':'exploratory_hypothesis_not_validated','calculation_complete':complete})
    protocol=protocol.replace('## ','### ').replace('# 40.08 Проверка достаточности перебора удельного сопротивления','## 2. Постановка, параметры и критерии')
    nb.cells=[
        nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT=Path.cwd()\nif ROOT.name=='Colab Notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/arm_sigma_20260908/python_solver_deps'))\nsys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))\nimport resistivity_refinement_report as report\nimport resistivity_refinement_interpretation as interpretation\nr=report.load_report()"),
        nbformat.v4.new_markdown_cell("# 40.08. Достаточность перебора удельного сопротивления для оценки чувствительности электродных сборок\n\n**Вычислительное исследование на фиксированной КТ-геометрии и конечноэлементной модели Nix при 50 кГц.** Работа относится к разработке методики и не имеет статуса независимой экспериментальной или клинической валидации.\n\n## Аннотация\n\nИсследование проверяет, достаточно ли пяти значений удельного сопротивления каждого из трёх целых компартментов для описания чувствительности 16 электродных сборок и их взаимного сравнения. Наряду со 125 узлами полного декартова перебора для каждой сборки рассчитаны 64 независимых промежуточных состояния. Основной результат, критерии решения и его практическое значение приведены перед подробным описанием метода."),
        nbformat.v4.new_code_cell('interpretation.report_header(r)\nreport.status(r)'),
        nbformat.v4.new_markdown_cell(protocol),
        nbformat.v4.new_markdown_cell("## 3. Структура параметрического перебора и геометрия сборок\n\nРисунок 1 необходим для проверки того, что контрольные состояния расположены между основными узлами и не используются при построении проверяемой интерполяции."),
        nbformat.v4.new_code_cell('report.design(r)\ninterpretation.design_analysis(r)'),
        nbformat.v4.new_markdown_cell("### Геометрия сравниваемых сборок\n\nНумерация из таблицы используется во всех последующих рисунках и сводках. Рисунок 2 связывает каждый номер с типом контакта и двумя расстояниями: d отсчитывается от границы туловища и модельной руки до внутреннего измерительного контакта, l — от внутреннего контакта до наружного токового электрода."),
        nbformat.v4.new_code_cell('report.assembly_table(r)\nreport.assembly_map(r,2)\ninterpretation.assembly_analysis(r)'),
        nbformat.v4.new_markdown_cell("### Цветовые матрицы чувствительности и правила их трактовки\n\nРасположение ячеек на рисунке 3 совпадает с картой сборок на рисунке 2. Верхний ряд показывает минимальное абсолютное влияние B для выбранного компартмента, нижний — минимальную избирательность Q. Числа и цвет позволяют сравнить сборки, а текст после рисунка объясняет результат и его ограничения."),
        nbformat.v4.new_code_cell('report.sensitivity_matrix(r,3)\ninterpretation.sensitivity_matrix_analysis(r)\ninterpretation.sensitivity_definitions(r)\nreport.sensitivity_metric_explanation(r,4)\ninterpretation.general_design_recommendations(r)')
    ]
    parts=[
        ('## 4. Диапазоны чувствительности\n\nПроизводная оценивает отклик на единичное изменение сопротивления целого компартмента при заданных фоновых сопротивлениях. Размах её значений по состояниям характеризует зависимость чувствительности от этих фонов. Различия геометрии электродов сравниваются при одинаковом наборе тканевых параметров.','report.derivative_ranges(r,5)\ninterpretation.derivative_analysis(r)'),
        ('## 5. Точность интерполяции производных\n\nИнтерполированные производные сравниваются с производными, рассчитанными непосредственно в контрольных состояниях. Ошибка делится на установленный допуск: значение меньше единицы означает выполнение критерия. Для каждого состояния приводится наибольшее отношение среди 16 сборок.','report.error_maps(r,"derivative",6)\nreport.error_table(r)\ninterpretation.individual_accuracy_analysis(r)'),
        ('## 6. Точность межсборочных различий\n\nДаже точная оценка каждой сборки не гарантирует точности их разности. Если различие между сборками мало, ошибка интерполяции может изменить их порядок. Поэтому ниже отдельно проверяется ошибка парной разности. Если прямое различие меньше заданной численной полосы, сборки считают неразличимыми, но ошибка их разности всё равно должна пройти допуск.','report.error_maps(r,"pairs",7)\nreport.pair_matrix(r,8)\ninterpretation.pair_accuracy_analysis(r)'),
        ('## 7. Устойчивость минимальных показателей\n\nМинимальные B и Q характеризуют наименее благоприятное из рассчитанных состояний. При добавлении новых состояний минимум может сохраниться или уменьшиться. Поэтому значения сравниваются на сетках 3³ и 5³, а затем после добавления 64 контрольных точек.','report.minima(r,"min_B",9)\nreport.minima(r,"min_Q",10)\ninterpretation.minima_analysis(r)'),
        ('### Предпочтительные множества сборок\n\nДля каждого компартмента и показателя выбираются сборки, находящиеся в заданной численной полосе около максимума. Список кандидатов сравнивается на трёх наборах состояний. Он считается устойчивым только при полном совпадении состава. Результат относится к выбранной модели и литературным диапазонам ρ.','report.selection(r)\ninterpretation.selection_analysis(r)'),
        ('## 8. Конечные изменения импеданса\n\nПроизводная S описывает локальный наклон зависимости, а конечное изменение — разность непосредственно рассчитанных Z на нижней и верхней границах диапазона ρ. Эти величины могут различаться, если зависимость нелинейна.','report.excursions(r,11)\ninterpretation.excursion_analysis(r)'),
    ]
    for text,code in (parts if complete else []):
        nb.cells.extend([nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell(code)])
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("## 9. Ограничения и границы применимости\n\nОграничения перечислены по источникам неопределённости, чтобы численная проверка параметрической сетки не подменяла пространственную, анатомическую и экспериментальную валидацию."),
        nbformat.v4.new_code_cell('interpretation.limitations(r)'),
        nbformat.v4.new_code_cell('interpretation.conclusion(r)')
    ])
    NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    nbformat.write(nb,TARGET)
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    css='<style>.jp-Notebook{max-width:1200px;margin:auto;padding:32px 40px}.jp-RenderedMarkdown p{font-size:16px;line-height:1.7}.jp-RenderedMarkdown h2{margin-top:36px}.jp-RenderedHTMLCommon table{width:100%;font-size:13px}.jp-RenderedHTMLCommon td,.jp-RenderedHTMLCommon th{padding:8px;white-space:normal}.jp-RenderedImage img{max-width:100%;height:auto}.jp-OutputArea-output{overflow-x:auto}</style>'
    html=html.replace('</head>',css+'</head>')
    soup=BeautifulSoup(html,'html.parser')
    for node in soup.select('.jp-InputArea'):
        if node.select_one('.jp-RenderedMarkdown'):node.unwrap()
        else:node.decompose()
    for node in soup.select('.jp-InputPrompt,.jp-OutputPrompt'):node.decompose()
    assert not soup.select('.jp-InputArea')
    assert len(soup.select('.jp-RenderedImage img'))==(11 if complete else 2)
    TARGET.with_suffix('.html').write_text(str(soup),encoding='utf-8')
    folder=BASE/'report';folder.mkdir(exist_ok=True)
    reader=[]
    for cell in nb.cells:
        if cell.cell_type=='markdown':reader.append(cell.source)
        for output in cell.get('outputs',[]):
            data=output.get('data',{})
            if 'text/markdown' in data:reader.append(data['text/markdown'])
            elif 'text/html' in data:reader.append(BeautifulSoup(data['text/html'],'html.parser').get_text(' ',strip=True))
    reader='\n\n'.join(reader)+'\n'
    (folder/'reader_text.md').write_text(reader,encoding='utf-8')
    validation={'calculation_complete':complete,'notebook_cells':len(nb.cells),'errors':len(errors),'figures':len(soup.select('.jp-RenderedImage img')),'tables':len(soup.select('table')),'input_nodes':len(soup.select('.jp-InputArea')),'reader_sha256':hashlib.sha256(reader.encode()).hexdigest(),'scientific_status':'exploratory_hypothesis_not_validated','language_review':'pending_full_final_model_review','paths':{'notebook':str(TARGET.relative_to(ROOT)),'html':str(TARGET.with_suffix('.html').relative_to(ROOT))}}
    (folder/'render_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
