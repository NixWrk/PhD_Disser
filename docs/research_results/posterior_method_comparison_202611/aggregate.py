"""Общие ошибки прогноза 200 мм; только стандартная библиотека Python.

Входы остаются локальными: comparison.csv и соседний primary_results.md.
Выход содержит только 11 агрегатов по четырём условиям, без исходных строк.
Новые прямые или обратные расчёты не выполняются.
"""
import argparse
import csv
import math
from pathlib import Path


# Названия соответствуют основной таблице; идентификаторы служат только отбору.
METHODS = {
    'two_layer_effective': ('flat', 'Плоская, два слоя, общее ρ'),
    'two_layer_series_q': ('flat', 'Плоская, два слоя, последовательная смесь по КТ'),
    'two_layer_parallel_q': ('flat', 'Плоская, два слоя, параллельная смесь по КТ'),
    'three_layer_fat_soft_q': ('flat', 'Плоская, три слоя, жир, мягкие ткани и лёгкое'),
    'fat_soft_full_h': ('flat', 'Плоская, те же слои с полной толщиной стенки'),
    'soft_fat_gapdrop': ('flat', 'Плоская, три слоя в порядке мягкие ткани, жир и лёгкое'),
    'three_layer_finite_lung_shared_wall': ('flat', 'Плоская, три слоя с конечной толщиной лёгкого'),
    'historical_halfspace60_then_two_layer': ('flat', 'Плоская, на 60 мм однородная среда, далее два слоя'),
    'pure_fem_common': ('fem', 'FEM, общее ρ мягких тканей и жира'),
    'pure_fem_separate_q': ('fem', 'FEM, разные ρ мягких тканей и жира'),
    'mixed_flat60_fem': ('mixed', 'Смешанная, однородная среда на 60 мм, далее FEM'),
}
TYPES = {'flat': 'Плоский аналитический', 'fem': 'Трёхмерный конечно-элементный', 'mixed': 'Смешанный'}
GEOMETRY = {'flat': 'ap_w10_volume', 'fem': 'anatomical_ct', 'mixed': 'anatomical_ct'}
OUTPUT = Path(__file__).with_name('aggregate.csv')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def compute(rows, reference_text):
    selected = [r for r in rows if float(r['lung_upper_ohm_m']) == 17]
    require(len(selected) == 44, 'Для основного варианта ожидаются ровно 44 строки.')
    require({r['model'] for r in selected} == set(METHODS), 'Состав 11 способов изменён.')
    conditions = {(r['participant'], r['posture']) for r in selected}
    require(len(conditions) == 4 and len({a for a, _ in conditions}) == 2
            and len({b for _, b in conditions}) == 2, 'Нарушен состав четырёх условий.')
    heading = '## Ошибка прогноза на 200 мм, %'
    require(reference_text.count(heading) == 1, 'Нет однозначной основной таблицы прогноза.')
    section = reference_text.split(heading, 1)[1].split('\n## ', 1)[0]
    reference = {}
    titles = {title for _, title in METHODS.values()}
    for line in section.splitlines():
        cells = [s.strip() for s in line.strip().strip('|').split('|')]
        if cells[0] in titles:
            require(len(cells) == 5 and cells[0] not in reference, 'Изменена структура эталонной строки.')
            reference[cells[0]] = [float(s.replace('−', '-').replace(',', '.')) for s in cells[1:]]
    require(set(reference) == titles, 'Основная таблица должна содержать все 11 способов.')
    result, observed_by_condition, common_order = [], {}, None
    for model, (family, title) in METHODS.items():
        group = [r for r in selected if r['model'] == model]
        order = [(r['participant'], r['posture']) for r in group]
        require(len(group) == 4 and set(order) == conditions, 'Пропуск или повтор условия внутри способа.')
        if common_order is None:
            common_order = order
        require(order == common_order, 'Порядок условий должен совпадать с основной таблицей.')
        errors, percentages = [], []
        for r, condition in zip(group, order):
            require(r['family'] == family and r['ct_geometry'] == GEOMETRY[family], 'Выбран другой тип расчёта или вариант КТ.')
            measured, predicted = float(r['observed200_ohm']), float(r['predicted200_ohm'])
            require(all(math.isfinite(x) and x > 0 for x in (measured, predicted)), 'Неверный импеданс.')
            require(condition not in observed_by_condition or measured == observed_by_condition[condition],
                    'Способы используют разные контрольные измерения.')
            observed_by_condition[condition] = measured
            error, percent = predicted - measured, 100 * (predicted - measured) / measured
            require(math.isclose(error, float(r['error200_ohm']), rel_tol=0, abs_tol=1e-9)
                    and math.isclose(percent, float(r['error200_percent']), rel_tol=0, abs_tol=1e-8),
                    'Ошибка в исходной таблице не воспроизводится из прогноза и измерения.')
            errors.append(error)
            percentages.append(percent)
        # Эталон публикует проценты с одним десятичным знаком; расчёт агрегатов
        # ниже использует полную точность исходного CSV, без этого округления.
        require([float(f'{p:.1f}') for p in percentages] == reference[title],
                'Прогнозы не совпадают с 44 значениями основной таблицы при её округлении.')
        metrics = (math.fsum(abs(e) for e in errors) / 4,
                   math.sqrt(math.fsum(e * e for e in errors) / 4),
                   math.fsum(abs(p) for p in percentages) / 4)
        result.append(dict(calculation_type=TYPES[family], method=title, n_conditions=4,
                           MAE_ohm=format(metrics[0], '.3f'), RMSE_ohm=format(metrics[1], '.3f'),
                           mean_abs_percent=format(metrics[2], '.2f')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--reference', type=Path, help='По умолчанию соседний primary_results.md.')
    parser.add_argument('--check', action='store_true', help='Сверить aggregate.csv без записи.')
    args = parser.parse_args()
    reference = args.reference or args.comparison.with_name('primary_results.md')
    rows = compute(read_csv(args.comparison), reference.read_text(encoding='utf-8-sig'))
    if args.check:
        require(read_csv(OUTPUT) == [{k: str(v) for k, v in row.items()} for row in rows], 'Агрегаты не воспроизвелись.')
    else:
        with OUTPUT.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print('Сверены 44 исходные строки и 11 агрегатов по четырём условиям.')


if __name__ == '__main__':
    main()
