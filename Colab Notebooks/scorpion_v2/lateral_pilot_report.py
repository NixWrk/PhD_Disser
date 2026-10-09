"""Local reader view for the candidate-input calculation owned by 33.20."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


def chart(fig):
    fig.update_layout(template='plotly_white', height=450, margin=dict(t=65, b=45),
                      legend=dict(orientation='h', y=-.2))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={'responsive': True, 'displaylogo': False})


def quicklook(out, primary, ensembles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for channel in (1, 2):
        ax = axes[0, channel - 1]
        for state in ('inhale', 'exhale'):
            data = primary.query('channel == @channel and state == @state').sort_values('size_mm')
            ax.plot(data.size_mm, data.baseline_raw_median_ohm, 'o-', label=state)
        ax.set(title=f'Общий BEI: базовый уровень, канал {channel}', xlabel='L, мм', ylabel='Z, Ом')
        ax.legend(); ax.grid(alpha=.2)
        ax = axes[1, channel - 1]
        data = primary.query('channel == @channel and state == "inhale"').sort_values('size_mm')
        for row in data.itertuples():
            e = ensembles[row.record_id]['windows']['inhale']['channels'][str(channel)]
            if e['n_cycles']:
                ax.plot(e['phase_rr'], 1000 * np.array(e['median_ohm']), label=f'{row.size_mm:g} мм')
        ax.set(title=f'Задержка вдоха: кандидатные циклы, канал {channel}', xlabel='Фаза QRS–QRS', ylabel='BEI 0,7–12 Гц, мОм')
        ax.legend(ncol=3); ax.grid(alpha=.2)
    fig.suptitle('Scorpion v2 — первый проход пяти размеров; роли каналов не подтверждены')
    fig.savefig(out / 'quicklook.png', dpi=160)
    plt.close(fig)


def research_sections(observations, ensembles):
    contacts = observations[observations.record_id.isin(['SCV2-G-002', 'SCV2-G-003', 'SCV2-G-004'])].copy()
    contacts['токовых на руке'] = contacts.record_id.map({'SCV2-G-002': 1, 'SCV2-G-003': 2, 'SCV2-G-004': 1})
    contacts['измерительных на руке'] = contacts.record_id.map({'SCV2-G-002': 1, 'SCV2-G-003': 2, 'SCV2-G-004': 4})
    contacts['межцикловое SD, мОм'] = [
        1000 * np.sqrt(np.mean(np.array(ensembles[row.record_id]['windows'][row.state]['channels'][str(row.channel)]['cycle_sd_ohm']) ** 2))
        if row.n_candidate_cycles > 1 else np.nan for row in contacts.itertuples()]
    return ['<h2>Число и площадь контактов ТТРКГ: наблюдения</h2>',
            contacts[['record_id', 'state', 'channel', 'токовых на руке',
                      'измерительных на руке', 'baseline_raw_median_ohm',
                      'median_curve_ptp_mohm', 'межцикловое SD, мОм',
                      'n_candidate_cycles']].to_html(index=False, float_format=lambda x: f'{x:.3f}'),
            '<p>При одинаковой площади A одного контакта суммарные площади на каждой '
            'руке равны (A,A), (2A,2A), (A,4A). Фактическая A не измерена. '
            'Контраст 1×1 ↔ 1×4 сохраняет число токовых контактов, но меняет '
            'площадь и пространственное распределение измерительных. 2×2 меняет '
            'обе группы. Чистой факторной оценки токовой площади эти три '
            'последовательные записи не обеспечивают.</p>',
            '<p>Межцикловой разброс включает физиологическую вариабельность и ошибки '
            'фазовой привязки; он не объявляется приборным шумом. Различия кривых '
            'между записями пока не приписываются одной площади.</p>',
            '<h2>Модельная проверка площадей и вкладов</h2>',
            '<p>Для индивидуальной CEM Георга предлагается факторный расчёт: '
            '1 или 2 токовых контакта и 1, 2 или 4 измерительных на каждой руке '
            '(6 сочетаний). Каждая раздельная группа подключена как один '
            'электрический узел; токи между площадками не задаются равными. '
            'Отдельный опыт изменяет диаметр при фиксированных центрах и числе '
            'контактов; другой сохраняет суммарную площадь при изменении числа '
            'площадок. Это отделяет площадь от распределения по окружности.</p>',
            '<p>Сравниваются Z, дыхательное ΔZ, форма и размах пульсовой кривой, '
            'разброс циклов, плотность тока и чувствительность к контактам и '
            'установке. Затем отдельно сравниваются знаковые производные по '
            'сопротивлениям мягких тканей, лёгких, сердца и других разрешённых '
            'масок. Для модельного вклада в ΔZ нужна также амплитуда изменения '
            'каждого компартмента; одна производная не является его наблюдаемым вкладом.</p>',
            '<p>Влияние геометрии контактов на распределение чувствительности '
            'обосновано <a href="https://www.bem.fi/edu/doctor/kauppine/p3.pdf">'
            'Kauppinen, Hyttinen, Malmivuo, 1998</a>. Их шейно-грудные монтажи '
            'не валидируют монтаж на руках Георга. Конечная площадь и контактный '
            'импеданс входят в <a href="https://arxiv.org/abs/1312.4202">'
            'CEM-постановку Dardé и Staboulis</a>; увеличение площади само по себе '
            'не принимается критерием улучшения.</p>',
            '<h2>Восстановление креста на КТ: постановка проверки</h2>',
            '<p>Проверяемая гипотеза: общий центр и известные базы позволяют '
            'задавать крест компактнее, чем отдельные произвольные контакты. '
            'В идеализированном ортогональном шаблоне на поверхности остаются '
            'две координаты центра и один угол. Если угол между осями не '
            'фиксирован, появляется ещё один параметр. Фактическая '
            'ортогональность и способ отсчёта расстояния по кривой поверхности '
            'должны быть проверены отдельно.</p>',
            '<p>Компактная геометрическая параметризация не доказывает '
            'восстановимость положения по импедансу: плоская изотропная модель '
            'не чувствительна к переносу или повороту. На индивидуальной КТ/FEM '
            'требуется карта обоих Z по положению и углу, профили совместно с '
            'неизвестными тканевыми параметрами и проверка различимых альтернативных '
            'мест. Сравнить одну ось и две оси при одинаковых ошибках установки. '
            'Метрики: ошибка координат, число неразличимых областей, изменение '
            'оценок ρ и устойчивость к смещению/повороту. Вычислительные сценарии '
            '5/10/20 мм и 5/10° не являются измеренными погрешностями данного опыта.</p>',
            '<h2>ТЕПС: моделирование и эксперимент — следующий расчёт</h2>',
            '<p>Для 012–017 строятся TEPC-2–7 на Георге при 100 кГц; 018 AEPC-5 '
            'остаётся отдельной веткой. Старые расчёты на другом добровольце '
            'и 50 кГц не переименовываются в это сравнение. Сначала сравнить '
            'абсолютные Z и дыхательные разности, затем знак, форму, фазу и '
            'размах усреднённых пульсов.</p>',
            '<p>Динамическая КТ может дать геометрические сценарии движения для '
            'прямого расчёта; разные сессии не сопоставляются поточечно. Не '
            'подгонять отдельные усиления и сдвиги времени каждого монтажа ради '
            'совпадения кривых. Производную dZ/dt рассчитывать из измеренной и '
            'модельной кривых одинаковым оператором. Источник схем — '
            '<a href="https://paperhost.org/proceedings/embs/EMBC21/files/1232.pdf">'
            'Metshein et al., 2021</a>; его результаты качества не считаются '
            'референсными значениями УО/ФВ Георга.</p>']


def build_report(out, observations, ensembles, profiles, cases, contract):
    out = Path(out)
    (out / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    content = ['<h1>33.20. Первый проход боковых сборок Scorpion v2</h1>',
               '<p><strong>Исторический первый проход до уточнения каналов 09.10.2026.</strong> '
               'Прежние постоянные номера не соответствуют сообщённому переходу 004–007; '
               'рабочий пересчёт находится в 33.21. Исследовательские сценарии, без независимой валидации. '
               'Дыхательные интервалы и QRS остаются кандидатными. УО/ФВ здесь не рассчитываются.</p>',
               '<p>Все составляющие BEI получены из одного записанного общего импеданса: '
               'базовый уровень — медиана исходного BEI на выбранном интервале, '
               'быстрая кривая — полоса 0,7–12 Гц этапа 11.20. Частотное разделение не определяет тканевое происхождение.</p>',
               '<p>Подтверждение автора 08.10.2026: общий центр боковых размеров, '
               'вертикальных и крестообразных монтажей — правый бок, область рёбер. '
               'Точные координаты на КТ отложены. Динамическая КТ Георга доступна '
               'в серии 21; совпадение её состояния с этой сессией не установлено.</p>',
               '<p><a href="../../scorpion_review/index.html">Просмотр 11.20</a> · '
               '<a href="../../../PROTOCOL.md">Протокол</a> · '
               '<a href="input_contract.local.json">Происхождение и допущения</a> · '
               '<a href="verification.local.json">Внутренние численные проверки</a></p>',
               '<h2>Входы и проверка сопоставимости</h2>',
               '<p>Окна задержек вырезаны из уже существующих диапазонов неопределённости '
               'границ с отступом 2 с; края файла исключены на 5 с. '
               'Полные кандидатные циклы QRS–QRS допускаются при RR 0,45–1,8 с; '
               'каждый цикл центрируется и приводится к 200 фазам. '
               'Полоса, окна и правила RR являются условиями этого расчёта.</p>']
    columns = ['record_id', 'montage', 'state', 'channel', 'start_s', 'end_s',
               'baseline_raw_median_ohm', 'slow_p05_p95_span_ohm',
               'second_minus_first_half_slow_ohm', 'n_candidate_cycles',
               'median_curve_ptp_mohm']
    content.append(observations[columns].to_html(index=False, float_format=lambda x: f'{x:.3f}'))
    content.extend(['<h2>Базовый импеданс пяти горизонтальных размеров</h2>',
                    '<p>Роли каналов не назначаются по величине сигнала. Показаны два '
                    'согласованных сценария всего ряда. Это разные сценарии одного набора измерений.</p>'])
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Канал 1', 'Канал 2'])
    primary = observations[observations.size_mm.notna()]
    for channel in (1, 2):
        for state in ('inhale', 'exhale'):
            df = primary.query('channel == @channel and state == @state').sort_values('size_mm')
            fig.add_trace(go.Scatter(x=df.size_mm, y=df.baseline_raw_median_ohm,
                                    mode='lines+markers', name=f'{state}, канал {channel}'), row=1, col=channel)
    for case in cases:
        best = case['best_on_computational_grid_only']
        if best['n_sizes'] == 5:
            for state, fit in best['independent_states'].items():
                fig.add_trace(go.Scatter(x=best['sizes_mm'], y=fit['predicted_ohm'],
                                        line=dict(dash='dash'),
                                        name=f'Модель {state}, канал {best["lateral_channel_scenario"]}'),
                              row=1, col=best['lateral_channel_scenario'])
    fig.update_xaxes(title_text='L, мм'); fig.update_yaxes(title_text='Z, Ом')
    content.append(chart(fig))
    quicklook(out, primary, ensembles)
    content.append('<p><strong>Наблюдение первого прохода:</strong> запись 140 мм '
                   'отличается от 120–80 мм по базовым уровням и амплитудно-фазовым '
                   'формам обоих каналов. В 60 мм особенно нестабильна быстрая '
                   'кривая канала 1. Причины этих различий не установлены. '
                   'В этой исторической редакции принималась постоянная нумерация ролей. '
                   'Уточнение автора от 09.10.2026 заменило это допущение; '
                   'исправленный размерный ряд используется в 33.21.</p>')
    content.append('<h2>Усреднённые кандидатные циклы</h2>')
    for state in ('inhale', 'exhale'):
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Канал 1', 'Канал 2'])
        for channel in (1, 2):
            df = primary.query('channel == @channel and state == @state').sort_values('size_mm')
            for row in df.itertuples():
                e = ensembles[row.record_id]['windows'][state]['channels'][str(channel)]
                if e['n_cycles']:
                    fig.add_trace(go.Scatter(x=e['phase_rr'], y=1000 * np.array(e['median_ohm']),
                                            name=f'{row.size_mm:g} мм, канал {channel}, n={e["n_cycles"]}'), row=1, col=channel)
        fig.update_layout(title=state)
        fig.update_xaxes(title_text='Фаза между кандидатами QRS')
        fig.update_yaxes(title_text='ΔZ, мОм')
        content.append(chart(fig))
    content.extend(['<h2>Конечные контакты и профиль толщины</h2>',
                    '<p>Плоская двуслойная модель: конечные диски диаметром 5 мм, '
                    'равномерный ток и усреднение потенциала по диску. Это условная '
                    'электродная модель; полная CEM и индивидуальная КТ/FEM Георга не использованы. '
                    'Рёбра, кривизна, анизотропия, контактный импеданс и влияние соседнего канала не моделируются.</p>',
                    '<p>Сетка h=5–60 мм с шагом 5 мм задаёт вычислительные сценарии. '
                    'Минимум невязки на ней не является измеренной толщиной. '
                    'Границы ρ взяты из профиля 100 кГц реестра ITIS проекта: '
                    'объединение диапазонов мышцы/жира и вдохнутого/выдохнутого лёгкого. '
                    'Низкочастотные интервалы перенесены как допущение; '
                    'они не являются индивидуальными нормами или доверительными интервалами.</p>'])
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Горизонтальная сборка на канале 1', 'На канале 2'])
    for channel in (1, 2):
        for count in (5, 4):
            selected = [p for p in profiles if p['lateral_channel_scenario'] == channel and p['n_sizes'] == count]
            for state in ('inhale', 'exhale'):
                fig.add_trace(go.Scatter(x=[p['h_mm_scenario'] for p in selected],
                                        y=[p['independent_states'][state]['rmse_ohm'] for p in selected],
                                        name=f'{state}, {count} размеров, канал {channel}'), row=1, col=channel)
    fig.update_xaxes(title_text='Сценарий h, мм'); fig.update_yaxes(title_text='RMSE, Ом')
    content.append(chart(fig))
    table = []
    for case in cases:
        best = case['best_on_computational_grid_only']
        for state, fit in best['independent_states'].items():
            pulse = case['pulse'].get(state, {})
            table.append({'боковой канал': best['lateral_channel_scenario'],
                          'размеров': best['n_sizes'], 'режим': state,
                          'h сетки, мм': best['h_mm_scenario'],
                          'ρ₁, Ом·м': fit['rho1_ohm_m'], 'ρ₂, Ом·м': fit['rho2_ohm_m'],
                          'базовая невязка, %': fit['relative_residual_percent'],
                          'пульсовая невязка, %': pulse.get('residual_percent'),
                          'при δρ₁=0, %': pulse.get('rho1_zero_residual_percent'),
                          'граница ρ': fit['bound_hit'], 'край сетки h': case['h_grid_edge_hit']})
    content.append(pd.DataFrame(table).to_html(index=False, float_format=lambda x: f'{x:.3f}'))
    content.append('<p>Основной набор — пять размеров; вариант без 60 мм показывает '
                   'чувствительность к записи с флагом помехи. Он не принимает и не исключает '
                   'запись автоматически. Статические подгонки невзвешенные; '
                   'разброс уровней и циклов показан отдельно и не свёрнут в один «шум».</p>')
    content.append('<h2>Пульсовая инверсия в условных базовых точках</h2>')
    for case in cases:
        best = case['best_on_computational_grid_only']
        fig = make_subplots(rows=1, cols=2, subplot_titles=['δρ₁ мягких тканей', 'δρ₂ лёгкого'])
        for state, pulse in case['pulse'].items():
            for layer in (1, 2):
                fig.add_trace(go.Scatter(x=pulse['phase_rr'], y=pulse['delta_rho_ohm_m'][layer - 1],
                                        name=f'{state}, слой {layer}'), row=1, col=layer)
        fig.update_layout(title=f'Боковой канал {best["lateral_channel_scenario"]}; {best["n_sizes"]} размеров; h={best["h_mm_scenario"]:g} мм — сценарий')
        fig.update_xaxes(title_text='Фаза QRS–QRS'); fig.update_yaxes(title_text='δρ, Ом·м')
        content.append(chart(fig))
    content.append('<h2>Проверка пульсового прогноза на пяти размерах</h2>')
    for case in cases:
        best = case['best_on_computational_grid_only']
        if best['n_sizes'] != 5:
            continue
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Вдох', 'Выдох'])
        colors = ['#2563eb', '#d97706', '#16a34a', '#9333ea', '#dc2626']
        for col, state in enumerate(('inhale', 'exhale'), 1):
            pulse = case['pulse'].get(state)
            if not pulse:
                continue
            for index, size in enumerate(best['sizes_mm']):
                for key, dash, label in [('observed_ohm', 'solid', 'измерение'),
                                         ('predicted_ohm', 'dash', 'модель')]:
                    fig.add_trace(go.Scatter(x=pulse['phase_rr'],
                                            y=1000 * np.array(pulse[key][index]),
                                            line=dict(color=colors[index], dash=dash),
                                            name=f'{size:g} мм, {label}', showlegend=col == 1), row=1, col=col)
        fig.update_layout(title=f'Горизонтальная сборка на канале {best["lateral_channel_scenario"]} — условный сценарий')
        fig.update_xaxes(title_text='Фаза QRS–QRS'); fig.update_yaxes(title_text='ΔZ, мОм')
        content.append(chart(fig))
    content.extend(research_sections(observations, ensembles))
    content.extend(['<p>Инверсия использует производные той же модели конечных контактов '
                    'и базовую точку каждого дыхательного состояния. В JSON сохранены '
                    'полные прогнозы, невязки, проверка положительности ρ и ошибка линеаризации. '
                    'Эти кривые не считаются истинными тканевыми изменениями.</p>',
                    '<h2>Контрольные направления и конфигурации контактов</h2>',
                    '<p>Записи 002–004 сравнивают конфигурации ТТРКГ при номинале боковой '
                    'сборки 140 мм. Запись 005 — самостоятельная вертикальная; '
                    '006 и 011 — две боковые оси. Они сохранены в таблице входов, '
                    'но не добавлены как новые независимые размеры в плоскую подгонку: '
                    'повтор одного L не создаёт новую глубину чувствительности. '
                    'Направления служат проверкой применимости изотропного упрощения; '
                    'числовые роли самостоятельной вертикальной записи не установлены. '
                    'Для 011 надёжный канал автоматической ЭКГ не выбран, поэтому '
                    'пульсовые циклы не формируются.</p>',
                    '<p>ТЕПС/АЕПС сохранены как отдельные наблюдения в таблице входов. '
                    'Их нельзя инвертировать формулой боковой сборки. Сравнение с '
                    'индивидуальным моделированием требует собственного монтажа и оператора.</p>',
                    '<h2>Что передаётся дальше</h2>',
                    '<p>Этот проход проверяет согласованность пяти размеров с заданной '
                    'моделью и зависимость пульсовых оценок от неё. Для коррекции ТТРКГ '
                    'нужны её собственные чувствительности на Георге. Переход к сердечному '
                    'объёму и УО/ФВ здесь не выполнен. Динамическая КТ может дать '
                    'геометрическую проверку, но неопределённость масок и разные сессии сохраняются.</p>'])
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>33.20 Scorpion v2</title>'
    page += '<script src="plotly.min.js"></script><style>body{font:16px/1.5 system-ui;margin:32px;max-width:1500px}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:6px;border:1px solid #ccc}h2{margin-top:40px}</style>'
    page += ''.join(content) + '</html>'
    (out / 'index.html').write_text(page, encoding='utf-8')
