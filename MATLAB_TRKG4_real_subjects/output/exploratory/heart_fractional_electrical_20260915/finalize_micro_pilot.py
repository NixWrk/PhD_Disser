"""Finalize the requested 13-state fractional P1 micro-pilot (no new solves)."""
from pathlib import Path
import json,sys,statistics,csv
out=Path(__file__).resolve().parent;root=out.parents[2];sys.path.insert(0,str(root/'tools'))
import heart_fractional_electrical_pilot as pilot
p=pilot.load_plan(out);n=pilot.summarize(out,p);assert n==13
s=pilot.read(out/'summary.json');assert len(s['derivatives'])==5 and len(s['comparisons'])==3
exports=[pilot.checked_export(out,p,state) for state in p['states']]
rows=[pilot.checked_result(out,p,state) for state in p['states']]
assert all(r is not None for r in rows)
processes=[pilot.read(out/('solve_'+state['id']+'_process.json')) for state in p['states']]
export_process=pilot.read(out/'weighted_export_process.json')
assert all(r['status']=='completed' for r in processes+[export_process])
qc=pilot.read(out/'numerical_qc.json')
resources={'weighted_export_process_seconds':export_process['elapsed_s'],'material_export_seconds':[r['seconds'] for r in exports],
           'median_state_solve_process_seconds':statistics.median(r['elapsed_s'] for r in processes),
           'total_stage_process_seconds':sum(r['elapsed_s'] for r in processes+[export_process]),
           'max_sampled_process_tree_rss_gib':max(r['peak_rss_gib'] for r in processes+[export_process]),
           'memory_guard_gib':32,'threads':4,'rhs_count':26,'factorizations':13,
           'max_homogeneous_assembly_error':max(r['homogeneous_error'] for r in exports),
           'max_heterogeneous_assembly_error':max(r['heterogeneous_error'] for r in exports),
           'max_symbolic_memory_estimate_gib':max(r['factorization_memory']['conservative_rss_plus_solver_estimate_gib'] for r in rows)}
pilot.write(out/'resource_report.json',resources)
with (out/'derivatives.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(s['derivatives'][0]));w.writeheader();w.writerows(s['derivatives'])
cols=['id','geometry','points_per_tet','volume_fraction','analytic_volume_ml','material_volume_ml','Z_ohm','reciprocal_Z_ohm','max_relative_residual','max_current_error_A']
with (out/'states.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
text=['# Дробный электрический микропилот: 13 новых состояний TEPC-2','',
 'Выполнены пять состояний сферы и пять состояний эллипсоида при 4096 точках интегрирования на частично пересечённый тетраэдр: исходный объём и ±0,5%, ±1%. Для сферы исходный объём и ±0,5% повторены при 512 точках. Всего 13 новых решений, 13 факторизаций, 26 правых частей (прямой и взаимный расчёт). Прежний пилот 18 состояний не изменён.','',
 'Используется замороженная статическая сетка Nix, TEPC-2 с контактами 5 мм, вещественная CEM при проводимостях для 50 кГц. В исходной области сердца фон — мягкие ткани; кость и лёгкие сохранены. Сердце имеет приоритет. Доли материалов меняются внутри прежних тетраэдров, без перемещения фоновых органов.','',
 'Объёмная производная рассчитана центральной разностью ΔZ/(2δV) по заданному аналитическому объёму в мл. Это проверка численного шага, не физиологическая амплитуда, не оценка УО/ФВ. Производная относительно фактической разности объёмов материала также сохранена как диагностическая, но не подменяет основной оператор.','',
 '| Форма | Точек | Шаг объёма | Z исходного состояния, Ом | dZ/dV, Ом/мл | dVмат/dVанал |',
 '|---|---:|---:|---:|---:|---:|']
for d in s['derivatives']:
 text.append(f"| {d['geometry']} | {d['points_per_tet']} | ±{100*d['step_fraction']:.1f}% | {d['Z_base_ohm']:.8f} | {d['dZ_dV_analytic_ohm_per_ml']:.9f} | {d['dVmaterial_dVanalytic']:.7f} |")
text+=['','## Сопоставления','']
for c in s['comparisons']:
 label='Уменьшение шага с ±1% до ±0,5%' if c['comparison']=='step_halving' else 'Уточнение квадратуры с 512 до 4096 точек при ±0,5%'
 text.append(f"- {label}, {c['geometry']}: модуль относительного различия производной {100*c['relative_difference']:.4f}%; знаковая разность {c['difference_ohm_per_ml']:.9g} Ом/мл.")
text+=['',
 'Эти различия наблюдаются в данной одной сборке, при фиксированном наборе удельных сопротивлений и на текущей сетке. До расчёта не задавался критерий приемлемой ошибки геометрической производной, поэтому результат не маркируется как окончательно сошедшийся. Для почти нулевой производной необходим абсолютный критерий; в таком случае относительная разность неустойчива.','',
 '## Корректность сборки и решений','',
 r'Материальные блоки собраны заново как $B_k=F^T\operatorname{diag}(\operatorname{repelem}(f_k,3))F$. В коде строки F умножаются на $\sqrt{f_k}$, затем вычисляется произведение с транспонированной матрицей. Сумма долей в каждом тетраэдре равна 1. $K=\sum_k\sigma_kB_k+C$. Старые блоки и старые решения не использованы для новых состояний. Геометрический оператор F общий, поскольку узлы и линейный базис не меняются.','',
 f"Максимальная относительная ошибка однородной сборки: {resources['max_homogeneous_assembly_error']:.3g}; для фактических средних проводимостей против calc_system_mat: {resources['max_heterogeneous_assembly_error']:.3g}. Максимальная относительная невязка решения: {qc['max_relative_residual']:.3g}; расхождение взаимности: {qc['max_reciprocity_absolute_ohm']:.3g} Ом; ошибка восстановленных токов: {qc['max_current_error_A']:.3g} А.",'',
 '## Ресурсы','',
 f"Общий MATLAB-экспорт 13 состояний: {resources['weighted_export_process_seconds']:.1f} с. Медиана процесса одного решения: {resources['median_state_solve_process_seconds']:.1f} с. Сумма времени вычислительных стадий: {resources['total_stage_process_seconds']/60:.2f} мин. Максимальный наблюдавшийся RSS дерева процесса: {resources['max_sampled_process_tree_rss_gib']:.2f} ГиБ (наблюдение каждые 2 с). Максимальная консервативная оценка памяти перед факторизацией: {resources['max_symbolic_memory_estimate_gib']:.2f} ГиБ. Лимит 32 ГиБ не достигнут.",'',
 '## Что результат пока не доказывает','',
 'Точное интегрирование проводимости на тетраэдре даёт правильный интеграл для выбранного P1-базиса. Оно не даёт базису способности разрешать скачок градиента потенциала внутри крупной ячейки. Рёбра около сердца остаются порядка 16 мм. Поэтому геометрическая квадратура, линейный решатель и пространственная аппроксимация поля требуют раздельных проверок.','',
 'Общий набор квадратурных точек создаёт связанные ошибки между состояниями. Малое различие при уменьшении шага не является независимым подтверждением отсутствия смещения. Другие монтажи, проводимости и переносы сердца здесь не проверялись. Сравнение автоматической маски целевой фазы и индивидуальной опорной формы, совмещение с 4D и физиологические закономерности отсутствуют в этом статическом микропилоте.','',
 'Следующий батч: отдельно согласовать критерии производных и выбрать один перенос для такой же пары шагов/уровней квадратуры; затем проверить одну локально уточнённую сердечную область при тех же материальных долях и монтажах. Большой перебор всех сборок и перемещений автоматически не запускался.','',
 '## Файлы и возобновление','',
 'Новые исходники: src/run_heart_fractional_electrical_pilot.m и tools/heart_fractional_electrical_pilot.py. Общие helpers, README, ноутбуки и реестр не менялись; коммит не выполнялся. Все локальные входы/версии/хэши сохранены в plan.json и source_used. Каждый state содержит новые blocks.mat, прямое и взаимное поле fields.npz, результаты, оценку памяти и completion.json. states.csv, derivatives.csv и summary.json — компактные выходы для отчёта.','',
 'Команда возобновления (из корня репозитория):','',
 '```powershell',
 '& C:/PC/Python/python.exe -B -X utf8 MATLAB_TRKG4_real_subjects/tools/heart_fractional_electrical_pilot.py run --output MATLAB_TRKG4_real_subjects/output/exploratory/heart_fractional_electrical_20260915',
 '```','',
 'Хэши защищают от подстановки чужой карты, матрицы или завершённого результата. Незавершённый state требует осмотра перед новым запуском; существующие поля не перезаписываются молча.']
(out/'RESULTS_AND_HANDOFF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
pilot.write(out/'verification.json',{'status':'all_13_completed_and_hash_verified','states':13,'derivatives':5,'comparisons':3,
    'summary_sha256':pilot.sha(out/'summary.json'),'resource_report_sha256':pilot.sha(out/'resource_report.json'),
    'finalizer_sha256':pilot.sha(__file__),'numerical_qc':qc,
    'spatial_convergence':False,'physical_validation':False})
print(json.dumps({'derivatives':s['derivatives'],'comparisons':s['comparisons'],'resources':resources},indent=2))
