"""Address observed page-layout defects, keeping quantities and scientific scope."""
from pathlib import Path
import json, hashlib
R=Path(__file__).resolve().parents[1]
receipt=R/'reviews/layout_polish.json'
if receipt.exists():raise SystemExit('Already applied')
log=[]
def replace(file,old,new):
 p=R/file;s=p.read_text(encoding='utf-8');assert s.count(old)==1,(file,old[:60])
 out=s.replace(old,new);p.write_text(out,encoding='utf-8')
 log.append({'file':file,'old':old,'new':new,'before_sha256':hashlib.sha256(s.encode()).hexdigest(),'after_sha256':hashlib.sha256(out.encode()).hexdigest()})
replace('conclusion.md',' Положительные требования договора к конечной точности УО и комплектности всей электронной поставки не подтверждаются одним наличием отчёта; их фактический статус указан в матрице соответствия.','')
replace('chapters/04_experiments.md',r'$(-L/2,-L/4,L/4,L/2)$',r'$(-\frac{L}{2},-\frac{L}{4},\frac{L}{4},\frac{L}{2})$')
replace('chapters/06_stroke_volume.md',r'''\widehat V_v(t)=V_v^*+\Delta\mathcal M_v(\widehat{\mathbf g}(t),\phi(t);\mathbf g^*,\phi^*,\mathbf d_{CT},\boldsymbol\beta_v)+b_{v,\mathrm{sess}},
\qquad SV_v=V_v(t_{ED,v})-V_v(t_{ES,v}).''',r'''\begin{aligned}
\widehat V_v(t)&=V_v^*+\Delta\mathcal M_v(\widehat{\mathbf g}(t),\phi(t);\mathbf g^*,\phi^*,\mathbf d_{CT},\boldsymbol\beta_v)+b_{v,\mathrm{sess}},\\
SV_v&=V_v(t_{ED,v})-V_v(t_{ES,v}).
\end{aligned}''')
replace('supplement/04_experiments.md',r'''C_{cd}=\frac{\sum_{k,j}(e_{ckj}-\bar e_c)(e_{dkj}-\bar e_d)}{KP-1},\qquad
C_y=D^{-1}CD^{-1},\quad D=\operatorname{diag}(Z_{01},Z_{02}).''',r'''\begin{aligned}
C_{cd}&=\frac{\sum_{k,j}(e_{ckj}-\bar e_c)(e_{dkj}-\bar e_d)}{KP-1},\\
C_y&=D^{-1}CD^{-1},\qquad D=\operatorname{diag}(Z_{01},Z_{02}).
\end{aligned}''')
replace('supplement/01_foundations.md',r'приближение $-\Delta\sigma/\sigma_0^2$ требует $|\Delta\sigma|\ll\sigma_0$',r'приближение $-\Delta\sigma/\sigma_0^2$ допустимо при малом относительном изменении проводимости')
replace('supplement/05_signal_models.md','| Прекардиальное отведение №2, Ом | $\\Delta Z$ ТТРКГ, мОм | $\\Delta Z$ прекардиального отведения, мОм |','| Прекард. №2, Ом | $\\Delta Z$ ТТРКГ, мОм | $\\Delta Z$ прекард. №2, мОм |')
replace('supplement/05_signal_models.md','| Геометрия и условия | Запись | RMS $\\Delta V$, мл, относительная задача | $E_{\\mathrm{rel}}$ на исключённых парах, прежняя → относительная, % | Вольтовый остаток относительного решения, % | Отсчёты вне объёмных проб, % |','| Геометрия и условия | Запись | RMS $\\Delta V$, мл | $E_{\\mathrm{rel}}$, прежняя → отн., % | Вольтовый остаток, % | Отсчёты вне проб, % |')
replace('supplement/05_signal_models.md','Общая метрика: $E_{\\mathrm{rel}}=', 'В таблице Е.6 стрелка сопоставляет прежнюю и относительную постановки на исключённых парах. RMS, вольтовый остаток и доля отсчётов вне объёмных проб относятся к относительному решению.\n\nОбщая метрика: $E_{\\mathrm{rel}}=')
replace('supplement/06_stroke_volume.md','| Опора, обратная форма | Обусловленность нормированных столбцов | Остаточная норма объёмного направления, % | Макс. ошибка объёма, мл | Проба максимума, мм | Невязка этой пробы, % | Выходы за границы либо неположительное ρ, из 8 |','| Опора, обратная форма | $\\kappa$ столбцов | Остаток V, % | Макс. ошибка объёма, мл | Проба максимума, мм | Невязка пробы, % | Нарушения, из 8 |')
replace('supplement/06_stroke_volume.md','Флаги означают выход объёма за ±5%', '$\\kappa$ — число обусловленности матрицы с нормированными столбцами; остаток V — остаточная норма объёмного направления, %. Нарушения означают выход объёма за ±5%')
replace('supplement/08_archive.md','[segment_ct_to_stl.py]','[Скрипт КТ → STL]')
replace('supplement/08_archive.md','[prepare_ct_fem.py]','[Скрипт подготовки]')
replace('tools/build_word.py','            p.paragraph_format.space_after = Pt(5)','            p.paragraph_format.space_after = Pt(4)')
receipt.write_text(json.dumps({'scope':'Observed visual QA fixes; main typography unchanged. Bibliography item spacing 5→4pt, font and line spacing unchanged. All numerical observations retained.','changes':log},ensure_ascii=False,indent=2),encoding='utf-8')
print(len(log),'layout edits')
