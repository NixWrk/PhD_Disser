"""One-time removal of exact duplicates after chapter handoffs."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'reviews/editorial_cuts.json'
if receipt.exists():raise SystemExit('Already applied; do not rerun over reviewed prose')
changes=[]
def save(p,before,text,reason):
    p.write_text(text,encoding='utf-8')
    changes.append({'file':p.relative_to(ROOT).as_posix(),'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(text.encode()).hexdigest(),'reason':reason})
p=ROOT/'chapters/06_stroke_volume.md';before=p.read_text(encoding='utf-8');t=before
start=t.index('Для формы из одной опоры проверен закон однородного масштабирования и переноса:')
end=t.index('Электрическая статическая STL',start)
t=t[:start]+'Динамика опорной формы задаётся масштабированием и переносом по (3.3). Восстановлению подлежат масштаб и смещение; целевые фазовые значения в этот расчёт не передаются. Фиксированное положение — частный сценарий; соответствие реальной деформации проверяется отдельно.\n\n'+t[end:]
t,n=re.subn(r'!\[[^\]]*\]\(\.\./figures/4af36cf17afe\.png\)\s*\nРисунок 6\.1 —[^\n]*', 'Изменение чувствительности после ложной тканевой поправки показано на рисунке 5.3.',t)
assert n==1
def eq(m):return m[1]+'6.'+str(int(m[2])-1)+m[3]
t=re.sub(r'(\\tag\{)6\.([2-7])(\})',eq,t)
t=re.sub(r'(\()6\.([2-7])(\))',eq,t)
def figs(m):
    out=m[1]+'6.'+str(int(m[2])-1)
    if m[3]:out+=m[3]+'6.'+str(int(m[4])-1)
    return out
t=re.sub(r'(рисун\w*\s+)6\.([2-4])(?:(\s+и\s+)6\.([2-4]))?',figs,t,flags=re.I)
save(p,before,t,'Scale/translation formula retained in3.3; duplicate image retained in5.3; chapter6 equations and figure references renumbered without scientific value changes.')
p=ROOT/'chapters/07_validation.md';before=p.read_text(encoding='utf-8');t=before
a=t.index('При оценке разности камерных объёмов необходимо учитывать зависимость ошибок:')
b=t.index('## 7.3',a)
t=t[:a]+'Зависимость ошибок конечных объёмов учитывается формулой (6.6). Она описывает распространение уже установленных ошибок и не задаёт численной погрешности при неизвестных дисперсиях и ковариации. Геометрическая и тканевая ошибки также могут зависеть от одних и тех же изменений, поэтому их независимое квадратичное сложение не обосновано.\n\n'+t[b:]
save(p,before,t,'Same covariance formula and its definitions retained in6.6; chapter7 keeps applicability limitation and cross-reference.')
p=ROOT/'chapters/03_anatomy.md';before=p.read_text(encoding='utf-8');t=before
a=t.index('## 3.6 Граница анатомического результата')
b=t.index('## Источники',a)
t=t[:a]+'''## 3.6 Граница анатомического результата

Подготовлены прослеживаемые анатомические входы и выполнены ограниченные сравнения форм. Полнота сегментации, межсерийная регистрация и дискретное представление сохраняют неопределённость. Геометрическое согласие не устанавливает электрическую эквивалентность: она проверяется в главе 5, а переход к камерным объёмам и УО — в главе 6. Дополнительные геометрические результаты приведены в приложении Г.

'''+t[b:]
save(p,before,t,'Removed repeated general physics/SV caveats already preserved in2,5,6; retained all anatomical limitations and explicit appendix reference.')
receipt.write_text(json.dumps({'changes':changes,'scientific_results_removed':False,'duplicates_relocated':True},ensure_ascii=False,indent=2),encoding='utf-8')
print('Removed one duplicate figure and two duplicate displayed formulas; corrected references.')
