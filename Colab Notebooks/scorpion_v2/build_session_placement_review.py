"""Self-contained HTML for reviewing two independent anatomical hypotheses."""
import base64
from pathlib import Path

from audit_scorpion_session import read, sha
from session_20261022 import EXPERIMENT, SUBJECT_NAMES

STEM='20.40_Проверка_боковой_установки_Ника_и_Ярослава_Scorpion'


def build():
    source=EXPERIMENT/'derived/exploratory/ct_preparation_3340/surface_and_candidate_summary.local.json'
    results=read(source);panels=[]
    for index,subject in enumerate(['nik','yaroslav']):
        r=results['subjects'][subject];p=Path(r['candidate_figure']['path'])
        if sha(p)!=r['candidate_figure']['sha256']:raise ValueError('Changed candidate image')
        data=base64.b64encode(p.read_bytes()).decode('ascii')
        xyz='; '.join(f'{x:.2f}'.replace('.',',') for x in r['candidate_center_lps_mm'])
        panels.append(f'''<section id="{subject}" {'hidden' if index else ''}>
<h2>{SUBJECT_NAMES[subject]}: кандидатная область боковой сборки</h2>
<p>Центр в системе LPS, мм: <strong>({xyz})</strong>.
Координаты определены по собственной КТ; это пока гипотеза положения.</p>
<div class="viewport" tabindex="0" role="region" aria-label="Схема с прокруткой">
<img src="data:image/png;base64,{data}" alt="{SUBJECT_NAMES[subject]}: КТ справа, центр и варианты расположения боковой сборки"></div>
<p class="caption">Рисунок {index+1}. Слева — поперечное КТ-сечение, в центре —
продольное сечение и пример четырёх контактов, справа — возможные смещения центра.
Жёлтая точка обозначает выбранный геометрический центр; зелёные точки — варианты области.
Правая сторона участника на КТ находится слева на изображении.</p></section>''')
    body='''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ник и Ярослав: проверка боковой установки на КТ</title>
<style>body{font:17px/1.65 system-ui,sans-serif;color:#172033;background:#f8fafc;margin:0}
main{max-width:1440px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:23px}
nav,.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
button,select{font:inherit;padding:7px 18px;border:1px solid #94a3b8;border-radius:6px;background:white;cursor:pointer}
button[aria-selected=true]{background:#1e40af;color:white}.viewport{max-height:75vh;overflow:auto;
border:1px solid #cbd5e1;background:white}.viewport img{display:block;width:150%;max-width:none}
.caption{color:#475569}.note{border-left:4px solid #d97706;padding:10px 18px;background:#fffbeb}
input{width:240px}strong{font-weight:650}@media(max-width:600px){main{padding:14px}h1{font-size:24px}}
</style><main><h1>Где находились боковые сборки Ника и Ярослава</h1>
<p>Схемы для сверки установки Scorpion от 22.10.2026. Выберите участника,
увеличьте изображение и прокрутите его по горизонтали. Страница содержит все изображения внутри файла.</p>
<nav aria-label="Участник"><button data-subject="nik" aria-selected="true">Ник</button>
<button data-subject="yaroslav" aria-selected="false">Ярослав</button></nav>
<div class="controls"><label>Масштаб: <input id="zoom" type="range" min="70" max="250" step="10" value="150"></label>
<output id="scale">150%</output><button id="fit">Вместить целиком</button></div>
<p>Центр выбран <strong>на уровне середины маски сердца и примерно посередине передней-задней ширины тела</strong>,
на правой кожной границе с лёгким под ней. Это анатомическое приближение боковой линии;
точная средняя подмышечная линия по складкам пока не определена.</p>
'''+''.join(panels)+'''
<p class="note">Проверяется только положение. Наружный контур пока не прошёл стандартные
геометрические проверки, поэтому расчёт FEM по нему не выполнен. Пример 140 мм
отложен вдоль кожной кривой и ещё не подтверждает фактические расстояния между электродами.</p>
<p>Для уточнения достаточно указать: соответствует примерно либо центр следует сдвинуть
выше, ниже, вперёд или назад — отдельно для каждого участника.</p>
</main><script>
const zoom=document.getElementById('zoom');
function resize(){document.querySelectorAll('.viewport img').forEach(img=>img.style.width=zoom.value+'%');document.getElementById('scale').textContent=zoom.value+'%';}
zoom.addEventListener('input',resize);
document.getElementById('fit').addEventListener('click',()=>{zoom.value=100;resize();});
document.querySelectorAll('[data-subject]').forEach(button=>button.addEventListener('click',()=>{
document.querySelectorAll('section').forEach(section=>section.hidden=section.id!==button.dataset.subject);
document.querySelectorAll('[data-subject]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));}));
</script></html>'''
    output=Path(__file__).resolve().parent/(STEM+'.html')
    output.write_text(body,encoding='utf8');print(output.name,output.stat().st_size)
    return output


if __name__=='__main__':build()
