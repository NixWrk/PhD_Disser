"""BSA indexing of saved conditional estimates; no anatomical correction."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7734766/"
MOSTELLER = "https://pubmed.ncbi.nlm.nih.gov/3657876/"
ESTIMATORS = ("observed_global_range", "periodic_composite")
METRICS = {"edv_ml": "EDV/BSA, мл/м²", "esv_ml": "ESV/BSA, мл/м²",
           "sv_ml": "УО/BSA, мл/м²", "ef_percent": "ФВ, %"}
# Published limits are retained verbatim: rounding mean/SD cannot recover them.
# Each tuple: pooled mean, pooled SD, lower limit, upper limit, sample size.
REFERENCE = {
    "ЛЖ": {"table": 3, "papillary_convention": "included_in_mass_excluded_from_cavity",
           "metrics": {"edv_ml": (86, 13, 61, 112, 51), "esv_ml": (34, 10, 14, 53, 51),
                       "sv_ml": (54, 7, 40, 68, 41), "ef_percent": (60, 7, 46, 74, 51)}},
    "ПЖ": {"table": 11, "papillary_convention": "included_in_cavity_volume",
           "metrics": {"edv_ml": (94, 15, 63, 124, 50), "esv_ml": (44, 11, 23, 66, 50),
                       "sv_ml": (51, 13, 26, 77, 40), "ef_percent": (52, 8, 36, 69, 50)}},
}


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def mosteller_bsa(height_cm, weight_kg):
    """Inputs must use the named units; result is in square metres."""
    return math.sqrt(positive(height_cm, "height_cm") * positive(weight_kg, "weight_kg") / 3600)


def index_estimates(estimates, anthropometry):
    """Index available raw estimates; missing inputs never borrow another subject's data."""
    records, availability = [], []
    for subject in sorted({r["subject"] for r in estimates}):
        person = anthropometry.get(subject, {})
        if any(person.get(k) is None for k in ("height_cm", "weight_kg")):
            availability.append({"subject": subject, "status": "missing_anthropometry"})
            continue
        bsa = mosteller_bsa(person["height_cm"], person["weight_kg"])
        age = person.get("age_years")
        if age is not None:
            age = positive(age, "age_years")
        reference_available = person.get("sex") == "male" and age is not None and 20 <= age < 30
        availability.append({"subject": subject, "status": "indexed" if reference_available else "indexed_no_matching_reference", "bsa_m2": bsa})
        selected = [r for r in estimates if r["subject"] == subject and r.get("scenario") == "raw"
                    and r.get("estimator") in ESTIMATORS and r.get("sv_ml") is not None]
        seen = set()
        for row in selected:
            chamber = row["chamber_label"]
            if chamber not in REFERENCE:
                continue
            identity = (chamber, row["estimator"])
            if identity in seen:
                raise ValueError(f"Duplicate source estimate: {subject}, {identity}")
            seen.add(identity)
            for metric in METRICS:
                raw = row[metric]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0 or (metric == "edv_ml" and raw == 0):
                    raise ValueError(f"Invalid source value for {metric}: {raw}")
                raw = float(raw)
                value = raw if metric == "ef_percent" else raw / bsa
                rec = {"subject": subject, "chamber": chamber, "estimator": row["estimator"],
                       "source_status": row.get("status", ""), "metric": metric, "raw_value": raw,
                       "bsa_m2": bsa, "value": value, "unit": "%" if metric == "ef_percent" else "ml/m2",
                       "comparison_status": "descriptive_not_validated", "reference_position": "unavailable"}
                if reference_available:
                    mean, sd, low, high, n = REFERENCE[chamber]["metrics"][metric]
                    rec.update(reference_mean=mean, reference_sd=sd, reference_low=low, reference_high=high,
                               reference_n=n, reference_table=REFERENCE[chamber]["table"],
                               reference_age="20–29", reference_sex="male", reference_modality="CMR",
                               papillary_convention=REFERENCE[chamber]["papillary_convention"],
                               reference_url=SOURCE,
                               reference_position="below" if value < low else "above" if value > high else "within")
                records.append(rec)
    return records, availability


METHOD = """## Индексация объёмов и сопоставление с референсами

### Задача и происхождение исходных сведений

Цель раздела — выразить условные объёмы с учётом размеров тела и сопоставить
их с опубликованной возрастной группой. Такое сравнение даёт внешний ориентир
для обсуждения УО и ФВ, но не определяет правильность границ автоматических масок.
Проверяется математическое следствие преобразования: деление объёмов одного
испытуемого на общую площадь поверхности тела сохраняет ФВ и отношение УО ПЖ/ЛЖ.

Антропометрия считывается из локального файла входных данных. Для Nix её источник —
сообщение автора в предоставленной выписке диалога с GPT. Связь сообщения с Nix
восстановлена по совпадению исходной таблицы объёмов. Это сведения из диалога,
а не проверенный протокол измерения роста и массы. Применимость указанных
возраста и антропометрии к моменту КТ принята как рабочее допущение;
даты измерений и исследования в этом сопоставлении не согласованы.

### Формулы и смысл преобразования

Использована [формула Mosteller](https://pubmed.ncbi.nlm.nih.gov/3657876/):

<div class="scientific-equations"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>BSA</mi><mo>=</mo><msqrt><mfrac><mrow><mi>H</mi><mo>·</mo><mi>W</mi></mrow><mn>3600</mn></mfrac></msqrt><mo>,</mo></mrow></math></div>

где BSA — расчётная площадь поверхности тела, м²; H — рост, см;
W — масса тела, кг. Площадь не измерялась непосредственно.
Для индексации используется одно и то же значение BSA во всех кадрах:

<div class="scientific-equations"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>EDV</mi><mi>i</mi></msub><mo>=</mo><mfrac><mi>EDV</mi><mi>BSA</mi></mfrac><mo>,</mo><mspace width="1em"/><msub><mi>ESV</mi><mi>i</mi></msub><mo>=</mo><mfrac><mi>ESV</mi><mi>BSA</mi></mfrac><mo>,</mo><mspace width="1em"/><msub><mi>SV</mi><mi>i</mi></msub><mo>=</mo><mfrac><mi>SV</mi><mi>BSA</mi></mfrac><mo>,</mo></mrow></math></div>

где EDV и ESV — кандидатные конечные диастолический и систолический объёмы,
SV — условный УО или размах, мл; индекс i обозначает деление на BSA,
результат выражается в мл/м². В таблице ниже сохраняются исходные способы
выбора экстремумов: доступные маски и условный периодический цикл.

<div class="scientific-equations"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>EF</mi><mo>=</mo><mn>100</mn><mfrac><mrow><msub><mi>EDV</mi><mi>i</mi></msub><mo>−</mo><msub><mi>ESV</mi><mi>i</mi></msub></mrow><msub><mi>EDV</mi><mi>i</mi></msub></mfrac><mo>=</mo><mn>100</mn><mfrac><mrow><mi>EDV</mi><mo>−</mo><mi>ESV</mi></mrow><mi>EDV</mi></mfrac><mo>,</mo></mrow></math></div>

где EF — ФВ, %. Общий делитель сокращается: ФВ дополнительно не делится
на площадь тела. Возраст и пол задают выбор референсной группы,
но не являются поправочными множителями в этих формулах.
Индексация на BSA отличается от нормировки кривой в диапазоне 0–1:
её результат имеет размерность мл/м²; порядок кадров и положение экстремумов
сохраняются.

### Основание для сравнения

Выбраны таблицы 3 и 11 обзора [Kawel-Boehm и соавт. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7734766/)
для мужчин 20–29 лет, полученные для МРТ сердца. Для ЛЖ папиллярные мышцы
включены в массу миокарда и исключены из полости; для ПЖ включены в объём
полости. Соответствие этих правил нашим автоматическим маскам не проверено.
Ни одна из таблиц не выбиралась по близости к полученному результату.

В таблице 9 приведены опубликованные границы, объединённое среднее,
стандартное отклонение SD и число участников n для каждого показателя.
Границы в источнике определены как среднее ± 2 SD. Они перенесены как
опубликовано, без восстановления из округлённых среднего и SD.
Это референсные интервалы выбранной совокупности, а не доверительные
интервалы наших оценок и не индивидуальные диагностические пороги.
"""

LIMITATIONS = """### Допущения, неопределённость и границы вывода

**Допущения.** Рост, масса и возраст из диалога относятся к исследованию Nix;
площадь тела постоянна в рассматриваемых кадрах. Модельный цикл сохраняет
допущение сопоставимости сокращений. Сравнение с МРТ используется как
предварительный ориентир при неустановленной сопоставимости контуров.

**Нерешённые вопросы.** Неизвестны ошибки и даты антропометрических измерений,
индивидуальная ошибка формулы BSA, анатомическая ошибка масок и вклад
неполного охвата конечных состояний. Межмодальное различие КТ–МРТ и правила
проведения границ являются отдельными источниками расхождения. Поэтому
суммарная погрешность и доверительный интервал не рассчитываются.

Для малых возмущений входов при ненулевом объёмном показателе локальная
чувствительность индекса имеет вид:

<div class="scientific-equations"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mrow><mi>δ</mi><msub><mi>X</mi><mi>i</mi></msub></mrow><msub><mi>X</mi><mi>i</mi></msub></mfrac><mo>≈</mo><mfrac><mrow><mi>δ</mi><mi>X</mi></mrow><mi>X</mi></mfrac><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mfrac><mrow><mi>δ</mi><mi>H</mi></mrow><mi>H</mi></mfrac><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mfrac><mrow><mi>δ</mi><mi>W</mi></mrow><mi>W</mi></mfrac><mo>,</mo></mrow></math></div>

где X — любой из объёмных показателей EDV, ESV или SV в мл;
Xᵢ = X/BSA — соответствующий индекс в мл/м²; H и W — рост в см и масса
в кг; δ обозначает малое изменение величины. Это математическая
линеаризация, а не вероятностная модель погрешности. Распределения и
независимость ошибок здесь не задаются. Например, увеличение массы на 1%
при фиксированных росте и объёме уменьшает индекс приблизительно на 0,5%,
не меняя ФВ. Ошибка маски при этом сохраняется.

**Интерпретация.** Попадание в референсный интервал не подтверждает точность
сегментации, а выход за него не позволяет разделить физиологическое отличие,
ошибку границ и ошибку выбора фаз. В частности, возрастная стратификация
не позволяет установить механические фазы и не уточняет временное разрешение
КТ. Значения V/A, изменения формы и межкадровые скорости остаются средствами
проверки гипотез о происхождении объёмных различий. Предположение, что их
согласованность доказывает нормальную функцию сердца, не принимается.

**Вывод раздела.** Получен воспроизводимый способ представления условных
оценок с учётом размеров тела. Индексация сама по себе не позволяет определить
анатомическую поправку к объёмам. Последующее уточнение УО/ФВ требует проверки
выбранных конечных состояний и устойчивости границ; приближение к референсному
среднему или равенство оценок ЛЖ и ПЖ не используются как критерий подгонки.

### Источники и воспроизведение расчёта

Формула: [Mosteller R. D., 1987, NEJM, DOI: 10.1056/NEJM198710223171717](https://pubmed.ncbi.nlm.nih.gov/3657876/).
Референсные числа: [Kawel-Boehm N. et al., 2020, JCMR 22:87, DOI: 10.1186/s12968-020-00683-3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7734766/).
Источники проверены отдельно от ответов GPT; в доступном резервном снимке
связанной библиотеки Zotero совпадающие записи не найдены.

[Индексированные значения](heart_rr_analysis/function_analysis/reference_indexing/indexed_estimates.csv),
[использованные референсы](heart_rr_analysis/function_analysis/reference_indexing/reference_values.json)
и [манифест нового расчёта](heart_rr_analysis/function_analysis/reference_indexing/manifest.json)
содержат происхождение входов, единицы и контрольные суммы. Антропометрия
хранится в локальном файле anthropometry.local.json рядом с этими результатами;
он исключён из Git вместе с медицинскими производными данными. При его
отсутствии индексация не выполняется; значения другого испытуемого не подставляются.
Пересборка читательского представления выполняется командой
`python build_cardiac_function_report.py --presentation-only`: она проверяет
и читает прежние расчёты и отдельно рассчитывает новые индексы. Исполнение
сохранённых ячеек вывода не заменяет этот шаг.
"""


def number(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def result_html(records, availability, anthropometry):
    out = ['<div id="reference-indexing-results"><h3>Расчёт и сопоставление</h3>']
    for item in availability:
        subject = item["subject"]
        label = html.escape(subject.capitalize())
        selected = [r for r in records if r["subject"] == subject]
        if item["status"] == "missing_anthropometry":
            out.append(f'<p><b>{label}:</b> рост и масса не заданы; BSA и индексированные объёмы не рассчитываются.</p>')
            continue
        person = anthropometry[subject]
        sex = {"male": "мужской", "female": "женский"}.get(person.get("sex"), "не задан")
        age = f'{person["age_years"]:g}'.replace(".", ",") if person.get("age_years") is not None else "не задан"
        out.append(f'<p><b>Входы {label}:</b> рост {number(person["height_cm"], 1)} см, масса {number(person["weight_kg"], 1)} кг; пол — {sex}; возраст — {age} лет. BSA = √({number(person["height_cm"], 1)} × {number(person["weight_kg"], 1)} / 3600) = <b>{number(item["bsa_m2"], 3)} м²</b>. Расчёт выполняется без промежуточного округления.</p>')
        if not selected:
            out.append('<p>Пригодные исходные оценки отсутствуют.</p>')
            continue
        out.append(f'<p class="scientific-caption"><b>Таблица 9.</b> Индексированные объёмы и исходная ФВ {label}; две оценки и выбранный МРТ-референс. Для экстремумов масок УО и ФВ обозначают размах и относительный размах.</p>')
        out.append('<table class="function-table"><thead><tr><th>Камера</th><th>Показатель</th><th>Экстремумы масок</th><th>Условный цикл</th><th>Референсный интервал</th><th>Среднее ± SD</th><th>n</th></tr></thead><tbody>')
        for chamber in REFERENCE:
            for metric, title in METRICS.items():
                pair = {r["estimator"]: r for r in selected if r["chamber"] == chamber and r["metric"] == metric}
                if not pair:
                    continue
                ref = next(iter(pair.values()))
                values = [number(pair[e]["value"]) if e in pair else "—" for e in ESTIMATORS]
                extra = [f'{ref["reference_low"]}–{ref["reference_high"]}', f'{ref["reference_mean"]} ± {ref["reference_sd"]}', str(ref["reference_n"])] if "reference_low" in ref else ["не выбран", "—", "—"]
                out.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in [chamber, title, *values, *extra])+'</tr>')
        out.append('</tbody></table>')
        if item["status"] != "indexed":
            out.append('<p>Для указанного пола или возраста референс в текущем разделе не задан. Таблицы мужчин 20–29 лет автоматически не переносятся на другую группу.</p>')
            continue
        positions = {"below": "ниже", "within": "внутри", "above": "выше"}
        for chamber in REFERENCE:
            group = [r for r in selected if r["chamber"] == chamber and r["estimator"] == "periodic_composite"]
            if group:
                description = '; '.join(f'{METRICS[r["metric"]]}: {number(r["value"])} — {positions[r["reference_position"]]} интервала' for r in group)
                out.append(f'<p><b>{chamber}, условный цикл:</b> {description}. Это положение расчётных значений относительно таблицы, без клинической классификации испытуемого.</p>')
        lv = next((r for r in selected if r["chamber"] == "ЛЖ" and r["metric"] == "sv_ml" and r["estimator"] == "periodic_composite"), None)
        rv = next((r for r in selected if r["chamber"] == "ПЖ" and r["metric"] == "sv_ml" and r["estimator"] == "periodic_composite"), None)
        if lv and rv:
            out.append(f'<p>Разность модельных УО ПЖ − ЛЖ составляет {number(rv["raw_value"]-lv["raw_value"])} мл, после индексации — {number(rv["value"]-lv["value"])} мл/м². Отношение УО ПЖ/ЛЖ остаётся {number(rv["value"]/lv["value"])}. Общий масштаб BSA не устраняет исходное расхождение.</p>')
            out.append(f'<p><b>Обратный перевод референса — условный ориентир.</b> Умножение границ интервала УО/BSA ЛЖ на принятую BSA даёт {number(lv["reference_low"]*item["bsa_m2"])}–{number(lv["reference_high"]*item["bsa_m2"])} мл; аналогичный перевод среднего даёт {number(lv["reference_mean"]*item["bsa_m2"])} мл. Это масштабирование группового референса при фиксированной BSA, а не предсказание индивидуального УО по росту и массе. Разность с этим ориентиром не используется для коррекции масок.</p>')
    return ''.join(out)+'</div>'


def build_reference_section(estimates, work):
    work = Path(work)
    destination = work / "reference_indexing"
    destination.mkdir(exist_ok=True)
    config_path = destination / "anthropometry.local.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"subjects": {}}
    records, availability = index_estimates(estimates, config["subjects"])
    with (destination / "indexed_estimates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = list(dict.fromkeys(k for r in records for k in r)) or ["subject", "metric", "value"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    reference_payload = {"doi": "10.1186/s12968-020-00683-3", "url": SOURCE, "modality": "CMR",
                         "sex": "male", "age_years": [20, 29], "limit_definition": "published_pooled_mean_plus_minus_2SD",
                         "tuple_fields": ["pooled_mean", "pooled_sd", "lower", "upper", "n"], "values": REFERENCE}
    (destination / "reference_values.json").write_text(json.dumps(reference_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "conditional_indexing_descriptive_reference_comparison_not_validated",
                "source_estimates_sha256": digest(work / "ventricular_function_estimates.csv"),
                "code_sha256": digest(Path(__file__)), "anthropometry_file": config_path.name,
                "anthropometry_sha256": digest(config_path) if config_path.exists() else None,
                "anthropometry_provenance": config.get("provenance", {}), "availability": availability,
                "record_count": len(records), "formula": "sqrt(height_cm*weight_kg/3600)",
                "original_estimates_modified": False, "bsa_rounded_before_indexing": False,
                "age_sex_reference_selection_only": True, "ct_time_anthropometry_confirmed": False,
                "modality_and_contour_equivalence_confirmed": False, "independent_validation": False,
                "outputs_sha256": {n: digest(destination / n) for n in ["indexed_estimates.csv", "reference_values.json"]}}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return METHOD, result_html(records, availability, config["subjects"]), LIMITATIONS, manifest
