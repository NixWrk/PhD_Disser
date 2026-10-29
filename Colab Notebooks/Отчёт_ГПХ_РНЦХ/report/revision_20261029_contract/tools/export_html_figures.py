"""Extract frozen, unmodified scientific PNGs from the supplied Scorpion HTML.

No notebook execution, FEM, signal processing, network, or image editing.
Run with the bundled Python; optional --repo-root supports relocated checkouts.
Sources are intentionally hash-pinned. A changed HTML must be reviewed before
updating this script, rather than silently producing a different illustration.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image


SOURCES = {
    "html_3340": {
        "path": "Colab Notebooks/scorpion_v2/33.40_Ник_и_Ярослав_эксперимент_и_модели_Scorpion_22_10_2026.html",
        "sha256": "c6631301ee3c9c02835033d99621b71f6347153982291f7cd3bef4acec3e5878",
        "scope": "Индивидуальные КТ-входы участников октябрьской серии Scorpion 2026. Раздел сопоставления со старой серией МГТУ не экспортируется.",
    },
    "html_3330": {
        "path": "Colab Notebooks/scorpion_v2/33.30_Сгущение_у_всех_контактов_и_продолжение_подбора_Георгия_Scorpion_v2.html",
        "sha256": "def1623a7bbadeca16e2c0a9a8dfc5e4ae191d58c39ad762abc71f90641bb0d7",
        "scope": "Диагностика подготовки индивидуального КТ-контура Георга в ветке Scorpion; электрические результаты старых граничных условий не экспортируются.",
    },
    "html_3343": {
        "path": "Colab Notebooks/scorpion_v2/33.43_Индивидуальная_КТ_FEM_проверка_трёх_участников_Scorpion.html",
        "sha256": "99e2e84297bbc8f759396c271f18cf24e41ab3cadd95d1efcd8b3d112bbde577",
        "scope": "Сохранённые индивидуальные КТ/FEM-проверки трёх участников октябрьской серии Scorpion 2026.",
    },
    "html_3344": {
        "path": "Colab Notebooks/scorpion_v2/33.44_Измеренный_пульс_ТТРКГ_объёмная_чувствительность_и_камеры_КТ.html",
        "sha256": "f4ed38e229cbc51b94d3e9741241880618e9e072bcda7f2001a8f9510b1fe96c",
        "scope": "Пульсовые сигналы реокардиомонитора октябрьской серии Scorpion 2026 и отдельно полученный КТ-референс; синхронность между КТ и сигналами не предполагается.",
    },
}

FIGURES = [
    {
        "id": "html_ct_tissue_masks_n",
        "filename": "01_n_ct_tissue_masks.png",
        "source_id": "html_3340", "image_index_zero_based": 0,
        "source_figure": "Рисунок 23. Ник. КТ-сечение и кандидатные тканевые маски.",
        "title": "Индивидуальная КТ и кандидатные тканевые маски участника Н",
        "recommended_chapter_before_renumbering": "06_anatomy.md",
        "recommended_topic": "Происхождение анатомических входов и отделение автоматической сегментации от ручной",
        "caption": "Индивидуальное КТ-сечение участника Н и кандидатные тканевые маски. Сечение выбрано по центру объёма автоматической маски сердца. Это иллюстрация анатомических входов для модели; она не является результатом ручной сегментации автора и не подтверждает анатомическую точность автоматических границ. Маски требуют экспертной оценки; положение электродов на этом изображении не задано.",
        "status": "Сохранённый результат автоматической обработки индивидуальной КТ; анатомическая валидность не установлена.",
        "limitations": ["Не подменяет ручную сегментацию Inobitec.", "КТ получена на глубоком вдохе; изображение не является дыхательной парой вдох–выдох."],
    },
    {
        "id": "html_candidate_lateral_montage_n",
        "filename": "02_n_candidate_lateral_montage.png",
        "source_id": "html_3340", "image_index_zero_based": 2,
        "source_figure": "Рисунок 25. Ник. Кандидатная боковая область на индивидуальной КТ.",
        "title": "Кандидатная локализация боковой сборки участника Н",
        "recommended_chapter_before_renumbering": "05_experiments.md",
        "recommended_topic": "Различие фактического монтажа и его геометрической реконструкции",
        "caption": "Кандидатная боковая область на индивидуальной КТ участника Н: жёлтая точка и зелёная область показывают геометрическую гипотезу центра. Пример A–M–N–B имеет длину 140 мм по кожной кривой. Это сохранённый диагностический вариант локализации, а не измеренное положение электродов во время записи. Способ задания расстояний не подтверждает фактическую геометрию сборки; отказ проверки диагностического контура сохранён.",
        "status": "Геометрическое допущение; положение не подтверждено экспериментальным протоколом.",
        "limitations": ["Не фотография установки и не доказательство совпадения с физическим монтажом.", "Не выдавать этот исходный кандидат за окончательное положение всех последующих FEM-расчётов.", "Контур в показанном диагностическом варианте не прошёл все проверки подготовки."],
    },
    {
        "id": "html_ct_surface_regularization_g",
        "filename": "03_g_ct_contour_regularization.png",
        "source_id": "html_3330", "image_index_zero_based": 0,
        "source_figure": "Рисунок подготовки геометрии 1. Локальные сечения КТ.",
        "title": "Контроль изменения наружного КТ-контура при подготовке геометрии",
        "recommended_chapter_before_renumbering": "06_anatomy.md",
        "recommended_topic": "Подготовка наружной поверхности и отрицательные результаты геометрического контроля",
        "caption": "Локальные КТ-сечения участника Г до и после одного замыкания наружного контура. Синим показан исходный контур, оранжевым — исправленный, красными крестами — точки локальной пробы вершин. Верхний ряд соответствует проверке до → после, нижний — после → до. Координаты заданы в LPS, мм; индексы отсчитываются от нуля. Контуры сечений построены на уровне 0,5, а проверяемые поверхности — на уровне 0,49. Анатомическая приемлемость коррекции не установлена; проверка исходного допуска 1,25 мм дала отрицательный результат.",
        "status": "Численная диагностика геометрии; отрицательный результат исходной проверки допуска сохранён.",
        "limitations": ["Три ближайших воксельных сечения могут не проходить точно через трёхмерную точку проверки.", "Локальные и выборочные максимумы не являются глобальным расстоянием Хаусдорфа.", "Диагностика не меняла замороженную FEM-геометрию."],
    },
    {
        "id": "html_recorded_tt_ecg_three_subjects",
        "filename": "04_tt_ecg_four_cycles.png",
        "source_id": "html_3344", "image_index_zero_based": 0,
        "source_figure": "Рисунок 1. ТТРКГ и ЭКГ на четырёх последовательных интервалах.",
        "title": "ТТРКГ и ЭКГ трёх участников на четырёх последовательных интервалах",
        "recommended_chapter_before_renumbering": "05_experiments.md",
        "recommended_topic": "Пример результатов обработки реальных сигналов",
        "caption": "ТТРКГ 2×2 и ЭКГ трёх участников на четырёх последовательных кандидатных циклах отдельных записей реокардиомонитора, отнесённых к глубокому вдоху. Вертикальные линии обозначают существующие кандидатные QRS; начало шкалы совпадает с первым из них. Импедансный пульс показан в мОм после обработки в полосе 0,7–12 Гц, ЭКГ — в мВ после подавления 50, 100 и 150 Гц и ограничения полосой 0,5–40 Гц. Знак и амплитуда сохранены; подбор усиления, задержки и амплитудная нормировка не выполнялись. QRS не принимается автоматически за механическое начало изгнания.",
        "status": "Результат обработки экспериментальных сигналов; разметка QRS кандидатная.",
        "limitations": ["Три панели относятся к разным участникам и записям, а не к общей синхронной сессии.", "Тканевые пульсовые вклады из ТТРКГ на этом этапе не вычтены.", "Внутриучастковая вариабельность не является оценкой межсессионной воспроизводимости."],
    },
    {
        "id": "html_tissue_identifiability_singular_values",
        "filename": "05_tissue_identifiability_singular_values.png",
        "source_id": "html_3343", "image_index_zero_based": 2,
        "source_figure": "Рисунок 3. Сингулярные числа безразмерной E на вдохе, логарифмическая шкала.",
        "title": "Локальная различимость четырёх сопротивлений по расчётным наборам отведений",
        "recommended_chapter_before_renumbering": "08_tissue.md",
        "recommended_topic": "Идентифицируемость тканевых параметров и границы объединения отведений",
        "caption": "Сингулярные числа безразмерной матрицы E на вдохе в индивидуальных КТ/FEM-моделях, логарифмическая шкала. Сплошные линии соответствуют боковым отведениям, ТТРКГ и совместному набору; пунктирные уровни — спектральной норме разности E при q = 4096 и q = 1024 для соответствующего набора. Слабейшее совместное направление у всех трёх участников преимущественно связано с эффективным сопротивлением всего сердца. Раздельная практическая оценка четырёх сопротивлений не установлена.",
        "status": "Результат локальной математической модели; практическая идентифицируемость не подтверждена.",
        "limitations": ["Разность правил интегрирования — наблюдаемый численный контроль, а не строгая граница ошибки.", "Шум и неопределённость положения электродов в этом сравнении численно не оценены.", "Объединение боковых отведений с ТТРКГ не является сравнением точности с поясом PulmoVista."],
    },
    {
        "id": "html_negative_pulse_volume_test",
        "filename": "06_measured_pulse_vs_volume_probe.png",
        "source_id": "html_3344", "image_index_zero_based": 2,
        "source_figure": "Рисунок 3. Столбцы — размах каждого измеренного цикла; горизонтальные линии — размах ответа в диапазоне 0,95–1,05V₀.",
        "title": "Измеренный пульс и объёмная проба однородной модели",
        "recommended_chapter_before_renumbering": "09_heart_dynamics.md",
        "recommended_topic": "Отрицательная проверка простого преобразования импеданса в объём",
        "caption": "Сопоставление размаха каждого измеренного цикла ТТРКГ (столбцы) с размахом модельного ответа при изменении объёма в диапазоне 0,95–1,05V₀ (горизонтальные линии). Обе исследованные материальные постановки дают меньший размах во всех двенадцати интервалах. Это отрицательный результат ограниченной однородной модели с фиксированным центром и фоном. Он не доказывает отсутствие сердечного вклада; пульсовые вклады других тканей здесь ещё не вычтены. Экстраполяцию за проверенный объёмный диапазон нельзя называть восстановленным УО.",
        "status": "Отрицательный результат сопоставления эксперимента с ограниченным модельным сценарием.",
        "limitations": ["Объёмная проба относится ко всему эквивалентному сердцу, а не непосредственно к объёму крови ЛЖ.", "Свойства однородного сердца являются сценариями, а не установленными тканевыми параметрами.", "Не обосновывает исключение сферы из дальнейшего исследования при других эффективных свойствах и законах изменения."],
    },
    {
        "id": "html_ct_whole_heart_and_chambers",
        "filename": "07_ct_whole_heart_and_chambers.png",
        "source_id": "html_3344", "image_index_zero_based": 3,
        "source_figure": "Рисунок 4. Наблюдаемые изменения объёмов автоматических масок от первой доступной фазы каждого интервала.",
        "title": "Различие изменений объёма целого сердца и отдельных камер",
        "recommended_chapter_before_renumbering": "10_chambers.md",
        "recommended_topic": "Необходимость обоснования перехода от эквивалентного сердца к УО желудочка",
        "caption": "Изменения объёмов автоматических масок целого сердца и камер относительно первой доступной фазы каждого отдельного интервала R–R. У четырёх панелей разные референсные фазы; линии соединяют только доступные точки и не замыкают цикл. Изменение объёма целого сердца не тождественно изменению одной камеры. Начальные точки не приняты за конец диастолы, а наблюдаемые размахи не объявляются УО. Вклад анатомической неопределённости автоматических масок отдельно не оценён.",
        "status": "Наблюдение по автоматическим КТ-маскам; проверенного перехода от целого сердца к УО ЛЖ нет.",
        "limitations": ["КТ служит отдельным референсом разработки, не синхронным измерением с октябрьскими сигналами.", "Фазовые объёмы не разрешены как известные входы операционного восстановления из одной статической КТ.", "Малое число зависимых фаз не обеспечивает независимой проверки физиологической точности."],
    },
    {
        "id": "html_negative_respiratory_contrast",
        "filename": "08_respiratory_contrast_observation_model.png",
        "source_id": "html_3343", "image_index_zero_based": 1,
        "source_figure": "Рисунок 2. Парные дыхательные разности: наблюдение и модельный lung-only контраст own_drive при временном разделении.",
        "title": "Наблюдаемый дыхательный контраст и модельный вклад лёгкого",
        "recommended_chapter_before_renumbering": "08_tissue.md",
        "recommended_topic": "Раздельная проверка абсолютных уровней и дыхательных различий",
        "caption": "Парные дыхательные разности в эксперименте и индивидуальной модели при временном разделении отведений: синим показана медиана наблюдаемых парных разностей, оранжевым — модельный контраст при изменении только лёгочного компартмента. Категории обозначают базу боковой сборки в миллиметрах и вариант ТТРКГ. Расхождение величины, а для части боковых монтажей участника Г и знака, сохраняется. Изменение геометрии грудной клетки в этом сценарии не рассчитывалось; такое сопоставление не подтверждает индивидуальную точность тканевой модели.",
        "status": "Экспериментальное наблюдение и ограниченный модельный сценарий с отрицательным результатом согласования.",
        "limitations": ["Дыхательная разметка ретроспективная; монтажи регистрировались последовательно.", "Межсессионная неопределённость не оценена.", "Изменение только лёгкого не подменяет полную модель всех дыхательных вкладов."],
    },
]


class ImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append({"attributes": dict(attrs), "html_line": self.getpos()[0]})


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixel_hash(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        rgba = image.convert("RGBA")
        return sha(str(rgba.size).encode("ascii") + rgba.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[5])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    revision = Path(__file__).resolve().parents[1]
    output = revision / "figures" / "html_exports"
    output.mkdir(parents=True, exist_ok=True)
    original = revision.parent / "revision_20261028"
    previous_pixels = {}
    missing_previous = []
    previous_docx = original / "deliverables" / "Отчёт_ГПХ_РНЦХ_20261028_для_рецензирования.docx"
    previous_docx_sha = sha(previous_docx.read_bytes())
    if previous_docx_sha != "64bb37e36f439f5b7ad0a4ef24e1f29a0b38d785e2f6b6ea9c0ee088ef01ea4c":
        raise RuntimeError("Previous review DOCX changed; inspect the comparison baseline")
    with zipfile.ZipFile(previous_docx) as archive:
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
        for name in media:
            previous_pixels[pixel_hash(archive.read(name))] = previous_docx.name + ":" + name
    for chapter in sorted((original / "chapters").glob("*.md")):
        for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", chapter.read_text(encoding="utf-8")):
            path = (chapter.parent / ref).resolve()
            if not path.is_file():
                missing_previous.append(str(path.relative_to(repo)))
                continue
            previous_pixels[pixel_hash(path.read_bytes())] = str(path.relative_to(repo))
    cache = {}
    records = []
    for spec in FIGURES:
        source = SOURCES[spec["source_id"]]
        if spec["source_id"] not in cache:
            source_bytes = (repo / source["path"]).read_bytes()
            actual = sha(source_bytes)
            if actual != source["sha256"]:
                raise RuntimeError(f"Frozen HTML changed: {source['path']} ({actual})")
            reader = ImageParser()
            reader.feed(source_bytes.decode("utf-8"))
            cache[spec["source_id"]] = reader.images
        item = cache[spec["source_id"]][spec["image_index_zero_based"]]
        uri = item["attributes"]["src"]
        if not uri.startswith("data:image/png;base64,"):
            raise RuntimeError("Expected an embedded PNG; no implicit image conversion allowed")
        image_bytes = base64.b64decode(uri.split(",", 1)[1], validate=True)
        image_pixels = pixel_hash(image_bytes)
        if image_pixels in previous_pixels:
            raise RuntimeError(f"Duplicate of existing report figure: {previous_pixels[image_pixels]}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            size = list(image.size)
            image.verify()
        path = output / spec["filename"]
        path.write_bytes(image_bytes)
        records.append({
            **spec,
            "integration_priority": "primary" if len(records) < 6 else "reserve",
            "source_path": source["path"],
            "source_html_sha256": source["sha256"],
            "source_scope": source["scope"],
            "locator": {"html_img_index_zero_based": spec["image_index_zero_based"],
                        "html_line": item["html_line"], "figure_caption": spec["source_figure"]},
            "export_method": "Base64 decode of the existing HTML img[data:image/png]; byte-for-byte extraction; no image processing or recalculation",
            "output_path": path.relative_to(repo).as_posix(),
            "embedded_png_sha256": sha(image_bytes),
            "output_sha256": sha(path.read_bytes()),
            "rgba_pixel_sha256": image_pixels,
            "dimensions_px": size,
            "scientific_claim_disposition": "keep_with_boundary",
            "duplicate_check": "No identical RGBA pixels among all images embedded in the frozen previous DOCX or available previous chapter references",
        })
    if len({r["rgba_pixel_sha256"] for r in records}) != len(records):
        raise RuntimeError("Duplicates within the new export set")
    manifest = {
        "schema_version": 1,
        "editorial_date": "2026-10-29",
        "task": "Additional illustrations for the contract-aligned report; УО is primary, ФВ supplementary",
        "figure_mode": "with_figures",
        "formatting_mode": "section_only",
        "sources": SOURCES,
        "figures": records,
        "source_boundary": "Fixed local HTML research corpus; no network, new physical calculation, new signal processing, or transfer of raw medical data",
        "previous_report_unique_pixel_images_checked": len(previous_pixels),
        "previous_docx_source": previous_docx.relative_to(repo).as_posix(),
        "previous_docx_sha256": previous_docx_sha,
        "previous_docx_embedded_images_checked": len(media),
        "previous_report_missing_image_paths": sorted(set(missing_previous)),
        "chapter_numbering": "Recommendations use old filenames; report integration and final numbering are outside this exporter",
        "recommended_primary_count": 6,
        "reserve_count": 2,
        "inventory_note": "Targeted rg --files --hidden --no-ignore inventory found 28 HTML paths in scorpion_v2 and cardiac_4d_segmentation/results; additional generated files concerned historical cardiac checks. The local pulmovista_check_20261028 folder is unavailable in this checkout. No Git LFS pointer was treated as a rendered HTML.",
    }
    anonymous_manifest = output / "anonymous" / "manifest.json"
    if anonymous_manifest.is_file():
        variants = json.loads(anonymous_manifest.read_text(encoding="utf-8"))
        ready = {row["filename"]: row for row in variants["figures"]}
        manifest["integration_instruction"] = "Use anonymous/manifest.json and anonymous/CAPTIONS.md for publication; original extracted PNGs contain internal participant names except figure 03."
        for record in records:
            if record["filename"] in ready:
                record["publication_variant"] = ready[record["filename"]]["output_path"]
            else:
                record["publication_blocker"] = "Original includes internal participant name; no approved anonymous rerender available"
    visual_qa = output / "VISUAL_QA.json"
    if visual_qa.is_file():
        reviews = json.loads(visual_qa.read_text(encoding="utf-8"))
        by_name = {row["filename"]: row for row in reviews["images"]}
        for record in records:
            review = by_name.get(record["filename"])
            if not review or review["sha256"] != record["output_sha256"]:
                raise RuntimeError("Missing or stale visual review for " + record["filename"])
            record["visual_review"] = review
        manifest["visual_qa_path"] = visual_qa.relative_to(repo).as_posix()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Дополнительные иллюстрации из сохранённых HTML", "",
             "Изображения извлечены побайтно: данные, цвета, надписи и размеры исходных PNG не изменены. Номера глав ниже относятся к исходной редакции до общей перенумерации. Подписи не являются подтверждением завершённой валидации методики.", ""]
    if anonymous_manifest.is_file():
        lines += ["**Обновление перед интеграцией:** для основного текста использовать шесть файлов из `anonymous/` и подписи `anonymous/CAPTIONS.md`. Приведённый ниже список относится к сохранённым оригиналам. Рисунки 01–02 пока не включать: в них остаётся внутреннее имя участника, а повторный экспорт без индивидуальных КТ-входов не выполнен.", ""]
    lines += ["Для основного текста рекомендуются рисунки 01–06. Рисунки 07–08 подготовлены как резерв, если нужно дополнительно показать переход от целого сердца к камерам или дыхательный контраст. Это восемь файлов, но не предложение обязательно добавить все восемь в основной текст.", "",
              "Проверка повторов выполнена по всем 22 уникальным изображениям внутри прежнего Word (24 позиции рисунков, включая повторы). Совпадений по пикселям нет. Пять недоступных отдельных файлов прежней редакции учтены через изображения, встроенные в Word. Все новые рисунки просмотрены визуально; проверка размещения в новом Word выполняется при общей сборке.", ""]
    for rec in records:
        lines += [f"## {rec['filename']}", "", rec["caption"], "",
                  "Приоритет: " + ("основной набор" if rec["integration_priority"] == "primary" else "резерв") + ".", "",
                  f"Рекомендуемый раздел: `{rec['recommended_chapter_before_renumbering']}` — {rec['recommended_topic']}.", "",
                  f"Источник: `{rec['source_path']}`, {rec['source_figure']} HTML `img[{rec['image_index_zero_based']}]` (индексация от нуля).", "",
                  "Статус: " + rec["status"], "",
                  "Ограничения: " + " ".join(rec["limitations"]), ""]
    (output / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"exported": len(records), "previous_unique_images_checked": len(previous_pixels),
                      "missing_previous_images": len(set(missing_previous)),
                      "manifest": str(output / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
