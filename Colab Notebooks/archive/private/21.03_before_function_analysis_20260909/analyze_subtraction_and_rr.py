#!/usr/bin/env python
"""Show whole-heart minus blood and quantify chamber volumes over DICOM R-R phase."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import nbformat
import numpy as np
import plotly.graph_objects as go
from nibabel.affines import apply_affine
from plotly.io import to_html
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator
from skimage import measure
from cardiac_volume_time_plots import build_volume_figure_ms
from cardiac_synchronized_view import build_synchronized_cardiac_figure


CHAMBERS = (
    ("heart_atrium_left", "Левое предсердие", "#3A86FF"),
    ("heart_ventricle_left", "Левый желудочек", "#0057B8"),
    ("heart_atrium_right", "Правое предсердие", "#FF8C42"),
    ("heart_ventricle_right", "Правый желудочек", "#D62828"),
)

CARDIAC_PHASE_BANDS = (
    (0.0, 45.0, "Уменьшение объёма<br>желудочков", "rgba(214, 39, 40, 0.07)"),
    (45.0, 80.0, "Раннее и среднее<br>наполнение", "rgba(42, 157, 143, 0.07)"),
    (80.0, 100.0, "Поздняя<br>диастола", "rgba(131, 56, 236, 0.07)"),
)


@dataclass(frozen=True)
class SurfaceStyle:
    key: str
    label: str
    color: str
    opacity: float


SURFACES = (
    SurfaceStyle("heart_minus_blood", "Производная остаточная маска", "#B9A27A", 0.42),
    SurfaceStyle("heart_atrium_left", "Кровь: левое предсердие", "#3A86FF", 1.0),
    SurfaceStyle("heart_ventricle_left", "Кровь: левый желудочек", "#0057B8", 1.0),
    SurfaceStyle("heart_atrium_right", "Кровь: правое предсердие", "#FF8C42", 1.0),
    SurfaceStyle("heart_ventricle_right", "Кровь: правый желудочек", "#D62828", 1.0),
    SurfaceStyle("myocardium", "Миокард ЛЖ и перегородки", "#F4A261", 0.72),
)

NOTEBOOK_CONTEXT_MD = """# Проверка автоматической сегментации сердца в 4D-КТ

**Статус:** исследовательский вычислительный тест; автоматические маски ожидают
покадровой экспертной проверки.

## Задача и происхождение масок

Ноутбук проверяет динамику формы сердца и объёмов четырёх камер на 39 фазах
контрастной 4D-КТ Adam, Nix и Georg. Ручная сегментация Inobitec остаётся
каноническим анатомическим референсом проекта, но в этот расчёт ещё не включена.

TotalSegmentator 2.18.0 запускался в независимых задачах `total/heart` и
`heartchambers_highres`. Маска крови — объединение четырёх камер без аорты и
лёгочной артерии. Геометрическая разность «всё сердце минус кровь» не является
валидированной маской полного миокарда. Класс `heart_myocardium` охватывает
преимущественно миокард ЛЖ и межжелудочковую перегородку.

## Что непосредственно получено из DICOM

Для каждого испытуемого ниже приведены: каталог исходной серии и число файлов,
`NominalPercentageOfCardiacPhase`, приватные метки GE `RpeakTimeStamps
(0049,100C)`, рассчитанные по ним R–R и ЧСС, распределение
`AvgHeartRateForImage`, показатели ЧСС при подтверждении и до него, временное
разрешение реконструкции, контроль по `TemporalCenterViewAngle`, тип монитора,
нерасшифрованные приватные коды и наличие GE ECG Report. Приватные коды
перенесены дословно: их числовым значениям не приписан смысл без документации
производителя. Стандартной DICOM-последовательности `WaveformSequence` нет.

У Adam и Nix доступны только растровые полосы из GE ECG Report с предупреждением
`Not for Diagnostic ECG use`; они не содержат калиброванных цифровых отсчётов.
У Georg отчёт ECG Report не найден.

Для каждого испытуемого 3D-маски, схематическая кривая P–QRS–T и графики
объёмов объединены в одном интерактивном представлении. Выбор фазы одним
ползунком одновременно обновляет геометрию и выделяет соответствующую точку
каждой камеры. Кривая P–QRS–T служит только ориентиром внутри R–R и не является
индивидуальной ЭКГ. Точное сопоставление КТ-фаз с электрическими и
механическими фазами сердца составляет отдельный этап проверки.

## Объединение соседних сердечных циклов

Основной график использует время в миллисекундах. Для Adam и Nix оно отсчитано
от первой R-метки и показано непрерывно через границу соседних интервалов; в
скобках у каждой точки указан процент соответствующего R–R. Для Georg короткий
и длинный интервалы показаны на двух отдельных панелях, каждая со своей шкалой
миллисекунд после R-триггера. Одинаковый процент двух разных R–R не считается
одинаковой механической фазой.

Для Adam и Nix объединение в отдельной численной модели предварительно
разрешено, поскольку относительное различие медианной ЧСС соседних циклов не
превышает рабочий порог 10%. Этот порог методический и ещё не валидирован
физиологически.

Средняя кривая построена периодической моделью Фурье с двумя гармониками.
Модель допускает отдельную аддитивную поправку для каждого записанного цикла;
поправки центрированы относительно нуля и регуляризованы, чтобы частично
наблюдаемый второй цикл не определял форму кривой произвольно. Равенство
V(0%) = V(100%) следует из периодического базиса, а не из добавления фиктивной
точки. Все исходные точки и принадлежность циклу сохранены. Остаточная ошибка
модели и рассчитанные межцикловые поправки записаны в итоговую таблицу.

Это описательная аппроксимация, а не доверительный интервал и не доказательство
того, что два цикла физиологически тождественны. В исходном представлении PCHIP
по-прежнему строится отдельно внутри наблюдаемого участка каждого цикла.

## Georg: почему объединение запрещено

Во всех 3840 изображениях серии 309 сохранена одна тройка R-меток: −0,026;
0,465 и 1,624 с. Из неё следуют R–R 491 и 1159 мс, или 122,20 и
51,77 уд/мин. `AvgHeartRateForImage` независимо повторяет эту пару: 122 уд/мин
в 2048 изображениях и 52 уд/мин в 1792 изображениях. Максимальное расхождение
проверки по `TemporalCenterViewAngle` составляет 7,66 мс. Следовательно,
коротко-длинная пара использована реконструкцией и не возникла из-за нашего
объединения таблиц или нормирования графика.

Пара совместима как минимум с тремя классами причин: преждевременный комплекс
с последующей паузой; лишний или пропущенный синхроимпульс при распознавании
R-зубца; редактирование/отбор циклов системой реконструкции. ЧСС до
подтверждения у Georg изменялась сильнее, чем у Adam и Nix (SD 10,1 против
примерно 5,5 уд/мин), но это не устанавливает механизм и не является диагнозом.
Без соседних R-меток, исходной цифровой ЭКГ и журнала ECG-edit различить эти
варианты нельзя.

Первичный анализ Georg поэтому сохраняет оба участка раздельно и не
экстраполирует их в общий цикл. Рабочая проверка должна идти в таком порядке:

1. запросить с консоли сканера исходную полосу/журнал кардиосинхронизации и,
   если сохранялись, необработанные временные метки синхроимпульсов;
2. проверить журнал ECG-edit/arrhythmia rejection и расшифровать приватные коды
   по документации конкретной версии GE;
3. посрезово сравнить короткий и длинный участки на двойные контуры, разрывы
   движения камер и коронарных артерий;
4. повторить расчёт как анализ чувствительности: оба участка раздельно;
   только длинный участок; абсолютное время после R. Ни один вариант не считать
   основным без внешнего подтверждения;
5. если дополнительные данные недоступны, исключить Georg из композитных
   фазовых оценок, но оставить как отдельный случай контроля качества с отрицательным результатом.

## Ограничения сегментации и неопределённости

Межмодельный Dice 0,906–0,947 показывает согласие автоматических выходов, а не
точность относительно анатомии. Соседний скачок объёма модельного миокарда
достигает 21,5–26,2%, отдельных камер — 44,1%. Клапанные плоскости, устья
сосудов, ушки предсердий и тонкие стенки требуют посрезовой проверки.

Доверительный интервал точности сегментации пока не рассчитан. Нужны независимые
разметки двух экспертов, повторная разметка части кадров одним экспертом и
метрики объёма, Dice, поверхностного Dice, среднего симметричного расстояния и
95-го процентиля расстояния Хаусдорфа. Фазы одного сердца зависимы, поэтому
интервалы следует строить кластерным бутстрэпом по испытуемым. При трёх
испытуемых популяционный интервал будет только пилотным.

## Литературные основания

- [DICOM PS3.3, Cardiac Synchronization Macro](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html): `NominalPercentageOfCardiacPhase` задаёт номинальное время относительно предшествующего R-зубца в процентах номинального R–R.
- [Matsutani et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18577814/): при аритмиях в кардиальной КТ применяют ECG-edit, включая удаление короткого R–R и выбор фазы из длинного интервала.
- [Kondo et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24582039/): желудочковые экстрасистолы могут сопровождаться постэкстрасистолической компенсаторной паузой; это возможная, но не доказанная здесь причина коротко-длинной пары.
- [Celeng et al., 2016](https://doi.org/10.1007/s10554-015-0755-2): одинаковый процент R–R может соответствовать разному абсолютному времени механического события при различной ЧСС.
- [Robinson et al., 2024](https://doi.org/10.1186/s44156-024-00051-2): механические фазы определяются клапанными событиями, а не одним фиксированным процентом R–R.
- [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z): метрики валидации сегментации следует выбирать с учётом клинической задачи.
- [Warfield et al., 2004](https://doi.org/10.1109/TMI.2004.828354): STAPLE позволяет вероятностно объединять экспертные разметки.
"""


SCIENTIFIC_CONTEXT_HTML = """
<section class="card context">
<h2>Научная постановка и статус</h2>
<p><strong>Статус:</strong> исследовательский тест TotalSegmentator 2.18.0 на
39 фазах 4D-КТ Adam, Nix и Georg. Ручная сегментация Inobitec в расчёт ещё не
включена; поверхности и объёмные кривые требуют экспертной проверки.</p>
<p>Маска всего сердца получена задачей <code>total/heart</code>, четыре камеры и
<code>heart_myocardium</code> — задачей <code>heartchambers_highres</code>.
Разность «всё сердце минус кровь» не является проверенной маской полного
миокарда; <code>heart_myocardium</code> охватывает преимущественно ЛЖ и
межжелудочковую перегородку.</p>
<h3>DICOM и объединение циклов</h3>
<p>Ниже для каждого испытуемого приведены все использованные поля
кардиосинхронизации DICOM. Приватные числовые коды GE сохранены дословно и не
расшифрованы без документации производителя. <code>RpeakTimeStamps</code>
являются метками синхроимпульсов реконструкции, а не цифровой ЭКГ.
<code>WaveformSequence</code> отсутствует.</p>
<p>Основные графики объёмов построены по времени в миллисекундах. Процент
соответствующего R–R указан в скобках у каждой измеренной точки. Для Adam и Nix
сохранена непрерывная временная ось от первой R-метки, но интерполяция
выполнена отдельно внутри каждого интервала. Для Georg короткий и длинный
интервалы вынесены на две панели с независимыми шкалами времени после
R-триггера.</p>
<p>Для каждого испытуемого 3D-маски, схематическая кривая P–QRS–T и кривые
объёмов объединены в одном интерактивном представлении. Один ползунок
одновременно меняет 3D-фазу и выделяет соответствующую точку каждой камеры на
графике. Кривая P–QRS–T является временным ориентиром внутри R–R и не заменяет
индивидуальную ЭКГ. Точное сопоставление электрических, клапанных и объёмных
событий требует отдельной проверки.</p>
<p>Для Adam и Nix соседние циклы проходят предварительный порог различия ЧСС
10%. Их общая кривая получена периодической моделью Фурье с двумя гармониками
и регуляризованными аддитивными поправками на цикл. Равенство 0=100% задаётся
периодическим базисом. Все исходные точки сохранены. Остаток модели и
межцикловые поправки являются описательными величинами, а не доверительным
интервалом точности.</p>
<h3>Отдельный QC-случай Georg</h3>
<p>Во всех 3840 изображениях серии 309 содержатся R-метки −0,026; 0,465 и
1,624 с, то есть интервалы 491 и 1159 мс (122,20 и 51,77 уд/мин).
<code>AvgHeartRateForImage</code> повторяет эту пару: 122 уд/мин в 2048
изображениях и 52 уд/мин в 1792. Остаток проверки по
<code>TemporalCenterViewAngle</code> не превышает 7,66 мс. Это подтверждает,
что коротко-длинная пара использована реконструкцией, но не объясняет её
происхождение.</p>
<p>Совместимы по меньшей мере три гипотезы: преждевременный комплекс с паузой;
лишний или пропущенный R-триггер; ECG-edit/отбор циклов реконструкцией.
Без цифровой ЭКГ, соседних R-меток и журнала ECG-edit различить их нельзя.
Поэтому циклы Georg показаны раздельно, без общего замыкания и экстраполяции.
Если дополнительные данные сканера недоступны, Georg следует исключить из
композитных фазовых оценок и сохранить как отдельный QC-случай.</p>
<h3>Ограничения</h3>
<p>Межмодельный Dice 0,906–0,947 не является точностью относительно анатомии.
Доверительный интервал точности не рассчитан: нужны независимые ручные маски,
повторная разметка и кластерный бутстрэп по испытуемым. При трёх добровольцах
популяционный интервал будет только пилотным.</p>
<h3>Литературные основания</h3>
<ul>
<li><a href="https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html">DICOM PS3.3</a> — определение номинальной фазы;</li>
<li><a href="https://pubmed.ncbi.nlm.nih.gov/18577814/">Matsutani et al., 2008</a> — ECG-edit при аритмиях в кардиальной КТ;</li>
<li><a href="https://pubmed.ncbi.nlm.nih.gov/24582039/">Kondo et al., 2014</a> — возможная постэкстрасистолическая пауза при PVC;</li>
<li><a href="https://doi.org/10.1007/s10554-015-0755-2">Celeng et al., 2016</a> — процент R–R и абсолютное время при разной ЧСС;</li>
<li><a href="https://doi.org/10.1038/s41592-023-02151-z">Metrics Reloaded</a> — выбор метрик сегментации по задаче.</li>
</ul>
</section>
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4)


def save_mask(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, path)


def volume_ml(mask: np.ndarray, affine: np.ndarray) -> float:
    return float(np.count_nonzero(mask) * abs(np.linalg.det(affine[:3, :3])) / 1000.0)


def surface(mask: np.ndarray, affine: np.ndarray, step_size: int = 6) -> dict[str, np.ndarray]:
    if not np.any(mask):
        return {name: np.asarray([], dtype=np.float32 if name in "xyz" else np.int32) for name in ("x", "y", "z", "i", "j", "k")}
    lower = []
    upper = []
    for axis in range(mask.ndim):
        projection_axes = tuple(index for index in range(mask.ndim) if index != axis)
        occupied = np.flatnonzero(np.any(mask, axis=projection_axes))
        lower.append(int(occupied[0]))
        upper.append(int(occupied[-1]) + 1)
    crop = tuple(slice(start, stop) for start, stop in zip(lower, upper))
    padded = np.pad(mask[crop].astype(np.uint8), 1, mode="constant")
    vertices, faces, _, _ = measure.marching_cubes(padded, 0.5, step_size=step_size, allow_degenerate=False)
    vertices = np.round(apply_affine(affine, vertices - 1.0 + np.asarray(lower)), 2).astype(np.float32)
    faces = faces.astype(np.int32, copy=False)
    return {"x": vertices[:, 0], "y": vertices[:, 1], "z": vertices[:, 2], "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2]}


def trace(mesh: dict[str, np.ndarray], style: SurfaceStyle, showlegend: bool) -> go.Mesh3d:
    return go.Mesh3d(
        **mesh, name=style.label, legendgroup=style.key, color=style.color,
        opacity=style.opacity, flatshading=False, showlegend=showlegend,
        lighting={"ambient": 0.6, "diffuse": 0.7, "specular": 0.15, "roughness": 0.7},
        hovertemplate=style.label + "<extra></extra>",
    )


def axis_range(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    margin = max(2.0, 0.03 * (high - low))
    return [low - margin, high + margin]

def schematic_ecg(x: np.ndarray) -> np.ndarray:
    """Illustrative periodic P-QRS-T waveform; not a measured signal."""
    phase = np.mod(x, 100.0)
    def pulse(center: float, width: float, amplitude: float) -> np.ndarray:
        distance = (phase - center + 50.0) % 100.0 - 50.0
        return amplitude * np.exp(-0.5 * (distance / width) ** 2)
    return (pulse(80, 4.5, 0.16) + pulse(97.5, 1.0, -0.20) +
            pulse(0, 0.75, 1.15) + pulse(2.2, 1.2, -0.30) + pulse(25, 7.0, 0.34))


def phase_band_label(phase_percent: float) -> str:
    for start, end, label, _ in CARDIAC_PHASE_BANDS:
        if start <= phase_percent < end or phase_percent == 100.0 and end == 100.0:
            return label.replace("<br>", " ")
    raise ValueError(f"Unsupported normalized phase: {phase_percent}")


def assess_cycle_merge(phases: list[dict], maximum_relative_hr_difference: float = 0.10) -> dict:
    cycle_indices = sorted({int(phase["cycle_index"]) for phase in phases})
    heart_rate_by_cycle = {
        cycle_index: float(np.median([
            phase["image_heart_rate_bpm"]
            for phase in phases if int(phase["cycle_index"]) == cycle_index
        ]))
        for cycle_index in cycle_indices
    }
    rates = np.asarray(list(heart_rate_by_cycle.values()), dtype=float)
    relative_difference = 0.0 if len(rates) < 2 else float((rates.max() - rates.min()) / rates.mean())
    allowed = len(rates) < 2 or relative_difference <= maximum_relative_hr_difference
    return {
        "allowed": allowed,
        "heart_rate_by_cycle_bpm": heart_rate_by_cycle,
        "maximum_relative_heart_rate_difference": relative_difference,
        "threshold": maximum_relative_hr_difference,
        "status": "provisional_merge_allowed" if allowed else "merge_rejected_heart_rate_inconsistency",
    }


def within_cycle_profile(phases: list[dict], chamber_key: str, cycle_index: int) -> dict:
    observations = sorted(({
        "phase_id": phase["phase_id"],
        "raw_rr_percent": float(phase["rr_percent"]),
        "phase_percent": float(phase["phase_percent_within_cycle"]),
        "volume_ml": float(phase["chambers_ml"][chamber_key]),
    } for phase in phases if int(phase["cycle_index"]) == cycle_index), key=lambda item: item["phase_percent"])
    x = np.asarray([item["phase_percent"] for item in observations], dtype=float)
    y = np.asarray([item["volume_ml"] for item in observations], dtype=float)
    dense_x = np.arange(float(np.ceil(x.min())), float(np.floor(x.max())) + 1.0, 1.0)
    if len(x) >= 2:
        dense_y = np.asarray(PchipInterpolator(x, y, extrapolate=False)(dense_x), dtype=float)
    else:
        dense_x = x.copy()
        dense_y = y.copy()
    return {"observations": observations, "phase_percent": dense_x, "volume_ml": dense_y}



def periodic_cycle_profile(
    phases: list[dict],
    chamber_key: str,
    harmonics: int = 2,
    harmonic_ridge: float = 0.10,
    cycle_contrast_ridge: float = 1.0,
) -> dict:
    """Fit a periodic mean curve while allowing shrunken beat-specific offsets.

    The fit is intentionally descriptive. It is not an accuracy estimate or a
    confidence interval: there are only two partially sampled adjacent beats.
    """
    observations = sorted(({
        "phase_id": phase["phase_id"],
        "cycle_index": int(phase["cycle_index"]),
        "raw_rr_percent": float(phase["rr_percent"]),
        "phase_percent": float(phase["phase_percent_within_cycle"]),
        "volume_ml": float(phase["chambers_ml"][chamber_key]),
    } for phase in phases), key=lambda item: (item["phase_percent"], item["cycle_index"]))
    x = np.asarray([item["phase_percent"] for item in observations], dtype=float)
    y = np.asarray([item["volume_ml"] for item in observations], dtype=float)
    cycles = np.asarray([item["cycle_index"] for item in observations], dtype=int)
    unique_cycles = sorted(set(cycles.tolist()))
    if len(x) < 2:
        raise RuntimeError(f"At least two normalized phases are required for {chamber_key}")

    mean_y = float(np.mean(y))
    scale_y = float(np.std(y, ddof=0))
    if np.isclose(scale_y, 0.0):
        scale_y = 1.0
    y_standardized = (y - mean_y) / scale_y

    def harmonic_columns(phase_percent: np.ndarray) -> list[np.ndarray]:
        angle = 2.0 * np.pi * phase_percent / 100.0
        columns: list[np.ndarray] = []
        for order in range(1, harmonics + 1):
            columns.extend((np.cos(order * angle), np.sin(order * angle)))
        return columns

    design_columns = [np.ones_like(x), *harmonic_columns(x)]
    cycle_contrast_labels: list[int] = []
    if len(unique_cycles) > 1:
        reference_cycle = unique_cycles[-1]
        for cycle_index in unique_cycles[:-1]:
            contrast = np.where(cycles == cycle_index, 1.0, 0.0)
            contrast -= np.where(cycles == reference_cycle, 1.0, 0.0)
            design_columns.append(contrast)
            cycle_contrast_labels.append(cycle_index)
    design = np.column_stack(design_columns)
    penalty = np.zeros(design.shape[1], dtype=float)
    penalty[1:1 + 2 * harmonics] = harmonic_ridge
    penalty[1 + 2 * harmonics:] = cycle_contrast_ridge
    coefficients = np.linalg.solve(
        design.T @ design + np.diag(penalty),
        design.T @ y_standardized,
    )

    dense_x = np.arange(0.0, 101.0, 1.0)
    dense_design = np.column_stack([np.ones_like(dense_x), *harmonic_columns(dense_x)])
    dense_y = mean_y + scale_y * (dense_design @ coefficients[:1 + 2 * harmonics])
    fitted_y = mean_y + scale_y * (design @ coefficients)
    residuals = y - fitted_y
    cycle_offsets_ml = {}
    if cycle_contrast_labels:
        contrast_coefficients = coefficients[1 + 2 * harmonics:]
        for cycle_index, coefficient in zip(cycle_contrast_labels, contrast_coefficients):
            cycle_offsets_ml[str(cycle_index + 1)] = float(scale_y * coefficient)
        cycle_offsets_ml[str(unique_cycles[-1] + 1)] = float(
            -scale_y * np.sum(contrast_coefficients)
        )
    else:
        cycle_offsets_ml[str(unique_cycles[0] + 1)] = 0.0
    boundary_volume = float(dense_y[0])
    return {
        "observations": observations,
        "phase_percent": dense_x,
        "volume_ml": dense_y,
        "delta_from_0_ml": dense_y - boundary_volume,
        "boundary_volume_ml": boundary_volume,
        "minimum_ml": float(np.min(dense_y)),
        "maximum_ml": float(np.max(dense_y)),
        "phase_of_minimum_percent": float(dense_x[int(np.argmin(dense_y))]),
        "phase_of_maximum_percent": float(dense_x[int(np.argmax(dense_y))]),
        "model": "periodic_fourier_ridge_with_cycle_contrasts",
        "harmonics": harmonics,
        "harmonic_ridge": harmonic_ridge,
        "cycle_contrast_ridge": cycle_contrast_ridge,
        "cycle_offsets_ml": cycle_offsets_ml,
        "residual_rmse_ml": float(np.sqrt(np.mean(residuals ** 2))),
        "residual_max_abs_ml": float(np.max(np.abs(residuals))),
        "fitted_observation_ml": fitted_y,
    }


def ecg_strip_html(
    subject: str,
    path: Path | None,
    window_path: Path | None = None,
    ecg_evidence: dict | None = None,
) -> str:
    if path is None:
        return (
            f'<figure class="ecg-strip missing"><figcaption>{subject.upper()}: '
            'отчёт GE ECG Report в доступных КТ-данных не найден.</figcaption></figure>'
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    full_figure = (
        f'<figure class="ecg-strip"><img src="data:image/png;base64,{encoded}" '
        f'alt="Обезличенная полоса кардиосинхронизации GE, {subject.upper()}">'
        '<figcaption>Общий вид фактической полосы ЭКГ, использованной системой GE '
        'для кардиосинхронизации. Это растр без цифровых отсчётов и калибровки; '
        'он не предназначен для диагностической интерпретации ЭКГ.</figcaption></figure>'
    )
    if window_path is None:
        return full_figure
    window_encoded = base64.b64encode(window_path.read_bytes()).decode("ascii")
    phase_range = "выделенного интервала"
    if ecg_evidence is not None:
        start = float(ecg_evidence["start_phase_percent"])
        end = float(ecg_evidence["end_phase_percent"])
        phase_range = f"интервала {start:g}–{end:g}% R–R"
    zoom_figure = (
        f'<figure class="ecg-strip zoom"><img src="data:image/png;base64,{window_encoded}" '
        f'alt="Увеличенный фрагмент кардиосинхронизации GE, {subject.upper()}">'
        f'<figcaption>Увеличенный фрагмент {phase_range}, отмеченного цветным фоном '
        'в отчёте GE. Файл получен точным пиксельным кадрированием с полями 8 пикселей '
        'без реконструкции сигнала; увеличение выполняет браузер при отображении.</figcaption></figure>'
    )
    return full_figure + zoom_figure


def first_metadata_value(timing: dict, label: str) -> str:
    values = timing.get("all_present_cardiac_metadata", {}).get(label, [])
    return str(values[0]["value"]) if len(values) == 1 else ""


def gating_source_table_html(rows: list[dict]) -> str:
    def number(value, digits: int = 1) -> str:
        if value in ("", None):
            return "—"
        return f"{float(value):.{digits}f}".replace(".", ",")

    body = []
    for row in rows:
        prior = "—"
        if row["average_heart_rate_prior_to_confirm_bpm"] != "":
            prior = (
                f"{number(row['average_heart_rate_prior_to_confirm_bpm'], 1)} "
                f"({number(row['minimum_heart_rate_prior_to_confirm_bpm'], 0)}–"
                f"{number(row['maximum_heart_rate_prior_to_confirm_bpm'], 0)})"
            )
        body.append(
            "<tr>"
            f"<td>{row['subject'].upper()}</td><td>{row['cycle_index']}</td>"
            f"<td>{number(row['rpeak_start_s'], 3)}</td><td>{number(row['rpeak_end_s'], 3)}</td>"
            f"<td>{number(row['rr_interval_ms'], 1)}</td>"
            f"<td>{number(row['heart_rate_from_rpeaks_bpm'], 2)}</td>"
            f"<td>{number(row['median_image_heart_rate_bpm'], 0)}</td>"
            f"<td>{number(row['heart_rate_at_confirm_bpm'], 0)}</td><td>{prior}</td>"
            "</tr>"
        )
    return (
        '<div style="overflow-x:auto"><h3>Исходные данные кардиосинхронизации GE</h3>'
        '<p>Метки R-зубцов взяты из приватного тега <code>RpeakTimeStamps '
        '(0049,100C)</code>. Это временные метки триггеров реконструкции, а не '
        'цифровые отсчёты ЭКГ.</p>'
        '<table style="border-collapse:collapse;width:100%">'
        '<thead><tr><th>Испытуемый</th><th>Цикл</th><th>R начало, с</th>'
        '<th>R конец, с</th><th>R–R, мс</th><th>ЧСС по R–R, уд/мин</th>'
        '<th>ЧСС изображения, уд/мин</th><th>ЧСС при подтверждении, уд/мин</th>'
        '<th>До подтверждения: средняя (мин.–макс.), уд/мин</th></tr></thead><tbody>'
        + "".join(body)
        + '</tbody></table></div><style>th,td{border:1px solid #d1d5db;'
        'padding:6px 8px;text-align:left}th{background:#f3f4f6}</style>'
    )


def compact_values(values: list, digits: int = 3) -> str:
    result = []
    for value in values:
        if isinstance(value, (float, int)):
            rendered = f"{float(value):.{digits}f}"
            result.append(rendered if digits == 0 else rendered.rstrip("0").rstrip("."))
        else:
            result.append(str(value))
    return "; ".join(result)


def metadata_distribution(timing: dict, label: str) -> str:
    values = timing.get("all_present_cardiac_metadata", {}).get(label, [])
    return "; ".join(f"{item['value']} (n={item['count']})" for item in values)


def build_subject_dicom_row(timing: dict, ecg_evidence: dict | None) -> dict:
    rpeak = timing.get("manufacturer_rpeak_timing", {})
    mapping = timing.get("derived_multicycle_time_mapping", {})
    phase_rows = timing.get("phase_rows", [])
    raw_phases = [float(item["nominal_percentage"]) for item in phase_rows]
    normalized_phases = [float(item["phase_percent_within_cycle"]) for item in phase_rows]
    evidence_hr = ""
    evidence_range = ""
    if ecg_evidence:
        evidence_range = f"{ecg_evidence['start_phase_percent']}–{ecg_evidence['end_phase_percent']}"
        evidence_hr = "; ".join(
            f"{item['label']}: {item['minimum_bpm']}/{item['average_bpm']}/{item['maximum_bpm']}"
            for item in ecg_evidence["heart_rate_statistics"]
        )
    return {
        "subject": timing["subject"],
        "source_series_folder_name": timing["source_folder_name"],
        "dicom_files_read": timing["dicom_files_read"],
        "nominal_phase_values_percent": compact_values(raw_phases, 0),
        "phase_values_within_cycle_percent": compact_values(normalized_phases, 0),
        "rpeak_timestamps_s": compact_values(rpeak.get("selected_rpeak_timestamps_s", []), 3),
        "rr_intervals_ms": compact_values(rpeak.get("rr_intervals_ms", []), 3),
        "heart_rate_from_rpeaks_bpm": compact_values(rpeak.get("heart_rates_from_rpeak_intervals_bpm", []), 3),
        "avg_heart_rate_for_image_bpm_and_image_count": metadata_distribution(timing, "AvgHeartRateForImage"),
        "heart_rate_at_confirm_bpm": first_metadata_value(timing, "HeartRateAtConfirm"),
        "prior_hr_mean_bpm": first_metadata_value(timing, "AvgHeartRatePriorToConfirm"),
        "prior_hr_min_bpm": first_metadata_value(timing, "MinHeartRatePriorToConfirm"),
        "prior_hr_max_bpm": first_metadata_value(timing, "MaxHeartRatePriorToConfirm"),
        "prior_hr_std_bpm": first_metadata_value(timing, "StdDevHeartRatePriorToConfirm"),
        "prior_hr_sample_count": first_metadata_value(timing, "NumHeartRateSamplesPriorToConfirm"),
        "ge_temporal_resolution_s": first_metadata_value(timing, "GETemporalResolutionSeconds"),
        "center_view_angle_max_residual_ms": mapping.get("center_view_angle_cross_check", {}).get("maximum_absolute_residual_ms", ""),
        "ecg_monitor_type": first_metadata_value(timing, "EkgMonitorType"),
        "ecg_gating_type_private_code": first_metadata_value(timing, "EkgGatingType"),
        "das_trigger_source_private_code": first_metadata_value(timing, "DASTriggerSource"),
        "cardiac_recon_algorithm_private_code": first_metadata_value(timing, "CardiacReconAlgorithm"),
        "auto_hr_detect_predict_private_code": first_metadata_value(timing, "AutoHeartRateDetectPredict"),
        "system_optimized_hr_private_code": first_metadata_value(timing, "SystemOptimizedHeartRate"),
        "reconstruction_sectors_private_code": first_metadata_value(timing, "NumReconSectors"),
        "number_of_triggers_private_value": first_metadata_value(timing, "NumberOfTriggers"),
        "trigger_frequency_private_value": first_metadata_value(timing, "TriggerFrequency"),
        "trigger_on_position_private_value": first_metadata_value(timing, "TriggerOnPosition"),
        "ecg_report_available": bool(ecg_evidence),
        "ecg_report_phase_range_percent": evidence_range,
        "ecg_report_hr_min_mean_max_bpm": evidence_hr,
        "digital_waveform_sequence_available": bool(ecg_evidence and ecg_evidence.get("digital_waveform_sequence_available")),
        "interpretation_status": (
            "short_long_trigger_pair_mechanism_unresolved_no_waveform"
            if timing["subject"] == "georg"
            else "adjacent_trigger_intervals_consistent_with_image_hr_no_waveform"
        ),
    }


def subject_dicom_table_html(rows: list[dict]) -> str:
    def number(value, digits: int = 1) -> str:
        if value in ("", None):
            return "—"
        return f"{float(value):.{digits}f}".replace(".", ",")

    body = []
    for row in rows:
        report = "нет"
        if row["ecg_report_available"]:
            report = (
                f"есть; {row['ecg_report_phase_range_percent']}% R–R; "
                f"{row['ecg_report_hr_min_mean_max_bpm']}"
            )
        private_codes = (
            f"gating={row['ecg_gating_type_private_code']}; "
            f"DAS={row['das_trigger_source_private_code']}; "
            f"recon={row['cardiac_recon_algorithm_private_code']}; "
            f"autoHR={row['auto_hr_detect_predict_private_code']}; "
            f"optimizedHR={row['system_optimized_hr_private_code']}; "
            f"sectors={row['reconstruction_sectors_private_code']}; "
            f"triggers={row['number_of_triggers_private_value']}; "
            f"frequency={row['trigger_frequency_private_value']}; "
            f"position={row['trigger_on_position_private_value']}"
        )
        body.append(
            "<tr>"
            f"<td>{row['subject'].upper()}</td><td>{row['source_series_folder_name']}<br>n={row['dicom_files_read']}</td>"
            f"<td>{row['nominal_phase_values_percent']}</td>"
            f"<td>{row['rpeak_timestamps_s']}</td><td>{row['rr_intervals_ms']}</td>"
            f"<td>{row['heart_rate_from_rpeaks_bpm']}</td>"
            f"<td>{row['avg_heart_rate_for_image_bpm_and_image_count']}</td>"
            f"<td>{row['heart_rate_at_confirm_bpm']}; "
            f"{number(row['prior_hr_mean_bpm'], 1)} "
            f"({number(row['prior_hr_min_bpm'], 0)}–{number(row['prior_hr_max_bpm'], 0)}), "
            f"SD={number(row['prior_hr_std_bpm'], 1)}, n={row['prior_hr_sample_count']}</td>"
            f"<td>{number(row['ge_temporal_resolution_s'], 3)} с; "
            f"остаток CVA ≤ {number(row['center_view_angle_max_residual_ms'], 2)} мс</td>"
            f"<td>{row['ecg_monitor_type']}; {private_codes}</td><td>{report}</td>"
            "</tr>"
        )
    return (
        '<div style="overflow-x:auto"><h3>Субъектная сводка полей DICOM кардиосинхронизации</h3>'
        '<p>Числовые приватные коды GE приведены дословно и не расшифрованы без документации производителя. '
        '<code>RpeakTimeStamps</code> — триггеры реконструкции, а не цифровой сигнал ЭКГ.</p>'
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        '<thead><tr><th>Испытуемый</th><th>Серия / DICOM</th><th>Фазы, %</th>'
        '<th>R-метки, с</th><th>R–R, мс</th><th>ЧСС по R–R</th>'
        '<th>AvgHeartRateForImage (n)</th><th>ЧСС confirm; до confirm</th>'
        '<th>Временное разрешение / проверка CVA</th><th>Монитор / приватные коды</th>'
        '<th>GE ECG Report</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
        '<style>th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:left;vertical-align:top}'
        'th{background:#f3f4f6}</style>'
    )


def subtraction_figure(subject: str, phases: list[dict]) -> go.Figure:
    all_meshes = [phase["surface_meshes"] for phase in phases]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for meshes in all_meshes:
        for mesh in meshes:
            xs.extend(mesh["x"]); ys.extend(mesh["y"]); zs.extend(mesh["z"])
    figure = go.Figure(
        data=[trace(mesh, style, True) for mesh, style in zip(all_meshes[0], SURFACES)],
        frames=[go.Frame(
            name=phase["phase_id"], traces=list(range(len(SURFACES))),
            data=[trace(mesh, style, False) for mesh, style in zip(meshes, SURFACES)],
        ) for phase, meshes in zip(phases, all_meshes)],
    )
    steps = [{
        "label": f"{phase['rr_percent']:g}%",
        "method": "animate",
        "args": [[phase["phase_id"]], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for phase in phases]
    figure.update_layout(
        title={"text": f"{subject.upper()}: геометрическое вычитание маски крови из маски сердца", "x": 0.5},
        template="plotly_white", height=760, margin={"l": 0, "r": 0, "t": 80, "b": 25},
        legend={"orientation": "h", "y": 1.01, "x": 0.5, "xanchor": "center", "yanchor": "bottom", "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        scene={
            "xaxis": {"title": "R–L, мм", "range": axis_range(xs)},
            "yaxis": {"title": "A–P, мм", "range": axis_range(ys)},
            "zaxis": {"title": "I–S, мм", "range": axis_range(zs)},
            "aspectmode": "data", "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.05}},
        },
        sliders=[{"active": 0, "currentvalue": {"prefix": "Исходное значение DICOM: "}, "pad": {"t": 55}, "steps": steps}],
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0, "y": 0, "pad": {"t": 70}, "buttons": [
            {"label": "▶", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 550, "redraw": True}, "transition": {"duration": 0}}]},
            {"label": "Ⅱ", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
        ]}],
        uirevision=f"subtraction-{subject}",
    )
    return figure


def volume_figure(
    subject: str,
    phases: list[dict],
    ecg_evidence: dict | None,
    profiles: dict[str, dict] | None,
    merge_assessment: dict,
) -> go.Figure:
    ecg_x = np.linspace(0.0, 100.0, 1201)
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.18, 0.18, 0.64], vertical_spacing=0.045,
    )
    figure.add_trace(go.Scatter(
        x=ecg_x, y=schematic_ecg(ecg_x), mode="lines", name="Стандартная схема P–QRS–T",
        line={"color": "#4B5563", "width": 2},
        hovertemplate="Схема P–QRS–T<br>Нормированная фаза: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    cycle_colors = ("#047857", "#7C3AED", "#B45309")
    cycle_symbols = ("circle", "diamond", "square")
    for cycle_index in sorted({int(phase["cycle_index"]) for phase in phases}):
        cycle_phases = sorted(
            (phase for phase in phases if int(phase["cycle_index"]) == cycle_index),
            key=lambda phase: phase["phase_percent_within_cycle"],
        )
        figure.add_trace(go.Scatter(
            x=[phase["phase_percent_within_cycle"] for phase in cycle_phases],
            y=[phase["image_heart_rate_bpm"] for phase in cycle_phases],
            customdata=[[
                phase["rr_percent"], phase["derived_rr_interval_ms"], phase["derived_time_from_R0_ms"],
            ] for phase in cycle_phases],
            mode="lines+markers", name=f"ЧСС GE, цикл {cycle_index + 1}",
            line={"color": cycle_colors[cycle_index % len(cycle_colors)], "width": 2, "dash": "dot"},
            marker={"size": 8, "symbol": cycle_symbols[cycle_index % len(cycle_symbols)]},
            hovertemplate="ЧСС изображения: %{y:.0f} уд/мин<br>Цикл: " + str(cycle_index + 1) +
            "<br>Исходная шкала: %{customdata[0]:.0f}%<br>Расчётный R–R: %{customdata[1]:.1f} мс<br>Время от R₀: %{customdata[2]:.1f} мс<extra></extra>",
        ), row=2, col=1)

    text_positions = (
        ("top left", "bottom left"),
        ("top right", "bottom right"),
        ("bottom left", "top left"),
        ("bottom right", "top right"),
    )
    if profiles is not None:
        for chamber_index, (key, label, color) in enumerate(CHAMBERS):
            profile = profiles[key]
            dense_customdata = [
                [float(delta), phase_band_label(float(phase_percent))]
                for phase_percent, delta in zip(profile["phase_percent"], profile["delta_from_0_ml"])
            ]
            figure.add_trace(go.Scatter(
                x=profile["phase_percent"], y=profile["volume_ml"], customdata=dense_customdata,
                mode="lines", name=label, legendgroup=key,
                line={"color": color, "width": 3},
                hovertemplate="%{fullData.name}<br>Фаза: %{x:.0f}%<br>%{customdata[1]}<br>Расчётный объём: %{y:.1f} мл<br>Δ от 0%: %{customdata[0]:+.1f} мл<extra></extra>",
            ), row=3, col=1)
            observations = profile["observations"]
            figure.add_trace(go.Scatter(
                x=[item["phase_percent"] for item in observations],
                y=[item["volume_ml"] for item in observations],
                text=[f"{item['volume_ml']:.1f}".replace(".", ",") for item in observations],
                textposition=[text_positions[chamber_index][item["cycle_index"] % 2] for item in observations],
                customdata=[[item["phase_id"], item["cycle_index"] + 1, item["raw_rr_percent"]] for item in observations],
                mode="markers+text", name=f"{label}: измерено", legendgroup=key, showlegend=False,
                marker={
                    "color": color, "size": 9,
                    "symbol": [cycle_symbols[item["cycle_index"] % len(cycle_symbols)] for item in observations],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                textfont={"color": color, "size": 10}, cliponaxis=False,
                hovertemplate="%{fullData.name}<br>Фаза внутри цикла: %{x:.0f}%<br>Цикл: %{customdata[1]}<br>Исходная шкала: %{customdata[2]:.0f}%<br>Измеренный объём: %{y:.1f} мл<br>%{customdata[0]}<extra></extra>",
            ), row=3, col=1)
            boundary_volume = profile["boundary_volume_ml"]
            figure.add_trace(go.Scatter(
                x=[0.0, 100.0], y=[boundary_volume, boundary_volume],
                text=[f"{boundary_volume:.1f}*".replace(".", ","), f"{boundary_volume:.1f}*".replace(".", ",")],
                textposition=[text_positions[chamber_index][0], text_positions[chamber_index][1]],
                mode="markers+text", name=f"{label}: граница 0/100%", legendgroup=key, showlegend=False,
                marker={"color": color, "size": 10, "symbol": "diamond-open", "line": {"color": color, "width": 2}},
                textfont={"color": color, "size": 10}, cliponaxis=False,
                hovertemplate="%{fullData.name}<br>Периодическая граница: %{x:.0f}%<br>Объём: %{y:.1f} мл<extra></extra>",
            ), row=3, col=1)
    else:
        cycle_indices = sorted({int(phase["cycle_index"]) for phase in phases})
        for chamber_index, (key, label, color) in enumerate(CHAMBERS):
            for cycle_order, cycle_index in enumerate(cycle_indices):
                profile = within_cycle_profile(phases, key, cycle_index)
                figure.add_trace(go.Scatter(
                    x=profile["phase_percent"], y=profile["volume_ml"],
                    customdata=[[cycle_index + 1, phase_band_label(float(value))] for value in profile["phase_percent"]],
                    mode="lines", name=label, legendgroup=key, showlegend=cycle_order == 0,
                    line={"color": color, "width": 3, "dash": "solid" if cycle_order == 0 else "dash"},
                    hovertemplate="%{fullData.name}<br>Цикл: %{customdata[0]}<br>Фаза: %{x:.0f}%<br>%{customdata[1]}<br>Расчётный объём: %{y:.1f} мл<extra></extra>",
                ), row=3, col=1)
                observations = profile["observations"]
                figure.add_trace(go.Scatter(
                    x=[item["phase_percent"] for item in observations],
                    y=[item["volume_ml"] for item in observations],
                    text=[f"{item['volume_ml']:.1f}".replace(".", ",") for item in observations],
                    textposition=[text_positions[chamber_index][cycle_order % 2]] * len(observations),
                    customdata=[[item["phase_id"], cycle_index + 1, item["raw_rr_percent"]] for item in observations],
                    mode="markers+text", name=f"{label}: измерено, цикл {cycle_index + 1}",
                    legendgroup=key, showlegend=False,
                    marker={
                        "color": color, "size": 9,
                        "symbol": cycle_symbols[cycle_order % len(cycle_symbols)],
                        "line": {"color": "#FFFFFF", "width": 1},
                    },
                    textfont={"color": color, "size": 10}, cliponaxis=False,
                    hovertemplate="%{fullData.name}<br>Фаза внутри цикла: %{x:.0f}%<br>Цикл: %{customdata[1]}<br>Исходная шкала: %{customdata[2]:.0f}%<br>Измеренный объём: %{y:.1f} мл<br>%{customdata[0]}<extra></extra>",
                ), row=3, col=1)

    for start, end, label, color in CARDIAC_PHASE_BANDS:
        figure.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0, layer="below", row=3, col=1)
        figure.add_annotation(
            x=(start + end) / 2.0, y=0.98, xref="x3", yref="y3 domain",
            text=label, showarrow=False, font={"size": 10, "color": "#4B5563"},
        )
    for boundary in (0.0, 10.0, 45.0, 55.0, 80.0, 100.0):
        figure.add_vline(x=boundary, line_dash="dash", line_color="#9CA3AF", line_width=1)
    for boundary in (0.0, 100.0):
        figure.add_annotation(x=boundary, y=1.08, text="R", showarrow=False, row=1, col=1, font={"color": "#B91C1C"})
    figure.add_annotation(x=85, y=0.23, text="P", showarrow=False, row=1, col=1)
    figure.add_annotation(x=30, y=0.41, text="T", showarrow=False, row=1, col=1)

    figure.update_xaxes(title_text="Нормированная фаза сердечного цикла, % R–R", dtick=10, range=[0, 100], row=3, col=1)
    figure.update_yaxes(title_text="Схема ЭКГ", showticklabels=False, zeroline=True, row=1, col=1)
    figure.update_yaxes(title_text="ЧСС, уд/мин", rangemode="tozero", row=2, col=1)
    figure.update_yaxes(title_text="Объём автоматической маски, мл", row=3, col=1)
    if ecg_evidence:
        scan = next(item for item in ecg_evidence["heart_rate_statistics"] if item["label"].lower().startswith("scan"))
        timing_note = (
            f"Отчёт GE ECG Report, серия {ecg_evidence['series_number_referenced_by_report']}: "
            f"{ecg_evidence['start_phase_percent']}–{ecg_evidence['end_phase_percent']}%; "
            f"ЧСС {scan['minimum_bpm']}/{scan['average_bpm']}/{scan['maximum_bpm']} уд/мин (мин./ср./макс.)."
        )
    else:
        timing_note = "Отчёт GE ECG Report не найден; временная привязка основана на DICOM и приватных GE-тегах реконструкции."
    if profiles is not None:
        figure_title = f"{subject.upper()}: замкнутый композитный цикл объёмов камер"
        closure_note = (
            "Сплошные линии — периодическая двухгармоническая модель с L2-регуляризацией и поправкой "
            "на цикл; 0=100% следует из базиса. Фоновые интервалы — ориентиры."
        )
    else:
        rates = list(merge_assessment["heart_rate_by_cycle_bpm"].values())
        rates_text = " и ".join(f"{value:g}" for value in rates)
        relative_difference = 100.0 * merge_assessment["maximum_relative_heart_rate_difference"]
        relative_difference_text = f"{relative_difference:.1f}".replace(".", ",")
        figure_title = f"{subject.upper()}: раздельные частичные циклы объёмов камер"
        closure_note = (
            f"Циклы не объединены: значения ЧСС {rates_text} уд/мин различаются на "
            f"{relative_difference_text}%. Интерполяция PCHIP построена отдельно в наблюдаемых диапазонах."
        )
    figure.update_layout(
        title={"text": figure_title, "x": 0.5, "y": 0.985},
        template="plotly_white", height=1000, hovermode="x unified",
        legend={
            "orientation": "h", "y": 1.04, "x": 0.5, "xanchor": "center", "yanchor": "bottom",
            "itemclick": "toggle", "itemdoubleclick": "toggleothers", "groupclick": "togglegroup",
        },
        margin={"l": 95, "r": 45, "t": 235, "b": 75},
    )
    figure.add_annotation(x=0.5, y=1.22, xref="paper", yref="paper", text=timing_note, showarrow=False, font={"size": 12, "color": "#374151"})
    figure.add_annotation(x=0.5, y=1.15, xref="paper", yref="paper", text=closure_note, showarrow=False, font={"size": 11, "color": "#6B7280"})
    return figure


def volume_figure_with_scale_toggle(
    subject: str,
    phases: list[dict],
    ecg_evidence: dict | None,
    profiles: dict[str, dict] | None,
    merge_assessment: dict,
    timing: dict,
) -> go.Figure:
    """Show raw DICOM coordinates by default and a switchable normalized view."""
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.18, 0.18, 0.64], vertical_spacing=0.045,
    )
    raw_indices: list[int] = []
    normalized_indices: list[int] = []
    cycle_indices = sorted({int(phase["cycle_index"]) for phase in phases})
    cycle_colors = ("#047857", "#7C3AED", "#B45309")
    cycle_symbols = ("circle", "diamond", "square")
    text_positions = (
        ("top left", "bottom left"),
        ("top right", "bottom right"),
        ("bottom left", "top left"),
        ("bottom right", "top right"),
    )

    def phase_orientation_label(value: float) -> str:
        if profiles is None:
            return "Без индивидуальной фазовой метки"
        return phase_band_label(value)

    def add(trace: go.Scatter, row: int, mode: str) -> None:
        trace.visible = mode == "raw"
        figure.add_trace(trace, row=row, col=1)
        (raw_indices if mode == "raw" else normalized_indices).append(len(figure.data) - 1)

    raw_max_observed = max(float(phase["rr_percent"]) for phase in phases)
    raw_axis_max = max(100.0, 10.0 * np.ceil(raw_max_observed / 10.0))
    raw_ecg_x = np.linspace(0.0, raw_axis_max, int(raw_axis_max * 12) + 1)
    normalized_ecg_x = np.linspace(0.0, 100.0, 1201)
    add(go.Scatter(
        x=raw_ecg_x, y=schematic_ecg(np.mod(raw_ecg_x, 100.0)),
        mode="lines", name="Стандартная схема P–QRS–T", legendgroup="schematic_ecg",
        line={"color": "#4B5563", "width": 2},
        hovertemplate="Схема P–QRS–T<br>Исходная координата DICOM: %{x:.1f}%<extra></extra>",
    ), 1, "raw")
    add(go.Scatter(
        x=normalized_ecg_x, y=schematic_ecg(normalized_ecg_x),
        mode="lines", name="Стандартная схема P–QRS–T", legendgroup="schematic_ecg",
        line={"color": "#4B5563", "width": 2},
        hovertemplate="Схема P–QRS–T<br>Нормированная фаза: %{x:.1f}%<extra></extra>",
    ), 1, "normalized")

    for mode in ("raw", "normalized"):
        for cycle_index in cycle_indices:
            cycle_phases = sorted(
                (phase for phase in phases if int(phase["cycle_index"]) == cycle_index),
                key=lambda phase: phase["phase_percent_within_cycle"],
            )
            add(go.Scatter(
                x=[
                    phase["rr_percent"] if mode == "raw" else phase["phase_percent_within_cycle"]
                    for phase in cycle_phases
                ],
                y=[phase["image_heart_rate_bpm"] for phase in cycle_phases],
                customdata=[[
                    phase["rr_percent"], phase["rr_interval_ms"],
                    phase["heart_rate_from_rpeaks_bpm"], phase["derived_time_from_R0_ms"],
                ] for phase in cycle_phases],
                mode="lines+markers", name=f"ЧСС GE, цикл {cycle_index + 1}",
                legendgroup=f"hr_cycle_{cycle_index}",
                line={"color": cycle_colors[cycle_index % len(cycle_colors)], "width": 2, "dash": "dot"},
                marker={"size": 8, "symbol": cycle_symbols[cycle_index % len(cycle_symbols)]},
                hovertemplate=(
                    "ЧСС изображения: %{y:.0f} уд/мин<br>Цикл: " + str(cycle_index + 1) +
                    "<br>Исходная шкала: %{customdata[0]:.0f}%"
                    "<br>R–R по RpeakTimeStamps: %{customdata[1]:.1f} мс"
                    "<br>ЧСС по R–R: %{customdata[2]:.2f} уд/мин"
                    "<br>Время от R₀: %{customdata[3]:.1f} мс<extra></extra>"
                ),
            ), 2, mode)

    for chamber_index, (key, label, color) in enumerate(CHAMBERS):
        for cycle_order, cycle_index in enumerate(cycle_indices):
            profile = within_cycle_profile(phases, key, cycle_index)
            raw_dense_x = profile["phase_percent"] + 100.0 * cycle_index
            add(go.Scatter(
                x=raw_dense_x, y=profile["volume_ml"],
                customdata=[[cycle_index + 1, phase_orientation_label(float(value % 100.0))] for value in raw_dense_x],
                mode="lines", name=label, legendgroup=key, showlegend=cycle_order == 0,
                line={"color": color, "width": 3, "dash": "solid" if cycle_order == 0 else "dash"},
                hovertemplate=(
                    "%{fullData.name}<br>Цикл: %{customdata[0]}<br>Исходная шкала DICOM: %{x:.0f}%"
                    "<br>%{customdata[1]}<br>Расчётный объём: %{y:.1f} мл<extra></extra>"
                ),
            ), 3, "raw")
            observations = profile["observations"]
            add(go.Scatter(
                x=[item["raw_rr_percent"] for item in observations],
                y=[item["volume_ml"] for item in observations],
                text=[f"{item['volume_ml']:.1f}".replace(".", ",") for item in observations],
                textposition=[text_positions[chamber_index][cycle_order % 2]] * len(observations),
                customdata=[[item["phase_id"], cycle_index + 1, item["phase_percent"]] for item in observations],
                mode="markers+text", name=f"{label}: измерено, цикл {cycle_index + 1}",
                legendgroup=key, showlegend=False,
                marker={
                    "color": color, "size": 9,
                    "symbol": cycle_symbols[cycle_order % len(cycle_symbols)],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                textfont={"color": color, "size": 10}, cliponaxis=False,
                hovertemplate=(
                    "%{fullData.name}<br>Исходная шкала DICOM: %{x:.0f}%"
                    "<br>Фаза внутри цикла: %{customdata[2]:.0f}%<br>Цикл: %{customdata[1]}"
                    "<br>Измеренный объём: %{y:.1f} мл<br>%{customdata[0]}<extra></extra>"
                ),
            ), 3, "raw")

    if profiles is not None:
        for chamber_index, (key, label, color) in enumerate(CHAMBERS):
            profile = profiles[key]
            dense_customdata = [
                [float(delta), phase_orientation_label(float(phase_percent))]
                for phase_percent, delta in zip(profile["phase_percent"], profile["delta_from_0_ml"])
            ]
            add(go.Scatter(
                x=profile["phase_percent"], y=profile["volume_ml"], customdata=dense_customdata,
                mode="lines", name=label, legendgroup=key,
                line={"color": color, "width": 3},
                hovertemplate=(
                    "%{fullData.name}<br>Фаза: %{x:.0f}%<br>%{customdata[1]}"
                    "<br>Расчётный объём: %{y:.1f} мл<br>Δ от 0%: %{customdata[0]:+.1f} мл<extra></extra>"
                ),
            ), 3, "normalized")
            observations = profile["observations"]
            add(go.Scatter(
                x=[item["phase_percent"] for item in observations],
                y=[item["volume_ml"] for item in observations],
                text=[f"{item['volume_ml']:.1f}".replace(".", ",") for item in observations],
                textposition=[text_positions[chamber_index][item["cycle_index"] % 2] for item in observations],
                customdata=[[item["phase_id"], item["cycle_index"] + 1, item["raw_rr_percent"]] for item in observations],
                mode="markers+text", name=f"{label}: измерено", legendgroup=key, showlegend=False,
                marker={
                    "color": color, "size": 9,
                    "symbol": [cycle_symbols[item["cycle_index"] % len(cycle_symbols)] for item in observations],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                textfont={"color": color, "size": 10}, cliponaxis=False,
                hovertemplate=(
                    "%{fullData.name}<br>Фаза внутри цикла: %{x:.0f}%<br>Цикл: %{customdata[1]}"
                    "<br>Исходная шкала: %{customdata[2]:.0f}%<br>Измеренный объём: %{y:.1f} мл"
                    "<br>%{customdata[0]}<extra></extra>"
                ),
            ), 3, "normalized")
            boundary_volume = profile["boundary_volume_ml"]
            add(go.Scatter(
                x=[0.0, 100.0], y=[boundary_volume, boundary_volume],
                text=[f"{boundary_volume:.1f}*".replace(".", ","), f"{boundary_volume:.1f}*".replace(".", ",")],
                textposition=[text_positions[chamber_index][0], text_positions[chamber_index][1]],
                mode="markers+text", name=f"{label}: граница 0/100%", legendgroup=key, showlegend=False,
                marker={"color": color, "size": 10, "symbol": "diamond-open", "line": {"color": color, "width": 2}},
                textfont={"color": color, "size": 10}, cliponaxis=False,
                hovertemplate="%{fullData.name}<br>Периодическая граница: %{x:.0f}%<br>Объём: %{y:.1f} мл<extra></extra>",
            ), 3, "normalized")
    else:
        for chamber_index, (key, label, color) in enumerate(CHAMBERS):
            for cycle_order, cycle_index in enumerate(cycle_indices):
                profile = within_cycle_profile(phases, key, cycle_index)
                add(go.Scatter(
                    x=profile["phase_percent"], y=profile["volume_ml"],
                    customdata=[[cycle_index + 1, phase_orientation_label(float(value))] for value in profile["phase_percent"]],
                    mode="lines", name=label, legendgroup=key, showlegend=cycle_order == 0,
                    line={"color": color, "width": 3, "dash": "solid" if cycle_order == 0 else "dash"},
                    hovertemplate=(
                        "%{fullData.name}<br>Цикл: %{customdata[0]}<br>Фаза: %{x:.0f}%"
                        "<br>%{customdata[1]}<br>Расчётный объём: %{y:.1f} мл<extra></extra>"
                    ),
                ), 3, "normalized")
                observations = profile["observations"]
                add(go.Scatter(
                    x=[item["phase_percent"] for item in observations],
                    y=[item["volume_ml"] for item in observations],
                    text=[f"{item['volume_ml']:.1f}".replace(".", ",") for item in observations],
                    textposition=[text_positions[chamber_index][cycle_order % 2]] * len(observations),
                    customdata=[[item["phase_id"], cycle_index + 1, item["raw_rr_percent"]] for item in observations],
                    mode="markers+text", name=f"{label}: измерено, цикл {cycle_index + 1}",
                    legendgroup=key, showlegend=False,
                    marker={
                        "color": color, "size": 9, "symbol": cycle_symbols[cycle_order % len(cycle_symbols)],
                        "line": {"color": "#FFFFFF", "width": 1},
                    },
                    textfont={"color": color, "size": 10}, cliponaxis=False,
                    hovertemplate=(
                        "%{fullData.name}<br>Фаза внутри цикла: %{x:.0f}%<br>Цикл: %{customdata[1]}"
                        "<br>Исходная шкала: %{customdata[2]:.0f}%<br>Измеренный объём: %{y:.1f} мл"
                        "<br>%{customdata[0]}<extra></extra>"
                    ),
                ), 3, "normalized")

    def phase_layout(axis_max: float, normalized: bool) -> tuple[list[dict], list[dict]]:
        shapes: list[dict] = []
        annotations: list[dict] = []
        cycle_bases = [0.0] if normalized else [100.0 * index for index in range(int(np.ceil(axis_max / 100.0)))]
        for base in cycle_bases:
            phase_bands = CARDIAC_PHASE_BANDS if profiles is not None else ()
            for start, end, label, color in phase_bands:
                left, right = base + start, min(base + end, axis_max)
                if left >= axis_max:
                    continue
                shapes.append({
                    "type": "rect", "xref": "x3", "yref": "y3 domain",
                    "x0": left, "x1": right, "y0": 0, "y1": 1,
                    "fillcolor": color, "line": {"width": 0}, "layer": "below",
                })
                annotations.append({
                    "x": (left + right) / 2.0, "y": 0.98, "xref": "x3", "yref": "y3 domain",
                    "text": label, "showarrow": False, "font": {"size": 10, "color": "#4B5563"},
                })
            phase_boundaries = (0.0, 45.0, 80.0, 100.0) if profiles is not None else (0.0, 100.0)
            for boundary in phase_boundaries:
                value = base + boundary
                if value <= axis_max:
                    shapes.append({
                        "type": "line", "xref": "x3", "yref": "y3 domain",
                        "x0": value, "x1": value, "y0": 0, "y1": 1,
                        "line": {"dash": "dash", "color": "#9CA3AF", "width": 1},
                    })
            for value in (base, base + 100.0):
                if value <= axis_max:
                    annotations.append({
                        "x": value, "y": 1.08, "xref": "x", "yref": "y domain",
                        "text": "R", "showarrow": False, "font": {"color": "#B91C1C"},
                    })
            if base + 85.0 <= axis_max:
                annotations.append({"x": base + 85.0, "y": 0.23, "xref": "x", "yref": "y", "text": "P", "showarrow": False})
            if base + 30.0 <= axis_max:
                annotations.append({"x": base + 30.0, "y": 0.41, "xref": "x", "yref": "y", "text": "T", "showarrow": False})
        return shapes, annotations

    rpeak = timing.get("manufacturer_rpeak_timing", {})
    rr_values = rpeak.get("rr_intervals_ms", [])
    hr_values = rpeak.get("heart_rates_from_rpeak_intervals_bpm", [])
    if rr_values:
        rr_text = "; ".join(
            f"R–R{index + 1} = {rr:.0f} мс ({hr:.1f} уд/мин)".replace(".", ",")
            for index, (rr, hr) in enumerate(zip(rr_values, hr_values))
        )
        timing_note = f"GE-метки RpeakTimeStamps (0049,100C): {rr_text}."
    else:
        timing_note = "RpeakTimeStamps в DICOM не найден; R–R рассчитан как 60000/ЧСС изображения."
    if ecg_evidence:
        scan = next(item for item in ecg_evidence["heart_rate_statistics"] if item["label"].lower().startswith("scan"))
        timing_note += (
            f"<br>GE ECG Report: {ecg_evidence['start_phase_percent']}–{ecg_evidence['end_phase_percent']}%; "
            f"ЧСС {scan['minimum_bpm']}/{scan['average_bpm']}/{scan['maximum_bpm']} уд/мин (мин./ср./макс.)."
        )
    elif subject == "georg":
        timing_note += (
            "<br>Это коротко-длинная пара триггеров в одной серии; без цифровой ЭКГ нельзя отличить "
            "истинное изменение ритма от ошибки детектирования R-зубца."
        )

    raw_shapes, raw_annotations = phase_layout(raw_axis_max, normalized=False)
    normalized_shapes, normalized_annotations = phase_layout(100.0, normalized=True)
    raw_note = (
        "Показаны исходные NominalPercentageOfCardiacPhase без переноса значений >100%. "
        "PCHIP построен отдельно внутри каждого записанного R–R-интервала."
    )
    if profiles is not None:
        normalized_title = f"{subject.upper()}: предварительный нормированный композитный цикл"
        normalized_note = (
            "Сплошные линии — периодическая двухгармоническая модель с L2-регуляризацией "
            "с регуляризованной поправкой на цикл; 0=100% задано базисом. "
            "Фон показывает объёмные ориентиры, "
            "а не индивидуальные клапанные фазы."
        )
    else:
        rates = list(merge_assessment["heart_rate_by_cycle_bpm"].values())
        rates_text = " и ".join(f"{value:g}" for value in rates)
        normalized_title = f"{subject.upper()}: нормированные, но не объединённые циклы"
        normalized_note = (
            f"Циклы с ЧСС {rates_text} уд/мин показаны на общей шкале 0–100%. "
            "Кривые не объединены, экстраполяция не выполнена, фазовый фон отключён."
        )

    def annotations_with_notes(base: list[dict], note: str) -> list[dict]:
        return base + [
            {"x": 0.5, "y": 1.25, "xref": "paper", "yref": "paper", "text": timing_note, "showarrow": False, "font": {"size": 12, "color": "#374151"}, "align": "center"},
            {"x": 0.5, "y": 1.17, "xref": "paper", "yref": "paper", "text": note, "showarrow": False, "font": {"size": 11, "color": "#6B7280"}, "align": "center"},
        ]

    raw_visibility = [index in raw_indices for index in range(len(figure.data))]
    normalized_visibility = [index in normalized_indices for index in range(len(figure.data))]
    raw_layout = {
        "title.text": f"{subject.upper()}: исходные DICOM-фазы и объёмы камер",
        "xaxis.range": [0, raw_axis_max], "xaxis2.range": [0, raw_axis_max], "xaxis3.range": [0, raw_axis_max],
        "xaxis3.title.text": "Исходная координата GE NominalPercentageOfCardiacPhase, %",
        "shapes": raw_shapes, "annotations": annotations_with_notes(raw_annotations, raw_note),
    }
    normalized_layout = {
        "title.text": normalized_title,
        "xaxis.range": [0, 100], "xaxis2.range": [0, 100], "xaxis3.range": [0, 100],
        "xaxis3.title.text": "Нормированная фаза сердечного цикла, % R–R",
        "shapes": normalized_shapes, "annotations": annotations_with_notes(normalized_annotations, normalized_note),
    }
    figure.update_xaxes(dtick=10, range=[0, raw_axis_max])
    figure.update_xaxes(title_text=raw_layout["xaxis3.title.text"], row=3, col=1)
    figure.update_yaxes(title_text="Схема ЭКГ", showticklabels=False, zeroline=True, row=1, col=1)
    figure.update_yaxes(title_text="ЧСС, уд/мин", rangemode="tozero", row=2, col=1)
    figure.update_yaxes(title_text="Объём автоматической маски, мл", row=3, col=1)
    figure.update_layout(
        title={"text": raw_layout["title.text"], "x": 0.5, "y": 0.985},
        template="plotly_white", height=1040, hovermode="x unified", separators=", ",
        legend={
            "orientation": "h", "y": 1.04, "x": 0.5, "xanchor": "center", "yanchor": "bottom",
            "itemclick": "toggle", "itemdoubleclick": "toggleothers", "groupclick": "togglegroup",
        },
        margin={"l": 95, "r": 45, "t": 285, "b": 75},
        shapes=raw_shapes, annotations=annotations_with_notes(raw_annotations, raw_note),
        updatemenus=[{
            "type": "buttons", "direction": "right", "active": 0,
            "x": 0.0, "y": 1.12, "xanchor": "left", "yanchor": "bottom",
            "buttons": [
                {"label": "Исходная шкала DICOM", "method": "update", "args": [{"visible": raw_visibility}, raw_layout]},
                {"label": "Нормированные циклы", "method": "update", "args": [{"visible": normalized_visibility}, normalized_layout]},
            ],
        }],
        uirevision=f"volume-toggle-{subject}",
    )
    return figure


def html_page(subject_sections: list[str], metadata_tables: list[str]) -> str:
    cards = [
        '<p><a href="21.00_Карта_4D_сердца_и_RR.md">21.00 — карта серии и пересборка</a> · <a href="21.01_4D_сердце_интерактивно.html">21.01 — обзор 4D</a></p>',
        SCIENTIFIC_CONTEXT_HTML,
    ]
    cards.append(
        "<h2>Исходные данные кардиосинхронизации</h2>"
        "<p>Для Adam и Nix встроены общий вид обезличенной растровой полосы GE "
        "ECG Report и точный фрагмент выделенного сканером интервала; для Georg "
        "такой отчёт не найден. R–R рассчитаны по "
        "<code>RpeakTimeStamps (0049,100C)</code>.</p>"
    )
    cards.extend(metadata_tables)
    cards.append(
        "<h2>Синхронные представления по испытуемым</h2>"
        "<p>В каждом блоке один ползунок управляет 3D-геометрией, положением точки "
        "на схематической P–QRS–T и выделением текущих объёмов четырёх камер. "
        "Схема ЭКГ не является индивидуальным зарегистрированным сигналом. "
        "Для Georg короткий и длинный R–R сохраняются на отдельных панелях.</p>"
        "<p>Основная горизонтальная ось показывает время в миллисекундах, а процент "
        "соответствующего R–R указан в скобках. Для Adam и Nix время отсчитано "
        "непрерывно от первой R-метки; PCHIP строится отдельно внутри каждого "
        "наблюдаемого интервала и не соединяет соседние сокращения. Для Georg "
        "короткий интервал 491 мс и длинный интервал 1159 мс показаны на двух "
        "панелях с собственными шкалами времени после R. ЧСС указана в заголовках "
        "и интерактивных подсказках. Серым отмечены части R–R, для которых фазы "
        "не реконструированы.</p>"
        '<p>Субъектная сводка DICOM: <a href="heart_rr_analysis/cardiac_dicom_metadata_by_subject.csv">'
        'cardiac_dicom_metadata_by_subject.csv</a>. Отдельные R–R: '
        '<a href="heart_rr_analysis/cardiac_gating_source_data.csv">cardiac_gating_source_data.csv</a>. '
        'Измеренные фазы: <a href="heart_rr_analysis/chamber_volume_changes_measured_cycles.csv">таблица</a>. '
        'Периодические профили Adam и Nix: '
        '<a href="heart_rr_analysis/chamber_volume_changes_closed_cycle.csv">шаг 1%</a> и '
        '<a href="heart_rr_analysis/chamber_volume_changes_closed_cycle_summary.csv">сводка</a>.</p>'
    )
    for fragment in subject_sections:
        cards.append(f'<section class="card">{fragment}</section>')
    return """<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Вычитание крови и объёмы камер</title><style>body{margin:0;background:#eef1f5;color:#1f2937;font-family:system-ui,-apple-system,'Segoe UI',sans-serif}main{max-width:1500px;margin:auto;padding:24px}.card{background:#fff;border-radius:14px;box-shadow:0 3px 16px #0001;margin:0 0 24px;padding:16px}h1,h2{margin-top:0}.warn{background:#fff4d6;border-left:5px solid #e0a100;padding:12px 16px;margin-bottom:20px}.sync-frame{display:block;width:100%;border:0}.ecg-strip{margin:16px 24px 8px}.ecg-strip.zoom{border-top:1px solid #d1d5db;padding-top:14px}.ecg-strip img{display:block;width:100%;height:auto;background:#fff}.ecg-strip figcaption{font-size:13px;color:#4b5563;margin-top:6px}.ecg-strip.missing{background:#f3f4f6;padding:12px;border-radius:8px}</style></head><body><main><h1>Вычитание крови из сердца и объёмы камер по R–R</h1><div class="warn">Разность «всё сердце − кровь» является геометрическим остатком двух независимых моделей, а не проверенной маской полного миокарда.</div>""" + "\n".join(cards) + "</main></body></html>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    workflow_root = Path(__file__).resolve().parent
    project_root = workflow_root.parent.parent
    data_root = project_root.parent / "Bitrix" / "ЭИТЛ" / "02 Big_data" / "3D" / "_DERIVED_CARDIAC4D"
    parser.add_argument("--derived-root", default=str(data_root))
    parser.add_argument("--output-dir", default=str(workflow_root / "heart_rr_analysis"))
    parser.add_argument(
        "--cardiac-metadata-summary",
        default=str(data_root / "_cohort_qc_v2" / "cardiac_phase_metadata_summary.json"),
    )
    parser.add_argument(
        "--ecg-report-evidence",
        default=str(data_root / "_cohort_qc_v2" / "ecg_report_evidence_summary.json"),
    )
    parser.add_argument("--subjects", nargs="+", default=["adam", "nix", "georg"])
    args = parser.parse_args()
    derived_root = Path(args.derived_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ecg_trace_dir = output_dir / "ecg_traces"
    ecg_trace_candidates = {
        "adam": ecg_trace_dir / "adam_series307_ecg_gating_strip.png",
        "nix": ecg_trace_dir / "nix_series307_ecg_gating_strip.png",
    }
    ecg_trace_paths = {
        subject: path if path.exists() else None for subject, path in ecg_trace_candidates.items()
    }
    ecg_window_candidates = {
        "adam": ecg_trace_dir / "adam_series307_ecg_scan_window.png",
        "nix": ecg_trace_dir / "nix_series307_ecg_scan_window.png",
    }
    ecg_window_paths = {
        subject: path if path.exists() else None for subject, path in ecg_window_candidates.items()
    }
    cardiac_metadata = json.loads(Path(args.cardiac_metadata_summary).read_text(encoding="utf-8"))
    ecg_report_evidence = json.loads(Path(args.ecg_report_evidence).read_text(encoding="utf-8"))
    timing_by_subject = {item["subject"]: item for item in cardiac_metadata["subjects"]}
    ecg_evidence_by_subject = {item["subject"]: item for item in ecg_report_evidence["subjects"]}
    config = {"responsive": True, "displaylogo": False, "scrollZoom": True}
    all_rows = []
    cycle_change_rows = []
    measured_cycle_change_rows = []
    cycle_profile_summaries = []
    gating_source_rows = []
    subject_dicom_rows = []
    summaries = []
    subtraction_figures = []
    volume_figures = []

    for subject in args.subjects:
        subject = subject.lower()
        conversion = json.loads((derived_root / subject / "phases" / "conversion_manifest.json").read_text(encoding="utf-8"))
        timing = timing_by_subject.get(subject)
        if timing is None:
            raise RuntimeError(f"No cardiac timing metadata for {subject}")
        phase_timing_rows = {float(item["nominal_percentage"]): item for item in timing["phase_rows"]}
        cycle_parameters = {
            int(item["cycle_index"]): item
            for item in timing["derived_multicycle_time_mapping"]["cycle_parameters"]
        }
        ecg_evidence = ecg_evidence_by_subject.get(subject)
        subject_dicom_rows.append(build_subject_dicom_row(timing, ecg_evidence))
        for cycle_index, cycle_parameter in sorted(cycle_parameters.items()):
            rr_ms = float(
                cycle_parameter.get("rr_interval_ms")
                or cycle_parameter["derived_rr_interval_ms"]
            )
            gating_source_rows.append({
                "subject": subject,
                "source_series_folder_name": timing["source_folder_name"],
                "dicom_files_read": timing["dicom_files_read"],
                "cycle_index": cycle_index + 1,
                "rpeak_start_s": cycle_parameter.get("rpeak_start_s", ""),
                "rpeak_end_s": cycle_parameter.get("rpeak_end_s", ""),
                "rr_interval_ms": rr_ms,
                "heart_rate_from_rpeaks_bpm": cycle_parameter.get(
                    "heart_rate_from_rpeak_interval_bpm", 60000.0 / rr_ms
                ),
                "median_image_heart_rate_bpm": cycle_parameter["median_image_heart_rate_bpm"],
                "image_vs_rpeak_heart_rate_difference_bpm": cycle_parameter.get(
                    "image_vs_rpeak_heart_rate_difference_bpm", ""
                ),
                "heart_rate_at_confirm_bpm": first_metadata_value(timing, "HeartRateAtConfirm"),
                "average_heart_rate_prior_to_confirm_bpm": first_metadata_value(timing, "AvgHeartRatePriorToConfirm"),
                "minimum_heart_rate_prior_to_confirm_bpm": first_metadata_value(timing, "MinHeartRatePriorToConfirm"),
                "maximum_heart_rate_prior_to_confirm_bpm": first_metadata_value(timing, "MaxHeartRatePriorToConfirm"),
                "stddev_heart_rate_prior_to_confirm_bpm": first_metadata_value(timing, "StdDevHeartRatePriorToConfirm"),
                "heart_rate_samples_prior_to_confirm": first_metadata_value(timing, "NumHeartRateSamplesPriorToConfirm"),
                "ecg_monitor_type": first_metadata_value(timing, "EkgMonitorType"),
                "reconstruction_sectors": first_metadata_value(timing, "NumReconSectors"),
                "source_tag": "GE RpeakTimeStamps (0049,100C)",
                "interpretation_status": (
                    "short_long_trigger_pair_mechanism_unresolved_no_waveform"
                    if subject == "georg" else
                    "two_consistent_trigger_intervals_no_waveform"
                ),
            })
        phases = []
        for phase_info in conversion["phases"]:
            phase_id = phase_info["phase_id"]
            rr_percent = float(phase_info["dicom_phase_value"])
            timing_row = phase_timing_rows.get(rr_percent)
            if timing_row is None:
                raise RuntimeError(f"No timing row for {subject}/{phase_id}: {rr_percent:g}%")
            cycle_index = int(timing_row["cycle_index"])
            cycle_parameter = cycle_parameters[cycle_index]
            product_dir = derived_root / subject / "products_v2" / phase_id
            raw_dir = derived_root / subject / "automatic_masks_v2" / phase_id / "heartchambers_highres"
            whole_image = nib.load(str(product_dir / "whole_heart.nii.gz"))
            blood_image = nib.load(str(product_dir / "blood_pool.nii.gz"))
            myocardium_image = nib.load(str(product_dir / "myocardium.nii.gz"))
            if not same_grid(whole_image, blood_image) or not same_grid(whole_image, myocardium_image):
                raise RuntimeError(f"Grid mismatch: {subject}/{phase_id}")
            whole = np.asanyarray(whole_image.dataobj) > 0
            blood = np.asanyarray(blood_image.dataobj) > 0
            myocardium = np.asanyarray(myocardium_image.dataobj) > 0
            remainder = whole & ~blood
            blood_outside = blood & ~whole
            phase_output = derived_root / subject / "heart_minus_blood" / phase_id
            save_mask(remainder, whole_image, phase_output / "heart_minus_blood.nii.gz")
            save_mask(blood_outside, whole_image, phase_output / "blood_outside_whole_heart.nii.gz")
            chambers_ml = {}
            chamber_meshes = []
            for key, _, _ in CHAMBERS:
                chamber_image = nib.load(str(raw_dir / f"{key}.nii.gz"))
                if not same_grid(whole_image, chamber_image):
                    raise RuntimeError(f"Grid mismatch: {subject}/{phase_id}/{key}")
                chamber = np.asanyarray(chamber_image.dataobj) > 0
                chambers_ml[key] = volume_ml(chamber, chamber_image.affine)
                chamber_meshes.append(surface(chamber & whole, whole_image.affine))
            del chamber
            surface_meshes = [surface(remainder, whole_image.affine), *chamber_meshes, surface(myocardium, whole_image.affine)]
            rr_interval_ms = float(
                cycle_parameter.get("rr_interval_ms")
                or cycle_parameter["derived_rr_interval_ms"]
            )
            phase = {
                "phase_id": phase_id, "rr_percent": rr_percent,
                "cycle_index": cycle_index,
                "phase_percent_within_cycle": float(timing_row["phase_percent_within_cycle"]),
                "image_heart_rate_bpm": float(timing_row["AvgHeartRateForImage"]["median"]),
                "rr_interval_ms": rr_interval_ms,
                "heart_rate_from_rpeaks_bpm": float(
                    cycle_parameter.get("heart_rate_from_rpeak_interval_bpm", 60000.0 / rr_interval_ms)
                ),
                "rpeak_start_s": cycle_parameter.get("rpeak_start_s", ""),
                "rpeak_end_s": cycle_parameter.get("rpeak_end_s", ""),
                "time_after_cycle_r_ms": float(
                    timing_row["phase_percent_within_cycle"] / 100.0 * rr_interval_ms
                ),
                "derived_time_from_R0_ms": float(timing_row["derived_time_from_R0_ms"]),
                "temporal_resolution_ms": 1000.0 * float(timing_row["GETemporalResolutionSeconds"]["median"]),
                "temporal_center_view_angle_degrees": float(timing_row["TemporalCenterViewAngle"]["median"]),
                "surface_meshes": surface_meshes,
                "chambers_ml": chambers_ml,
                "whole_heart_ml": volume_ml(whole, whole_image.affine),
                "heart_minus_blood_ml": volume_ml(remainder, whole_image.affine),
                "blood_outside_whole_heart_ml": volume_ml(blood_outside, whole_image.affine),
            }
            phases.append(phase)
            row = {k: v for k, v in phase.items() if k not in {"surface_meshes", "chambers_ml"}}
            row.update({f"{key}_ml": value for key, value in chambers_ml.items()})
            all_rows.append({"subject": subject, **row})

        merge_assessment = assess_cycle_merge(phases)
        profiles = (
            {key: periodic_cycle_profile(phases, key) for key, _, _ in CHAMBERS}
            if merge_assessment["allowed"]
            else None
        )
        for cycle_index in sorted({int(phase["cycle_index"]) for phase in phases}):
            cycle_phases = sorted(
                (phase for phase in phases if int(phase["cycle_index"]) == cycle_index),
                key=lambda phase: phase["phase_percent_within_cycle"],
            )
            for key, label, _ in CHAMBERS:
                baseline = float(cycle_phases[0]["chambers_ml"][key])
                previous = baseline
                for phase in cycle_phases:
                    value = float(phase["chambers_ml"][key])
                    measured_cycle_change_rows.append({
                        "subject": subject,
                        "cycle_index": cycle_index + 1,
                        "chamber_key": key,
                        "chamber_label": label,
                        "raw_rr_percent": float(phase["rr_percent"]),
                        "phase_percent_within_cycle": float(phase["phase_percent_within_cycle"]),
                        "heart_rate_bpm": float(phase["image_heart_rate_bpm"]),
                        "volume_ml": value,
                        "change_from_first_measured_phase_ml": value - baseline,
                        "change_from_previous_measured_phase_ml": value - previous,
                        "cardiac_phase": (
                            phase_band_label(float(phase["phase_percent_within_cycle"]))
                            if subject != "georg" else "Без индивидуальной фазовой метки"
                        ),
                        "status": "measured_automatic_segmentation_pending_manual_review",
                    })
                    previous = value
        sub_figure = build_synchronized_cardiac_figure(
            subject,
            phases,
            SURFACES,
            CHAMBERS,
            within_cycle_profile,
            schematic_ecg,
        )
        vol_figure = build_volume_figure_ms(subject, phases, CHAMBERS, within_cycle_profile)
        subtraction_figures.append((subject, sub_figure))
        volume_figures.append((subject, vol_figure, profiles, merge_assessment))
        sub_figure.write_html(output_dir / f"{subject}_heart_minus_blood_4d.html", include_plotlyjs=True, full_html=True, auto_play=False, config=config)
        vol_figure.write_html(output_dir / f"{subject}_chamber_volumes_rr.html", include_plotlyjs=True, full_html=True, config=config)
        summary = {
            "subject": subject,
            "phase_count": len(phases),
            "rr_percent_range": [phases[0]["rr_percent"], phases[-1]["rr_percent"]],
            "manufacturer_cycle_parameters": timing["derived_multicycle_time_mapping"]["cycle_parameters"],
            "manufacturer_rpeak_timing": timing.get("manufacturer_rpeak_timing"),
            "center_view_angle_cross_check": timing["derived_multicycle_time_mapping"]["center_view_angle_cross_check"],
            "ecg_report_evidence": ecg_evidence,
            "cycle_merge_assessment": merge_assessment,
            "chambers": {},
            "closed_cycle_profiles": {},
        }
        for key, label, _ in CHAMBERS:
            values = [phase["chambers_ml"][key] for phase in phases]
            summary["chambers"][key] = {"label_ru": label, "minimum_ml": min(values), "maximum_ml": max(values), "range_ml": max(values) - min(values)}
            if profiles is None:
                profile_summary = {
                    "label_ru": label,
                    "status": "not_calculated_cycle_merge_rejected",
                    "method": "not_calculated",
                    "boundary_volume_ml": "",
                    "minimum_ml": "",
                    "maximum_ml": "",
                    "peak_to_peak_ml": "",
                    "phase_of_minimum_percent": "",
                    "phase_of_maximum_percent": "",
                    "measured_phase_count": len({float(phase["phase_percent_within_cycle"]) for phase in phases}),
                    "measured_observation_count": len(phases),
                }
                summary["closed_cycle_profiles"][key] = profile_summary
                cycle_profile_summaries.append({
                    "subject": subject,
                    "chamber_key": key,
                    "chamber_label": label,
                    **profile_summary,
                })
                continue
            profile = profiles[key]
            profile_summary = {
                "label_ru": label,
                "status": "calculated_provisional",
                "method": profile["model"],
                "harmonics": profile["harmonics"],
                "harmonic_ridge": profile["harmonic_ridge"],
                "cycle_contrast_ridge": profile["cycle_contrast_ridge"],
                "cycle_offsets_ml": json.dumps(profile["cycle_offsets_ml"], ensure_ascii=False),
                "residual_rmse_ml": profile["residual_rmse_ml"],
                "residual_max_abs_ml": profile["residual_max_abs_ml"],
                "boundary_volume_ml": profile["boundary_volume_ml"],
                "minimum_ml": profile["minimum_ml"],
                "maximum_ml": profile["maximum_ml"],
                "peak_to_peak_ml": profile["maximum_ml"] - profile["minimum_ml"],
                "phase_of_minimum_percent": profile["phase_of_minimum_percent"],
                "phase_of_maximum_percent": profile["phase_of_maximum_percent"],
                "measured_phase_count": len({item["phase_percent"] for item in profile["observations"]}),
                "measured_observation_count": len(profile["observations"]),
            }
            summary["closed_cycle_profiles"][key] = profile_summary
            cycle_profile_summaries.append({
                "subject": subject,
                "chamber_key": key,
                "chamber_label": label,
                **profile_summary,
            })
            for index, (phase_percent, modeled_volume, delta_from_0) in enumerate(zip(
                profile["phase_percent"], profile["volume_ml"], profile["delta_from_0_ml"]
            )):
                measured = [
                    item["volume_ml"] for item in profile["observations"]
                    if np.isclose(item["phase_percent"], phase_percent)
                ]
                previous_delta = 0.0 if index == 0 else modeled_volume - profile["volume_ml"][index - 1]
                cycle_change_rows.append({
                    "subject": subject,
                    "chamber_key": key,
                    "chamber_label": label,
                    "phase_percent": float(phase_percent),
                    "cardiac_phase": phase_band_label(float(phase_percent)),
                    "modeled_volume_ml": float(modeled_volume),
                    "change_from_0_ml": float(delta_from_0),
                    "change_from_0_percent": (
                        float(100.0 * delta_from_0 / profile["boundary_volume_ml"])
                        if not np.isclose(profile["boundary_volume_ml"], 0.0) else ""
                    ),
                    "change_from_previous_1pct_ml": float(previous_delta),
                    "measured_count": len(measured),
                    "measured_mean_ml": float(np.mean(measured)) if measured else "",
                    "measured_min_ml": float(np.min(measured)) if measured else "",
                    "measured_max_ml": float(np.max(measured)) if measured else "",
                    "model": profile["model"],
                    "model_residual_rmse_ml": profile["residual_rmse_ml"],
                    "cycle_offsets_ml": json.dumps(profile["cycle_offsets_ml"], ensure_ascii=False),
                    "phase_kind": (
                        "periodic_model_boundary" if phase_percent in (0.0, 100.0)
                        else "measured_phase_reference" if measured
                        else "periodic_model"
                    ),
                })
        outside = [phase["blood_outside_whole_heart_ml"] for phase in phases]
        summary["blood_outside_whole_heart_ml"] = {"minimum": min(outside), "maximum": max(outside)}
        summaries.append(summary)
        (derived_root / subject / "heart_minus_blood" / "manifest.json").write_text(json.dumps({"schema_version": 1, "created_at": utc_now(), "subject": subject, "definition": "whole_heart AND NOT blood_pool", "warning": "geometric remainder of independent models; not validated full myocardium", "phases": [{k: v for k, v in row.items() if k not in {"subject"}} for row in all_rows if row["subject"] == subject]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "chamber_volumes_by_rr.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    with (output_dir / "chamber_volume_changes_measured_cycles.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measured_cycle_change_rows[0]))
        writer.writeheader(); writer.writerows(measured_cycle_change_rows)
    with (output_dir / "chamber_volume_changes_closed_cycle.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cycle_change_rows[0]))
        writer.writeheader(); writer.writerows(cycle_change_rows)
    with (output_dir / "chamber_volume_changes_closed_cycle_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cycle_profile_summaries[0]))
        writer.writeheader(); writer.writerows(cycle_profile_summaries)
    with (output_dir / "cardiac_gating_source_data.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gating_source_rows[0]))
        writer.writeheader(); writer.writerows(gating_source_rows)
    with (output_dir / "cardiac_dicom_metadata_by_subject.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subject_dicom_rows[0]))
        writer.writeheader(); writer.writerows(subject_dicom_rows)
    summary_payload = {
        "schema_version": 6,
        "created_at": utc_now(),
        "dicom_phase_field": "NominalPercentageOfCardiacPhase",
        "phase_interpretation_status": "multicycle_coordinate_supported_by_ge_rpeak_timestamps_for_all_subjects_and_ecg_report_for_adam_nix",
        "supporting_cardiac_timing_fields_present": True,
        "timing_definition": "piecewise time from GE RpeakTimeStamps; cross-checked against AvgHeartRateForImage and TemporalCenterViewAngle",
        "ecg_context": "full scanner ECG report strip and pixel-exact GE-highlighted scan-window crop embedded for Adam and Nix; no digital DICOM waveform; schematic P-QRS-T retained only as orientation",
        "georg_heart_rate_interpretation": "single-series short-long trigger pair: 491 and 1159 ms, corresponding to 122.20 and 51.77 bpm; physiological versus trigger-detection cause unresolved without waveform",
        "cycle_merge_policy": {
            "criterion": "maximum relative difference between median image heart rates of cycles",
            "threshold": 0.10,
            "status": "provisional methodological gate, not physiologically validated",
        },
        "closed_cycle_subjects": [item["subject"] for item in summaries if item["cycle_merge_assessment"]["allowed"]],
        "rejected_cycle_merge_subjects": [item["subject"] for item in summaries if not item["cycle_merge_assessment"]["allowed"]],
        "volume_plot_coordinate": "milliseconds; local R-R percentage is retained as a secondary point label",
        "adam_nix_volume_plot": "continuous milliseconds from the first R marker; separate within-interval PCHIP segments",
        "georg_volume_plot": "two independent panels in milliseconds after each R trigger; no cross-interval interpolation or uniform percentage alignment",
        "normalized_cycle_definition": "provisional composite only for cycles passing the 10 percent heart-rate consistency gate; all observations remain visible",
        "closure_method": "accepted composites: two-harmonic periodic ridge model; equality at 0 and 100 percent follows from the Fourier basis",
        "interpolation_method": "display: separate PCHIP within each observed trigger interval in milliseconds; secondary Adam/Nix numeric composite: periodic two-harmonic ridge model with shrunken cycle contrasts on a 1 percent grid",
        "cycle_variation_model": "cycle-specific additive contrasts with sum-to-zero interpretation; ridge penalties are fixed descriptive regularization, not inferred uncertainty",
        "phase_annotation_status": "empirical volume-direction bands for accepted Adam/Nix composite only; no valve-event phase labels; disabled for Georg",
        "phase_bands": [
            {"start_percent": start, "end_percent": end, "label_ru": label.replace("<br>", " ")}
            for start, end, label, _ in CARDIAC_PHASE_BANDS
        ],
        "embedded_ecg_strip_subjects": sorted(
            subject for subject, path in ecg_trace_paths.items() if path is not None
        ),
        "embedded_ecg_scan_window_subjects": sorted(
            subject for subject, path in ecg_window_paths.items() if path is not None
        ),
        "output_tables": {
            "raw_measured_volumes": "chamber_volumes_by_rr.csv",
            "cardiac_gating_source_data": "cardiac_gating_source_data.csv",
            "cardiac_dicom_metadata_by_subject": "cardiac_dicom_metadata_by_subject.csv",
            "measured_cycle_changes": "chamber_volume_changes_measured_cycles.csv",
            "closed_cycle_1pct_changes": "chamber_volume_changes_closed_cycle.csv",
            "closed_cycle_summary": "chamber_volume_changes_closed_cycle_summary.csv",
        },
        "subjects": summaries,
        "status": "automatic_segmentation_pending_manual_review",
    }
    (output_dir / "rr_and_subtraction_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subject_sections = []
    for subject, _ in subtraction_figures:
        frame_height = 1260 if subject == "georg" else 940
        trace_fragment = ecg_strip_html(
            subject,
            ecg_trace_paths.get(subject),
            ecg_window_paths.get(subject),
            ecg_evidence_by_subject.get(subject),
        )
        subject_sections.append(
            f"<h2>{subject.upper()}: 3D-маски, схема ЭКГ и объёмы камер</h2>"
            f'<iframe class="sync-frame" src="heart_rr_analysis/{subject}_heart_minus_blood_4d.html" '
            f'height="{frame_height}" loading="lazy"></iframe>'
            + trace_fragment
        )
    metadata_tables = [subject_dicom_table_html(subject_dicom_rows), gating_source_table_html(gating_source_rows)]
    combined = output_dir.parent / "21.03_Вычитание_крови_и_объёмы_камер_RR.html"
    combined.write_text(html_page(subject_sections, metadata_tables), encoding="utf-8")

    notebook = nbformat.v4.new_notebook(metadata={"language_info": {"name": "python"}, "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "[21.00 — карта серии и пересборка](21.00_Карта_4D_сердца_и_RR.md) · [21.01 — обзор 4D](21.01_4D_сердце_интерактивно.ipynb)\n\n"
        + NOTEBOOK_CONTEXT_MD
    ))
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "## Исходные данные кардиосинхронизации\n\n"
        "В первой таблице собраны субъектные поля DICOM, включая исходные фазы, "
        "R-метки, частоты, параметры реконструкции и наличие GE ECG Report: "
        "[cardiac_dicom_metadata_by_subject.csv]"
        "(heart_rr_analysis/cardiac_dicom_metadata_by_subject.csv). "
        "Во второй таблице приведены отдельные R–R-интервалы. Полная таблица: "
        "[cardiac_gating_source_data.csv]"
        "(heart_rr_analysis/cardiac_gating_source_data.csv)."
    ))
    notebook.cells.append(nbformat.v4.new_code_cell(
        source="# Исходные временные метки GE и рассчитанные по ним R–R/ЧСС.",
        outputs=[nbformat.v4.new_output("display_data", data={"text/html": subject_dicom_table_html(subject_dicom_rows) + gating_source_table_html(gating_source_rows)}, metadata={})],
    ))
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "## Отдельный разбор коротко-длинной пары Georg\n\n"
        "DICOM подтверждает использование реконструкцией интервалов 491 и 1159 мс, "
        "но не позволяет установить причину. Возможны преждевременный комплекс с "
        "паузой, лишний либо пропущенный R-триггер или ECG-edit. До получения цифровой "
        "ЭКГ и журнала редактирования оба участка анализируются раздельно. При "
        "отсутствии дополнительных данных Georg не включается в композитные фазовые "
        "оценки. Временные графики короткого и длинного интервалов показаны на "
        "отдельных панелях в миллисекундах после R; проценты оставлены в скобках "
        "как вторичная координата. Литературные основания: "
        "[Matsutani et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18577814/) и "
        "[Kondo et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24582039/)."
    ))
    for subject, _ in subtraction_figures:
        frame_height = 1260 if subject == "georg" else 940
        iframe = (
            f'<iframe src="heart_rr_analysis/{subject}_heart_minus_blood_4d.html" '
            f'width="100%" height="{frame_height}" style="border:0"></iframe>'
        )
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            f"## {subject.upper()}: синхронное представление 3D-масок, ЭКГ и объёмов"
        ))
        notebook.cells.append(nbformat.v4.new_code_cell(
            source=(
                "# Один ползунок одновременно меняет 3D-фазу и выделяет "
                "соответствующие точки на схеме ЭКГ и графиках объёма."
            ),
            outputs=[nbformat.v4.new_output(
                "display_data", data={"text/html": iframe}, metadata={}
            )],
        ))
        trace_path = ecg_trace_paths.get(subject)
        if trace_path is not None:
            trace_fragment = ecg_strip_html(
                subject,
                trace_path,
                ecg_window_paths.get(subject),
                ecg_evidence_by_subject.get(subject),
            )
            notebook.cells.append(nbformat.v4.new_code_cell(
                source="# Общий вид полосы GE и точный фрагмент выделенного интервала встроены как PNG.",
                outputs=[nbformat.v4.new_output("display_data", data={"text/html": trace_fragment}, metadata={})],
            ))
        else:
            notebook.cells.append(nbformat.v4.new_markdown_cell(
                f"**{subject.upper()}:** отчёт GE ECG Report в доступных КТ-данных не найден."
            ))
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "## Расчётные изменения объёмов по фазам\n\n"
        "Изменения между фактически реконструированными фазами каждого отдельного "
        "цикла приведены для всех трёх испытуемых: "
        "[chamber_volume_changes_measured_cycles.csv]"
        "(heart_rr_analysis/chamber_volume_changes_measured_cycles.csv). "
        "Периодические профили с шагом 1% R–R рассчитаны только для Adam и Nix: "
        "относительное различие медианной ЧСС между их циклами не превышает "
        "методический порог 10%. Для Georg объединение отклонено из-за ЧСС "
        "122 и 52 уд/мин; два частичных цикла показаны раздельно без экстраполяции. "
        "Таблица профилей содержит объём, изменение относительно 0%, остаток модели, "
        "межцикловые поправки и "
        "изменение относительно предыдущего шага: "
        "[chamber_volume_changes_closed_cycle.csv](heart_rr_analysis/chamber_volume_changes_closed_cycle.csv). "
        "Сводка по диапазонам и фазам экстремумов: "
        "[chamber_volume_changes_closed_cycle_summary.csv](heart_rr_analysis/chamber_volume_changes_closed_cycle_summary.csv)."
    ))
    def summary_row_html(row: dict) -> str:
        prefix = f"<tr><td>{row['subject'].upper()}</td><td>{row['chamber_label']}</td>"
        if row["status"] != "calculated_provisional":
            return (
                prefix
                + '<td colspan="6">Не рассчитано: объединение циклов отклонено '
                'из-за несогласованной ЧСС.</td></tr>'
            )
        number_ru = lambda value: f"{value:.1f}".replace(".", ",")
        return (
            prefix
            + f"<td>{number_ru(row['boundary_volume_ml'])}</td>"
            f"<td>{number_ru(row['minimum_ml'])} ({row['phase_of_minimum_percent']:.0f}%)</td>"
            f"<td>{number_ru(row['maximum_ml'])} ({row['phase_of_maximum_percent']:.0f}%)</td>"
            f"<td>{number_ru(row['peak_to_peak_ml'])}</td>"
            f"<td>{number_ru(row['residual_rmse_ml'])}</td>"
            f"<td><code>{row['cycle_offsets_ml']}</code></td></tr>"
        )

    summary_table_body = "".join(
        summary_row_html(row) for row in cycle_profile_summaries
    )
    summary_table_html = (
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">'
        '<thead><tr><th>Испытуемый</th><th>Камера</th><th>V(0)=V(100), мл</th>'
        '<th>Минимум, мл (фаза)</th><th>Максимум, мл (фаза)</th>'
        '<th>Размах, мл</th><th>RMSE модели, мл</th><th>Поправки циклов, мл</th></tr></thead><tbody>' + summary_table_body + '</tbody></table></div>'
        '<style>th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:left}th{background:#f3f4f6}</style>'
    )
    notebook.cells.append(nbformat.v4.new_code_cell(
        source="# Сводка периодических расчётных профилей четырёх камер.",
        outputs=[nbformat.v4.new_output("display_data", data={"text/html": summary_table_html}, metadata={})],
    ))
    nbformat.write(notebook, output_dir.parent / "21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb")
    print(json.dumps({"subjects": [item[0] for item in subtraction_figures], "rows": len(all_rows), "combined_html": combined.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
