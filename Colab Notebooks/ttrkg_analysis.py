"""Проверяемые численные операции для серий 40 и 50.

Модуль не кодирует приборную калибровку, анатомию или физиологическую
интерпретацию. Эти входы должны быть приняты внешними контрактами ноутбуков.
"""

from __future__ import annotations

import numpy as np


def ensemble_waveform(time_s, signal, rpeaks_s, interval_s, grid_s, baseline_s):
    """Строит знаковый ансамбль с отдельной базовой линией каждого цикла."""
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    rpeaks_s = np.asarray(rpeaks_s, dtype=float)
    grid_s = np.asarray(grid_s, dtype=float)
    if time_s.ndim != 1 or signal.shape != time_s.shape:
        raise ValueError("Время и сигнал должны быть одномерными и равной длины")
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
        raise ValueError("Время должно строго возрастать")
    left, right = map(float, interval_s)
    base_left, base_right = map(float, baseline_s)
    base_mask = (grid_s >= base_left) & (grid_s <= base_right)
    if base_mask.sum() < 2:
        raise ValueError("Базовое окно должно содержать не менее двух точек")
    beats = []
    rejected = 0
    for rpeak in rpeaks_s:
        sample_time = rpeak + grid_s
        if sample_time[0] < left or sample_time[-1] > right or sample_time[0] < time_s[0] or sample_time[-1] > time_s[-1]:
            rejected += 1
            continue
        beat = np.interp(sample_time, time_s, signal)
        if not np.isfinite(beat).all():
            rejected += 1
            continue
        beats.append(beat - beat[base_mask].mean())
    if len(beats) < 3:
        raise RuntimeError("После QC осталось менее трёх циклов")
    beats = np.asarray(beats)
    return {
        "mean": beats.mean(axis=0),
        "se_within_record": beats.std(axis=0, ddof=1) / np.sqrt(len(beats)),
        "n_beats": int(len(beats)),
        "n_rejected": int(rejected),
    }


def validate_fractional_operator(region_names, sensitivities, *, complete_partition, sum_tolerance=1e-6):
    """Проверяет региональный оператор, не запрещая отрицательную чувствительность."""
    names = list(region_names)
    values = np.asarray(sensitivities, dtype=float)
    if values.ndim != 1 or len(names) != len(values) or not names:
        raise ValueError("Названия и чувствительности должны задавать непустой вектор")
    if len(set(names)) != len(names) or not np.isfinite(values).all():
        raise ValueError("Названия должны быть уникальны, значения — конечны")
    total = float(values.sum())
    if complete_partition and abs(total - 1.0) > float(sum_tolerance):
        raise ValueError("Полный региональный оператор нарушает проверку однородности")
    return {"sum": total, "has_negative": bool(np.any(values < 0.0))}


def apply_fractional_operator(sensitivities, fractional_changes):
    """Вычисляет знаковый вклад тканей: sum_i S_i * delta_rho_i/rho_i."""
    sensitivities = np.asarray(sensitivities, dtype=float)
    fractional_changes = np.asarray(fractional_changes, dtype=float)
    if sensitivities.ndim != 1 or fractional_changes.ndim != 2:
        raise ValueError("Ожидаются вектор чувствительностей и матрица регион×время")
    if fractional_changes.shape[0] != sensitivities.size:
        raise ValueError("Число регионов оператора и сигналов не совпадает")
    if not np.isfinite(sensitivities).all() or not np.isfinite(fractional_changes).all():
        raise ValueError("Оператор и сигналы должны быть конечными")
    return sensitivities @ fractional_changes


def residual_candidate(measured, predicted, measured_variance=None, predicted_variance=None, cross_covariance=None):
    """Вычитает модельный вклад и, если возможно, переносит дисперсию."""
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if measured.shape != predicted.shape or not np.isfinite(measured).all() or not np.isfinite(predicted).all():
        raise ValueError("Измеренный и предсказанный сигналы должны совпадать по форме и быть конечными")
    result = {"residual": measured - predicted, "variance": None}
    supplied = [measured_variance, predicted_variance, cross_covariance]
    if all(value is None for value in supplied):
        return result
    if measured_variance is None or predicted_variance is None:
        raise ValueError("Для переноса неопределённости нужны обе дисперсии")
    vm = np.asarray(measured_variance, dtype=float)
    vp = np.asarray(predicted_variance, dtype=float)
    cov = np.zeros_like(vm) if cross_covariance is None else np.asarray(cross_covariance, dtype=float)
    if vm.shape != measured.shape or vp.shape != measured.shape or cov.shape != measured.shape:
        raise ValueError("Формы дисперсий и ковариации должны совпадать с сигналом")
    if not np.isfinite(vm).all() or not np.isfinite(vp).all() or not np.isfinite(cov).all():
        raise ValueError("Дисперсии и ковариация должны быть конечными")
    if np.any(vm < 0.0) or np.any(vp < 0.0):
        raise ValueError("Дисперсии не могут быть отрицательными")
    covariance_limit = np.sqrt(vm * vp)
    if np.any(np.abs(cov) > covariance_limit + 1e-15):
        raise ValueError("Ковариация нарушает неравенство Коши—Буняковского")
    variance = vm + vp - 2.0 * cov
    if np.any(variance < -1e-15):
        raise ValueError("Получена отрицательная дисперсия остатка")
    result["variance"] = np.maximum(variance, 0.0)
    return result


def matrix_diagnostics(matrix, relative_tolerance=None):
    """Возвращает ранг, сингулярные числа, обусловленность и правое ядро."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Матрица должна быть конечной и двумерной")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    if relative_tolerance is None:
        tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    else:
        tolerance = float(relative_tolerance) * singular[0]
    rank = int(np.sum(singular > tolerance))
    condition = float(np.inf if singular[-1] <= tolerance else singular[0] / singular[-1])
    nullspace = vh[rank:].T
    return {
        "rank": rank,
        "singular_values": singular,
        "condition": condition,
        "nullspace": nullspace,
        "tolerance": float(tolerance),
    }


def robust_signal_metrics(signal, lower_quantile=0.01, upper_quantile=0.99):
    """Описывает уровень и размах сигнала без физиологической интерпретации."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Сигнал должен быть конечным одномерным массивом")
    lower_quantile, upper_quantile = float(lower_quantile), float(upper_quantile)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Некорректные границы квантилей")
    q_low, q_high = np.quantile(values, [lower_quantile, upper_quantile])
    minimum, maximum, median = float(values.min()), float(values.max()), float(np.median(values))
    return {"n_samples": int(values.size), "median": median, "mad": float(np.median(np.abs(values - median))), "minimum": minimum, "maximum": maximum, "peak_to_peak": float(maximum - minimum), "robust_range": float(q_high - q_low), "fraction_at_exact_extremes": float(np.mean((values == minimum) | (values == maximum)))}


def central_difference(y_minus, y_plus, step):
    """Центральная конечная разность dY/dp для скаляра или массива Y."""
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Шаг возмущения должен быть положительным")
    minus, plus = np.asarray(y_minus, dtype=float), np.asarray(y_plus, dtype=float)
    if minus.shape != plus.shape or not np.isfinite(minus).all() or not np.isfinite(plus).all():
        raise ValueError("Y(p-step) и Y(p+step) должны совпадать по форме и быть конечными")
    return (plus - minus) / (2.0 * step)


def scaled_local_sensitivity(y_minus, y_zero, y_plus, parameter_zero, step, output_scale=None):
    """Возвращает dY/dp и масштабированную чувствительность p0/Yscale*dY/dp."""
    baseline = np.asarray(y_zero, dtype=float)
    if not np.isfinite(baseline).all():
        raise ValueError("Базовый выход должен быть конечным")
    parameter_zero = float(parameter_zero)
    if not np.isfinite(parameter_zero) or parameter_zero == 0.0:
        raise ValueError("Базовое значение параметра должно быть конечным и ненулевым")
    derivative = central_difference(y_minus, y_plus, step)
    if output_scale is None:
        if baseline.ndim != 0 or float(abs(baseline)) == 0.0:
            raise ValueError("Для массива или нулевого выхода нужен явный output_scale")
        scale = float(abs(baseline))
    else:
        scale = float(output_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("output_scale должен быть положительным")
    return {"derivative": derivative, "scaled_sensitivity": derivative * parameter_zero / scale, "output_scale": scale}


def relative_derivative_disagreement(first, second):
    """Относительное расхождение двух оценок производной в норме L2."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Производные должны совпадать по форме и быть конечными")
    denominator = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), np.finfo(float).eps)
    return float(np.linalg.norm(first - second) / denominator)
