"""
tests/test_pipeline.py — Smoke-тесты пайплайна на синтетическом фантоме.

Тесты не требуют реальных DICOM-данных и TotalSegmentator.
Они проверяют логику постобработки и QC на numpy-фантомах с известными HU.

Запуск:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py        # без pytest
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Добавить корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Вспомогательные функции ────────────────────────────────────────────────

def make_phantom(shape=(30, 40, 50)) -> tuple[np.ndarray, dict]:
    """
    Создаёт синтетический KT-фантом с известными зонами HU.

    Структура (по оси Z):
      - внешний слой (2 вокселя): воздух, HU = -1000
      - подкожный жир (2 вокселя): HU = -100
      - мышцы (середина, широко): HU = 50
      - лёгкие (левая половина Y): HU = -700
      - кость (центральный столбик): HU = 700
      - сердце (правая половина Y, Z-центр): HU = 60

    Возвращает ct_data и dict с ground-truth масками.
    """
    Z, Y, X = shape
    ct = np.full(shape, 50, dtype=np.float32)  # мышцы по умолчанию

    # Воздух вокруг тела
    ct[:, :, :] = 50

    # Лёгкие — левая половина по Y, весь X
    ct[:, :Y//2, :] = -700

    # Кость — центральный столбик X
    cx = X // 2
    ct[:, :, cx-2:cx+2] = 700

    # Сердце — правая четверть Y, Z-центр
    zc = Z // 2
    ct[zc-3:zc+3, Y//2:3*Y//4, X//4:3*X//4] = 65

    # Подкожный жир — внешняя оболочка (1 воксель)
    ct[0, :, :] = -100
    ct[-1, :, :] = -100
    ct[:, 0, :] = -100
    ct[:, -1, :] = -100

    # Ground-truth маски (бинарные)
    gt = {
        "bone": (ct >= 300).astype(np.uint8),
        "lung": ((ct <= -500) & (ct > -1000)).astype(np.uint8),
        "muscle": ((ct >= 20) & (ct < 300)).astype(np.uint8),
    }

    return ct, gt


def make_fake_tissue_masks(shape, gt: dict) -> dict[str, np.ndarray]:
    """Имитирует вывод TotalSegmentator — бинарные маски по тканям."""
    Z, Y, X = shape
    masks = {
        "bone": gt["bone"].copy(),
        "lung": gt["lung"].copy(),
        "heart": np.zeros(shape, dtype=np.uint8),
        "muscle": gt["muscle"].copy(),
        "fat_subcutaneous": np.zeros(shape, dtype=np.uint8),
    }
    # Небольшое сердце в центре
    zc, yc, xc = Z//2, 3*Y//4, X//2
    masks["heart"][zc-2:zc+2, Y//2:3*Y//4, X//4:3*X//4] = 1

    return masks


# ─── Тесты постобработки ────────────────────────────────────────────────────

def test_generate_body_mask():
    """Маска тела должна включать все воксели выше порога."""
    sys.path.insert(0, str(ROOT))
    from importlib import import_module
    pp = import_module("03_postprocess")

    ct, _ = make_phantom()
    body = pp.generate_body_mask(ct, threshold=-500)

    # Все воксели выше -500 HU должны быть в маске тела
    expected = (ct > -500).astype(np.uint8)
    # После fill_holes и keep_largest — должно быть примерно то же
    assert body.sum() >= expected.sum() * 0.9, (
        f"Маска тела слишком мала: {body.sum()} vs {expected.sum()}"
    )
    print("  PASS: generate_body_mask")


def test_priority_rule_full_coverage():
    """Все воксели тела должны получить ровно одну метку после правила приоритета."""
    from importlib import import_module
    pp = import_module("03_postprocess")

    shape = (20, 30, 40)
    body = np.ones(shape, dtype=np.uint8)

    tissue_masks = {
        "bone": np.zeros(shape, dtype=np.uint8),
        "lung": np.zeros(shape, dtype=np.uint8),
        "heart": np.zeros(shape, dtype=np.uint8),
        "muscle": np.zeros(shape, dtype=np.uint8),
        "fat_subcutaneous": np.zeros(shape, dtype=np.uint8),
        "skin": np.zeros(shape, dtype=np.uint8),
        "other": np.zeros(shape, dtype=np.uint8),
    }
    # Несколько тканей с частичным перекрытием
    tissue_masks["bone"][:, :, 0:5] = 1
    tissue_masks["lung"][:, :5, :] = 1
    tissue_masks["muscle"][:, 5:, 5:] = 1

    label_map = {
        "bone": 1, "lung": 2, "heart": 3,
        "muscle": 4, "fat_subcutaneous": 5, "skin": 6, "other": 7,
    }
    priority = ["bone", "lung", "heart", "muscle", "fat_subcutaneous", "skin", "other"]

    labelmap = pp.apply_priority_rule(tissue_masks, priority, body, label_map)

    # Все воксели тела должны иметь метку > 0
    uncovered = int((labelmap == 0).sum())
    assert uncovered == 0, f"Непокрытых вокселей: {uncovered}"

    # Нет вокселей за пределами тела с меткой
    outside = int((labelmap[body == 0] > 0).sum())
    assert outside == 0, f"Вокселей вне тела с меткой: {outside}"

    print("  PASS: apply_priority_rule — полное покрытие и нет утечек за тело")


def test_priority_rule_bone_wins():
    """Кость должна иметь приоритет над мышцей при перекрытии."""
    from importlib import import_module
    pp = import_module("03_postprocess")

    shape = (10, 10, 10)
    body = np.ones(shape, dtype=np.uint8)

    tissue_masks = {k: np.zeros(shape, dtype=np.uint8) for k in
                    ["bone", "lung", "heart", "muscle", "fat_subcutaneous", "skin", "other"]}
    # Перекрытие: весь объём — и кость, и мышца
    tissue_masks["bone"][:] = 1
    tissue_masks["muscle"][:] = 1

    label_map = {
        "bone": 1, "lung": 2, "heart": 3,
        "muscle": 4, "fat_subcutaneous": 5, "skin": 6, "other": 7,
    }
    priority = ["bone", "lung", "heart", "muscle", "fat_subcutaneous", "skin", "other"]

    labelmap = pp.apply_priority_rule(tissue_masks, priority, body, label_map)

    # Все воксели должны быть помечены как кость (1), не мышца (4)
    assert (labelmap == 1).all(), "Кость должна побеждать при перекрытии с мышцей"
    print("  PASS: apply_priority_rule — кость имеет приоритет над мышцей")


# ─── Тесты QC ───────────────────────────────────────────────────────────────

def make_valid_labelmap(shape=(20, 30, 40)) -> tuple[np.ndarray, tuple]:
    """Создаёт валидный labelmap для тестирования QC."""
    Z, Y, X = shape
    lm = np.zeros(shape, dtype=np.uint8)

    # Заполним простой анатомически правдоподобный фантом
    lm[:, :, :] = 4          # мышцы (label 4) везде
    lm[:, :, X//2-1:X//2+1] = 1  # кость в центре
    lm[:, :Y//2, :] = 2      # лёгкие слева
    lm[Z//4:3*Z//4, Y//2:, X//4:3*X//4] = 3  # сердце справа

    spacing = (1.5, 1.5, 1.5)
    return lm, spacing


def test_qc_volumes_computed():
    """compute_volumes_ml должна вернуть объёмы для всех тканей."""
    from importlib import import_module
    qc = import_module("04_qc")

    lm, spacing = make_valid_labelmap()
    labels = {
        "1": "bone", "2": "lung", "3": "heart",
        "4": "muscle", "5": "fat_subcutaneous", "6": "skin", "7": "other",
    }

    volumes = qc.compute_volumes_ml(lm, spacing, labels)

    assert set(volumes.keys()) == set(labels.values()), "Не все ткани в результате"
    for tissue, info in volumes.items():
        assert info["volume_ml"] >= 0, f"Отрицательный объём для {tissue}"

    # Мышцы должны быть самой большой тканью в нашем фантоме
    assert volumes["muscle"]["volume_ml"] > volumes["bone"]["volume_ml"]
    print("  PASS: compute_volumes_ml")


def test_qc_coverage_full():
    """Labelmap без пропусков должен давать покрытие = 1.0."""
    from importlib import import_module
    qc = import_module("04_qc")

    shape = (10, 10, 10)
    lm = np.ones(shape, dtype=np.uint8) * 4  # все воксели — мышцы

    coverage = qc.check_coverage(lm, body_threshold=1)
    assert coverage["coverage_ratio"] == 1.0, (
        f"Ожидалось 1.0, получено {coverage['coverage_ratio']}"
    )
    print("  PASS: check_coverage — полное покрытие")


def test_qc_connectivity_single_component():
    """Связный объём без островов не должен давать предупреждений."""
    from importlib import import_module
    qc = import_module("04_qc")

    shape = (10, 10, 10)
    lm = np.zeros(shape, dtype=np.uint8)
    lm[2:8, 2:8, 2:8] = 1  # один монолитный куб

    labels = {"1": "bone"}
    issues = qc.check_connectivity(lm, labels, max_islands=3)
    assert len(issues) == 0, f"Не ожидались предупреждения: {issues}"
    print("  PASS: check_connectivity — один компонент, нет предупреждений")


def test_qc_connectivity_detects_fragments():
    """Два несвязных куска должны быть обнаружены при max_islands=1."""
    from importlib import import_module
    qc = import_module("04_qc")

    shape = (20, 20, 20)
    lm = np.zeros(shape, dtype=np.uint8)
    lm[2:5, 2:5, 2:5] = 1   # первый кусок
    lm[15:18, 15:18, 15:18] = 1  # второй кусок (отдельно)

    labels = {"1": "bone"}
    issues = qc.check_connectivity(lm, labels, max_islands=1)
    assert len(issues) == 1, "Должно быть одно предупреждение о фрагментации"
    assert issues[0]["islands"] == 2
    print("  PASS: check_connectivity — фрагментация обнаружена")


# ─── Запуск ─────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_generate_body_mask,
    test_priority_rule_full_coverage,
    test_priority_rule_bone_wins,
    test_qc_volumes_computed,
    test_qc_coverage_full,
    test_qc_connectivity_single_component,
    test_qc_connectivity_detects_fragments,
]


def run_all():
    """Запускает все тесты и выводит итог."""
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*55}")
    print("Smoke-тесты пайплайна сегментации")
    print(f"{'='*55}")

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            print(f"\n[RUN] {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
            errors.append((name, str(e)))

    print(f"\n{'='*55}")
    print(f"Результат: {passed} PASS, {failed} FAIL")
    if errors:
        print("\nПровалившиеся тесты:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"{'='*55}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
