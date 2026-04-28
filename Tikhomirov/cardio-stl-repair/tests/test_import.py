"""Smoke-тесты импорта (Фаза 0, #04)."""

import importlib


def test_top_level_import() -> None:
    mod = importlib.import_module("cardio_stl_repair")
    assert mod.__version__


def test_repair_module_importable() -> None:
    importlib.import_module("cardio_stl_repair.repair")
