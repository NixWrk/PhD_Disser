"""Smoke-тесты импорта (Фаза 0, #04)."""

import importlib


def test_top_level_import() -> None:
    mod = importlib.import_module("cardio_model")
    assert mod.__version__


def test_submodules_importable() -> None:
    """Все основные модули должны импортироваться без ошибок (даже если пустые)."""
    for name in [
        "cardio_model.models",
        "cardio_model.finders",
        "cardio_model.radial",
        "cardio_model.geometry",
        "cardio_model.volume",
        "cardio_model.sphere_fit",
        "cardio_model.sistole",
        "cardio_model.pipeline",
        "cardio_model.dsp",
        "cardio_model.plotting",
        "cardio_model.importers",
        "cardio_model.importers.reo32",
        "cardio_model.importers.comsol",
        "cardio_model.data",
    ]:
        importlib.import_module(name)
