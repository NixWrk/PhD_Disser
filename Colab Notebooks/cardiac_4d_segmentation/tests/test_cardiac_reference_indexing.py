"""Mathematical and provenance safeguards for BSA indexing."""
from copy import deepcopy
import pytest

from cardiac_reference_indexing import mosteller_bsa, index_estimates, REFERENCE


def estimates():
    return [{"subject": "sample", "scenario": "raw", "chamber_label": chamber,
             "estimator": estimator, "edv_ml": edv, "esv_ml": esv,
             "sv_ml": edv-esv, "ef_percent": 100*(edv-esv)/edv}
            for chamber, edv, esv in [("ЛЖ", 150., 90.), ("ПЖ", 240., 130.)]
            for estimator in ("observed_global_range", "periodic_composite")]


def person(**updates):
    value = dict(height_cm=176, weight_kg=80, sex="male", age_years=26)
    value.update(updates)
    return {"sample": value}


def test_formula_and_invariants_without_mutation():
    raw = estimates()
    original = deepcopy(raw)
    bsa = mosteller_bsa(176, 80)
    assert bsa == pytest.approx(1.9776529298921768)
    records, status = index_estimates(raw, person())
    assert status[0]["status"] == "indexed"
    assert len(records) == 16 and raw == original
    for chamber in ("ЛЖ", "ПЖ"):
        for estimator in ("observed_global_range", "periodic_composite"):
            values = {r["metric"]: r for r in records if r["chamber"] == chamber and r["estimator"] == estimator}
            v = {k: r["value"] for k, r in values.items()}
            assert v["edv_ml"] - v["esv_ml"] == pytest.approx(v["sv_ml"])
            assert 100*v["sv_ml"]/v["edv_ml"] == pytest.approx(v["ef_percent"])
            for metric, row in values.items():
                assert row["value"]*(1 if metric == "ef_percent" else bsa) == pytest.approx(row["raw_value"])
    sv = {r["chamber"]: r for r in records if r["estimator"] == "periodic_composite" and r["metric"] == "sv_ml"}
    assert sv["ПЖ"]["value"]/sv["ЛЖ"]["value"] == pytest.approx(110/60)
    assert (sv["ПЖ"]["value"]-sv["ЛЖ"]["value"])*bsa == pytest.approx(50)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "176"])
def test_invalid_anthropometry_fails(bad):
    with pytest.raises(ValueError):
        mosteller_bsa(bad, 80)
    with pytest.raises(ValueError):
        mosteller_bsa(176, bad)


def test_missing_inputs_do_not_borrow_other_subject():
    raw = estimates()
    raw += [dict(raw[0], subject="other")]
    records, status = index_estimates(raw, person())
    assert {r["subject"] for r in records} == {"sample"}
    assert next(s for s in status if s["subject"] == "other")["status"] == "missing_anthropometry"
    assert index_estimates(raw, {})[0] == []
    assert index_estimates(estimates(), person(height_cm=None))[0] == []


@pytest.mark.parametrize("age,sex,matched", [(20, "male", True), (29.9, "male", True),
    (19, "male", False), (30, "male", False), (26, "female", False), (None, "male", False)])
def test_reference_selection_does_not_change_index(age, sex, matched):
    records, _ = index_estimates(estimates(), person(age_years=age, sex=sex))
    assert len(records) == 16
    assert all(("reference_low" in r) == matched for r in records)
    assert records[0]["value"] == pytest.approx(150/mosteller_bsa(176, 80))


def test_published_rounding_and_contour_conventions_are_preserved():
    # 86±2*13 would be 60–112: use the published 61–112 instead.
    assert REFERENCE["ЛЖ"]["metrics"]["edv_ml"][2:4] == (61, 112)
    assert REFERENCE["ПЖ"]["metrics"]["ef_percent"][2:4] == (36, 69)
    assert REFERENCE["ЛЖ"]["papillary_convention"] != REFERENCE["ПЖ"]["papillary_convention"]
    raw = estimates()
    raw[0]["edv_ml"] = 61*mosteller_bsa(176, 80)
    records, _ = index_estimates(raw, person())
    assert records[0]["reference_position"] == "within"
    raw[0]["edv_ml"] -= 0.01
    records, _ = index_estimates(raw, person())
    assert records[0]["reference_position"] == "below"


def test_zero_stroke_and_ef_are_not_dropped():
    row = dict(estimates()[0], edv_ml=90., esv_ml=90., sv_ml=0., ef_percent=0.)
    records, _ = index_estimates([row], person())
    assert len(records) == 4
    assert [r["value"] for r in records][-2:] == [0., 0.]


def test_duplicate_source_estimate_cannot_silently_overwrite():
    raw = estimates()
    with pytest.raises(ValueError, match="Duplicate"):
        index_estimates(raw+[raw[0]], person())
