import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart_cardiac_cycle_report import validate_summary


def complete():
    return {'completed':72,'required':72,'states':[{'id':f'phase_{j:02d}','montage':f'tepc_{m}','Z_ohm':1.0} for j in range(12) for m in range(2,8)]}


def test_unique_coverage_required_even_with_full_count():
    d=complete();validate_summary(d)
    d['states'][-1]=dict(d['states'][0])
    with pytest.raises(ValueError,match='72 unique'):validate_summary(d)


def test_partial_and_nonfinite_are_rejected():
    d=complete();d['completed']=71
    with pytest.raises(ValueError):validate_summary(d)
    d=complete();d['states'][0]['Z_ohm']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):validate_summary(d)
