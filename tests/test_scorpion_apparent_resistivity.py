"""Regression for the Scorpion TXT trailing-tab / phase-column shift."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('uniform_scorpion',Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools/scorpion_apparent_resistivity.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


@pytest.mark.parametrize('trailing_tab',['','\t'])
def test_bei_columns_keep_physical_channel_and_ohm_units(tmp_path,trailing_tab):
    path=tmp_path/'scorpion.txt'
    header='COUNT\tEMG1mV\tEMG2mV\tBEI1ohm_100kHz\tBEI2ohm_100kHz\tBPH1grad_100kHz\tBPH2grad_100kHz\tCE1ohm_100kHz\tCE2ohm_100kHz\tCPH1grad_100kHz\tCPH2grad_100kHz\n'
    rows=['0\t-1,00\t2,00\t79,123450\t45,987650\t-8,50\t-3,20\t600,00\t500,00\t-9,00\t-4,00',
          '1\t-2,00\t3,00\t80,500000\t46,250000\t-8,00\t-3,00\t610,00\t510,00\t-8,00\t-3,00']
    path.write_text(header+'\n'.join(s+trailing_tab for s in rows)+'\n',encoding='utf-8')
    assert worker.read_impedance(path,1).tolist()==[79.12345,80.5]
    assert worker.read_impedance(path,2).tolist()==[45.98765,46.25]


def test_unresolved_physical_channel_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='physical BEI channel'):
        worker.read_impedance(tmp_path/'not_read.txt',3)
