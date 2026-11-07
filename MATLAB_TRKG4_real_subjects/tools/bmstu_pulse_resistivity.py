"""Пульсовое кажущееся rho: только канал 2 реокардиомонитора МГТУ.

Повторяет принятые R-метки и правила 12.01/33.03. Не фильтрует RHEO,
не меняет разметку и не подбирает приборные коэффициенты. Первичные данные
доступны для чтения; все создаваемые материалы остаются в новом каталоге.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

from bmstu_apparent_resistivity import ROOT, read, sha, write_new

PHASE = np.linspace(0.0, 1.0, 200, endpoint=False)
STATUS = 'exploratory_hypothesis_not_validated'


def normalized_cycle(time, signal, start, stop):
    """Та же сетка R–R, что в 33.03; постоянная медиана не меняет размах."""
    time = np.asarray(time, float); signal = np.asarray(signal, float)
    if time.ndim != 1 or signal.shape != time.shape or len(time) < 2:
        raise ValueError('Время и сигнал должны иметь одинаковую одномерную форму')
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError('Некорректное время')
    if not time[0] <= start < stop <= time[-1]:
        raise ValueError('Цикл выходит за пределы записи')
    y = np.interp(start + PHASE * (stop-start), time, signal)
    if not np.isfinite(y).all():
        raise ValueError('Нечисловой пульсовой сигнал')
    return y - np.median(y)


def pulse_statistics(waves_mohm, flags, rho0, slope_logrho):
    """Медиана индивидуальных размахов; отдельно размах медианной формы."""
    waves = np.asarray(waves_mohm, float); flags = np.asarray(flags, bool)
    if waves.ndim != 2 or waves.shape[0] < 3 or waves.shape[1] != len(PHASE):
        raise ValueError('Нужны хотя бы три полных цикла на сетке 33.03')
    if flags.shape != (len(waves),) or not np.isfinite(waves).all():
        raise ValueError('Неверные метки качества или нечисловые циклы')
    if not np.isfinite([rho0, slope_logrho]).all() or min(rho0, slope_logrho) <= 0:
        raise ValueError('Нужны положительные базовое rho и чувствительность')
    factor = rho0 / slope_logrho * 1e-3  # (Ohm m) per mOhm
    rho = waves * factor
    amplitudes = np.ptp(rho, axis=1)
    med = np.median(rho, axis=0)
    q25, q75 = np.quantile(rho, [.25, .75], axis=0)
    result = dict(n_cycles=len(waves), n_candidate_ecg_flags=int(flags.sum()),
        conversion_ohm_m_per_mohm=float(factor),
        delta_rho_median_ohm_m=float(np.median(amplitudes)),
        delta_rho_q25_ohm_m=float(np.quantile(amplitudes, .25)),
        delta_rho_q75_ohm_m=float(np.quantile(amplitudes, .75)),
        delta_Z_median_ohm=float(np.median(np.ptp(waves, axis=1)) * 1e-3),
        ensemble_peak_to_peak_ohm_m=float(np.ptp(med)),
        max_abs_delta_Z_ohm=float(np.max(np.abs(waves)) * 1e-3),
        median_wave_ohm_m=med.tolist(), q25_wave_ohm_m=q25.tolist(), q75_wave_ohm_m=q75.tolist(),
        cycle_peak_to_peak_ohm_m=amplitudes.tolist())
    if (~flags).sum() >= 3:
        alt = rho[~flags]; alt_med = np.median(alt, axis=0)
        result['without_candidate_ecg_flags'] = dict(n_cycles=len(alt),
            delta_rho_median_ohm_m=float(np.median(np.ptp(alt, axis=1))),
            delta_rho_q25_ohm_m=float(np.quantile(np.ptp(alt, axis=1), .25)),
            delta_rho_q75_ohm_m=float(np.quantile(np.ptp(alt, axis=1), .75)),
            ensemble_peak_to_peak_ohm_m=float(np.ptp(alt_med)),
            max_wave_difference_ohm_m=float(np.max(np.abs(med-alt_med))),
            median_wave_ohm_m=alt_med.tolist())
    else:
        result['without_candidate_ecg_flags'] = None
    return result


def collect(observations, baseline_paths, derived_root, out):
    observations = Path(observations).resolve(); derived_root = Path(derived_root).resolve()
    out = Path(out).resolve()
    data = read(observations)
    if data['experiment_id'] != 'exp02' or data['instrument'] != 'BMSTU_reocardiomonitor':
        raise ValueError('Разрешена только серия реокардиомонитора МГТУ')
    if [s['subject_alias'] for s in data['subjects']] != ['georg', 'nik']:
        raise ValueError('Разрешены только Георгий и Ник')
    qc_path = derived_root/'exp02/qc/12.01_signal_qc.candidate.json'; qc = read(qc_path)
    reference_path = derived_root/'exp02/exploratory/33.03_pulse_ensembles.provisional.json'
    reference = read(reference_path)
    if qc['artifact_type'] != 'post_annotation_signal_qc' or qc['experiment_id'] != 'exp02':
        raise ValueError('Чужой QC')
    if sha(qc_path) != reference['upstream']['12.01_sha256']:
        raise ValueError('12.01 изменён относительно сохранённого 33.03')
    if reference['author_decision']['status'] != 'provisionally_accepted_for_method_development':
        raise ValueError('Нет сохранённого решения о применении правил отбора циклов')
    qc_records = {r['record_id']:r for r in qc['records']}
    old_waves = {(r['record_id'], r['mode']):r for r in reference['ensembles']}
    baselines = {read(p)['subject_alias']:(Path(p).resolve(), read(p)) for p in baseline_paths}
    if set(baselines) != {'georg','nik'}:
        raise ValueError('Нужны оба индивидуальных базовых расчёта')
    subjects = []; hashes = {observations:sha(observations), qc_path:sha(qc_path), reference_path:sha(reference_path)}
    for subject in data['subjects']:
        alias = subject['subject_alias']; bp, baseline = baselines[alias]
        request_path = bp.parent.parent/(bp.parent.name+'_input.local.json'); request = read(request_path)
        if (sha(request_path) != baseline['identity']['geometry_plan_sha256'] or
                request['observations_plan_sha256'] != sha(observations)):
            raise ValueError('Не совпадает происхождение базового FEM-расчёта')
        if (baseline['subject_alias'] != alias or baseline['TTRKG_coupling_modelled'] or
                not baseline['four_lateral_contacts_only']):
            raise ValueError('Другая базовая электрическая постановка')
        for path, digest in ((request['mesh_path'], request['mesh_sha256']),
                             (request['geometry_source_path'], request['geometry_source_sha256'])):
            if sha(path) != digest:
                raise ValueError('Изменилась индивидуальная геометрия')
            hashes[Path(path)] = digest
        hashes[bp] = sha(bp); hashes[request_path] = sha(request_path)
        answers = {r['record_id']:r for r in baseline['answers']}; records = []
        for o in subject['observations']:
            source = Path(o['source_path']); breathing_path = Path(o['annotation_path'])
            ecg_path = derived_root/'exp02/annotations/ecg'/(o['source_record_id']+'.json')
            breathing = read(breathing_path); ecg = read(ecg_path)
            for path, digest in ((source,o['source_sha256']), (breathing_path,o['annotation_sha256'])):
                if sha(path) != digest:
                    raise ValueError('Изменены исходная запись или дыхательная разметка')
                hashes[path] = digest
            hashes[ecg_path] = sha(ecg_path)
            for annotation in (breathing, ecg):
                if (annotation['qc']['status'] != 'accepted' or
                    annotation['input']['sha256'] != o['source_sha256'] or
                    annotation['subject_id'] != 'exp02_'+alias or annotation['size_mm'] != o['size_mm']):
                    raise ValueError('Непринятая либо чужая разметка')
            peaks = np.asarray(ecg['rpeaks_s'], float)
            if not np.isfinite(peaks).all() or not np.all(np.diff(peaks)>0):
                raise ValueError('Некорректные принятые R-метки')
            old = old_waves[o['source_record_id'], 'задержка_вдох']
            if (old['input_provenance']['ecg_sidecar_sha256'] != hashes[ecg_path] or
                old['input_provenance']['breathing_sidecar_sha256'] != hashes[breathing_path] or
                old['input_provenance']['input_sha256'] != hashes[source]):
                raise ValueError('33.03 построен по другой версии записи или разметки')
            qr = qc_records[o['source_record_id']]
            if not qr['component_use']['RHEO2'] or qr['input_sha256'] != hashes[source]:
                raise ValueError('RHEO2 не разрешён поканальным QC')
            frame = pd.read_csv(source, usecols=['TIME_s','BASE_2_Ω','RHEO_2_mΩ'])
            time = frame['TIME_s'].to_numpy(float); signal = frame['RHEO_2_mΩ'].to_numpy(float)
            base = frame['BASE_2_Ω'].to_numpy(float)
            if not np.isfinite(signal).all():
                raise ValueError('Нечисловой RHEO2')
            left, right = o['selected_window_s']; mask = (time>=left)&(time<=right)
            if abs(np.median(base[mask])-o['Z_ohm']) > 1e-8:
                raise ValueError('Не воспроизведён базовый уровень того же окна')
            candidates = [c for c in qr['cycles'] if c['mode']=='задержка_вдох']
            chosen = [c for c in candidates if c['eligible_without_plateau']]
            if len(chosen) != len(old['cycles']):
                raise ValueError('Изменился состав циклов относительно 33.03')
            cycles = []; waves = []; flags = []; max_reproduction = 0.0
            for c, previous in zip(chosen, old['cycles']):
                start, stop = c['start_r_s'], c['stop_r_s']
                j = np.flatnonzero(np.isclose(peaks,start,rtol=0,atol=1e-9))
                if (len(j)!=1 or j[0]+1>=len(peaks) or abs(peaks[j[0]+1]-stop)>1e-9 or
                        start<left-1e-9 or stop>right+1e-9 or c['extreme_plateau_overlap']):
                    raise ValueError('Цикл не соответствует принятым R-меткам, окну или QC')
                if start != previous['start_r_s'] or stop != previous['stop_r_s']:
                    raise ValueError('Различаются границы циклов 12.01 и 33.03')
                wave = normalized_cycle(time,signal,start,stop)
                error = float(np.max(np.abs(wave-np.asarray(previous['centered_native_mohm']))))
                max_reproduction = max(max_reproduction,error)
                if error > 1e-9:
                    raise ValueError('Не воспроизведена знаковая форма цикла 33.03')
                flagged = bool(c['baseline_artifact_candidate_overlap'])
                raw = signal[(time>=start)&(time<stop)]
                cycles.append(dict(start_r_s=start, stop_r_s=stop, rr_s=stop-start,
                    candidate_ecg_baseline_flag=flagged, native_sample_peak_to_peak_mohm=float(np.ptp(raw)),
                    normalized_peak_to_peak_mohm=float(np.ptp(wave)),centered_native_mohm=wave.tolist()))
                waves.append(wave); flags.append(flagged)
            a = answers[o['record_id']]
            if (a['observed_Z_ohm'] != o['Z_ohm'] or a['size_mm'] != o['size_mm'] or
                    abs(a['residual_ohm']) > a['tolerance_ohm']):
                raise ValueError('Чужой или не сошедшийся базовый результат')
            statistics = pulse_statistics(waves,flags,a['rho_apparent_ohm_m'],a['dZ_dlogrho_ohm'])
            reproduction = float(np.max(np.abs(np.median(waves,axis=0)-old['primary']['median_mohm'])))
            if reproduction > 1e-9:
                raise ValueError('Не воспроизведён медианный ансамбль 33.03')
            records.append(dict(record_id=o['record_id'],source_record_id=o['source_record_id'],
                subject_alias=alias,size_mm=o['size_mm'],source_path=str(source),source_sha256=hashes[source],
                ecg_path=str(ecg_path),ecg_sha256=hashes[ecg_path],breathing_path=str(breathing_path),
                breathing_sha256=hashes[breathing_path],window_s=[left,right],Z0_ohm=o['Z_ohm'],
                rho0_ohm_m=a['rho_apparent_ohm_m'],dZ_dlogrho_ohm=a['dZ_dlogrho_ohm'],
                n_qc_inhale_cycles=len(candidates),n_excluded_plateau=len(candidates)-len(chosen),
                rr_median_s=float(np.median([c['rr_s'] for c in cycles])),
                rr_range_s=[min(c['rr_s'] for c in cycles),max(c['rr_s'] for c in cycles)],
                reproduction_33_03_max_error_mohm=max(max_reproduction,reproduction),
                **statistics,cycles=cycles))
            print(o['record_id'],f"N={len(cycles)} flags={sum(flags)} delta_rho={statistics['delta_rho_median_ohm_m']:.8g} Ohm m",flush=True)
        subjects.append(dict(subject_alias=alias,baseline_path=str(bp),baseline_sha256=sha(bp),
            geometry_path=request['geometry_source_path'],geometry_sha256=request['geometry_source_sha256'],
            placement=baseline['placement'],records=records))
    all_records = [r for s in subjects for r in s['records']]
    if len(all_records)!=19 or len({r['source_sha256'] for r in all_records})!=19:
        raise ValueError('Нужно 19 независимых записей без дубликата Ника 100 мм')
    payload = dict(schema='bmstu_uniform_pulse_v1',status=STATUS,experiment_id='exp02',channel=2,
        instrument='BMSTU_reocardiomonitor',signal_column='RHEO_2_mΩ',unit_scale_to_ohm=1e-3,
        additional_gain=1,sign_multiplier=1,additional_filtering=False,
        measured_impedance_interpretation='declared_mOhm_as_delta_Z_working_assumption',
        phase_rr=PHASE.tolist(),phase_endpoint_included=False,cycle_center='per_cycle_median',
        primary_amplitude='median_of_individual_cycle_peak_to_peak_on_200_point_RR_grid',
        candidate_ECG_flags_excluded_from_primary=False,
        QC_status=qc['status'],provisional_cycle_policy=reference['author_decision'],
        subjects=subjects,excluded=data['excluded'],
        protected_inputs=[dict(path=str(p),sha256=h) for p,h in hashes.items()],producer_sha256=sha(__file__))
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'pulse.local.json',payload)
    (out/'pulse_worker_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return out/'pulse.local.json'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--baselines',type=Path,nargs=2,required=True)
    p.add_argument('--derived-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(collect(a.observations,a.baselines,a.derived_root,a.out),flush=True)
