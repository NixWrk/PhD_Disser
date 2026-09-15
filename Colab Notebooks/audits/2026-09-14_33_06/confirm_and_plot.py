"""Confirm the influential-record finding with the original 33.06 optimizer."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exploratory_analysis import fit_static_at_h
from two_layer_model import evaluate, geometry_from_size

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
cfg = json.loads(args.config.read_text(encoding='utf-8'))
art = json.loads((Path(cfg['derived_root'])/'exp02/exploratory/33.06_side_arrays_exploratory.json').read_text(encoding='utf-8'))
loo = json.loads((HERE/'leave_one_out.json').read_text(encoding='utf-8'))
out = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for column, (sid, sub) in enumerate(art['subjects'].items()):
    sizes = np.array(sub['sizes_mm'])/1000
    keep = sizes != .05
    zin, zex = np.array(sub['observed_z_inhale_ohm']), np.array(sub['observed_z_exhale_ohm'])
    rows = []
    for h in [.001, .010]:
        fit = fit_static_at_h(sizes[keep], zin[keep], zex[keep], h)
        fit['held_out_50mm_errors_ohm'] = [evaluate(fit['rho1_ohm_m'], fit[f'rho2_{s}_ohm_m'], h, *geometry_from_size(.05)).z - z
                                            for s, z in [('inhale', zin[0]), ('exhale', zex[0])]]
        rows.append(fit)
        print(sid, 'original fit without 50mm', h, fit['residual_rms_ohm'], flush=True)
    pulse = {}
    for mode, dyn in sub.get('dynamic', {}).items():
        wave = np.array([e['mean_ohm'] for e in dyn['ensembles']])
        residual = np.array(dyn['residual_waveforms_ohm'])
        pulse[mode] = {'observed_rms_ohm': float(np.sqrt(np.mean(wave**2))),
                       'relative_residual_rms': float(np.linalg.norm(residual)/np.linalg.norm(wave)),
                       'fraction_squared_norm_explained': float(1-np.sum(residual**2)/np.sum(wave**2))}
    out[sid] = {'without_50_original_optimizer': rows, 'pulse_fit': pulse}
    label = {'exp02_georg': 'Георг', 'exp02_nik': 'Ник'}[sid]
    ax = axes[0, column]
    best = sub['static_h_profile']['best']
    ax.plot(sizes*1000, zin, 'o', label='Вдох: данные')
    ax.plot(sizes*1000, zex, 's', label='Выдох: данные')
    ax.plot(sizes*1000, best['predicted_inhale_ohm'], '-', label='Вдох: модель, h=1 мм')
    ax.plot(sizes*1000, best['predicted_exhale_ohm'], '--', label='Выдох: модель, h=1 мм')
    ax.set_title(label)
    ax.set_xlabel('Размер сборки, мм')
    ax.set_ylabel('Импеданс, Ом')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[1, column]
    for dropped, style, lab in [(None, '-', 'Все записи'), (50, '--', 'Диагностика без 50 мм')]:
        profile = next(r['profile'] for r in loo[sid] if r['omitted_size_mm'] == dropped)
        ax.plot([p['h_mm'] for p in profile], [p['rms_ohm'] for p in profile], style, label=lab)
    ax.set_xlabel('h, мм')
    ax.set_ylabel('RMS на включённых записях, Ом')
    ax.set_xlim(0, 40)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
fig.suptitle('Аудит 33.06: размерная зависимость и влияние записи 50 мм', fontsize=13)
fig.text(.5, .01, 'Исключение 50 мм — проверка устойчивости, не решение об исключении данных. Сетки h дискретные.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .035, 1, .96))
fig.savefig(HERE/'diagnostics.png', dpi=160)
(HERE/'confirmation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print('Saved confirmation.json and diagnostics.png', flush=True)
