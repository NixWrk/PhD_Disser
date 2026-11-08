"""Recreate the two summary figures from the tracked, coded aggregate CSV."""
import csv
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    root = Path(__file__).resolve().parent
    with (root / 'summary.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = sorted({(row['series'], row['participant_code']) for row in rows})
    styles = {'': ('o', '#6748a1', 'Боковая сборка МГТУ'),
              '1x1': ('o', '#2563eb', 'ТТРКГ 1×1'),
              '2x2': ('s', '#ac5400', 'ТТРКГ 2×2'),
              '1x4': ('^', '#148260', 'ТТРКГ 1×4')}
    names = {'BMSTU': 'МГТУ', 'Scorpion': '«Скорпион»'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'apparent-resistivity-202611'})
    for pulse in (False, True):
        scale = 1000 if pulse else 1
        upper_key = 'delta_rho_q75_ohm_m' if pulse else 'rho0_ohm_m'
        upper = 1.10 * scale * max(float(row[upper_key]) for row in rows)
        fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.1), sharey=True, sharex=True)
        fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.86, wspace=.25, hspace=.35)
        for series, participant in groups:
            ax = axes[0 if series == 'BMSTU' else 1, int(participant[1:]) - 1]
            group = [row for row in rows if (row['series'], row['participant_code']) == (series, participant)]
            for variant in sorted({row['ttrkg_variant'] for row in group}):
                marker, color, label = styles[variant]
                for i, row in enumerate(r for r in group if r['ttrkg_variant'] == variant):
                    value = float(row['delta_rho_median_ohm_m' if pulse else 'rho0_ohm_m'])
                    error = None
                    if pulse:
                        lo, hi = (float(row[key]) for key in ('delta_rho_q25_ohm_m', 'delta_rho_q75_ohm_m'))
                        error = [[scale * (value - lo)], [scale * (hi - value)]]
                    flagged = pulse and row['weak_qrs_agreement'] == 'true'
                    same_size = sorted(r['ttrkg_variant'] for r in group if r['size_mm'] == row['size_mm'])
                    x = int(row['size_mm']) + 2.5 * (same_size.index(variant) - (len(same_size) - 1) / 2)
                    ax.errorbar(x, scale * value, yerr=error, fmt=marker, capsize=2.5,
                                color=color, markerfacecolor='white' if flagged else color,
                                label=label if i == 0 else None)
            ax.set_title(f'{names[series]} · {participant}')
            ax.set(xlim=(44, 146), ylim=(0, upper), xticks=(50, 80, 110, 140))
            ax.tick_params(labelbottom=True)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ',')))
            ax.grid(alpha=.2)
        axes[0, 2].axis('off')
        legend = [plt.Line2D([], [], marker=m, color=c, ls='none', label=label)
                  for m, c, label in styles.values()]
        if pulse:
            legend.append(plt.Line2D([], [], marker='o', color='#555', markerfacecolor='white',
                                     ls='none', label='F₁ ЭКГ < 0,65'))
        axes[0, 2].legend(handles=legend, loc='center', frameon=False, fontsize=10)
        fig.suptitle('Пульсовое кажущееся Δρ' if pulse else 'Базовое кажущееся ρ', fontsize=17)
        fig.text(.5, .915, 'Задержка глубокого вдоха · индивидуальная КТ · однородная область', ha='center')
        fig.supxlabel('Номинальный размер боковой сборки, мм', y=.09)
        fig.supylabel('Медиана размахов Δρ, 10⁻³ Ом·м' if pulse else 'Кажущееся ρ, Ом·м', x=.025)
        if pulse:
            fig.text(.10, .047, 'Отрезки показывают квартили 25 и 75 % внутри записи. Разметка «Скорпиона» кандидатная.', fontsize=9)
        fig.text(.10, .023, 'При одном номинальном размере маркеры разных монтажей разнесены по горизонтали.', fontsize=9)
        buffer = StringIO()
        fig.savefig(buffer, format='svg', metadata={'Date': None})
        target = root / ('pulse.svg' if pulse else 'baseline.svg')
        target.write_text('\n'.join(line.rstrip() for line in buffer.getvalue().splitlines()) + '\n',
                          encoding='utf-8', newline='\n')
        plt.close(fig)


if __name__ == '__main__':
    main()
