"""Standalone scientific figure for the fixed-rho two-layer/FEM diagnostic."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
here=Path(__file__).resolve().parent
r=json.loads((here/f'compare_fixed_rho_h_{args.tag}.json').read_text(encoding='utf-8'))
x=np.array(r['curves']['L_mm']);fem=np.array(r['curves']['FEM_ohm'])
geom=np.array(r['curves']['geom_ohm']);fit=np.array(r['curves']['fit_ohm'])
loo=np.array([row['prediction_ohm'] for row in r['loo']])
fig,axes=plt.subplots(1,2,figsize=(13,5.5))
colors=['#305aab','#cf791f','#64716f']
axes[0].plot(x,fem,'ko-',label='FEM / CEM',linewidth=2)
for y,c,label in [(geom,colors[0],f"h геометрическое = {r['h_geom_mm']:.2f} мм"),
                  (fit,colors[1],f"общее h по FEM = {r['h_eff']['h_mm']:.2f} мм")]:
    axes[0].plot(x,y,'o-',color=c,label=label)
axes[0].plot(x,loo,'x',color=colors[2],label='прогноз исключённого размера',markersize=7)
for y,c,label in [(geom,colors[0],f"геометрическое h: RMSE {r['rmse_geom_ohm']:.2f} Ом"),
                  (fit,colors[1],f"общее h: RMSE {r['rmse_fit_ohm']:.2f} Ом"),
                  (loo,colors[2],f"исключённые размеры: RMSE {r['rmse_loo_ohm']:.2f} Ом")]:
    axes[1].plot(x,y-fem,'o-',color=c,label=label)
axes[1].axhline(0,color='black',linewidth=.8)
for ax in axes:
    ax.set_xlabel('Номинальный размер сборки L, мм');ax.set_xticks(x);ax.tick_params(axis='x',labelsize=9)
    ax.grid(alpha=.22);ax.legend(fontsize=8.5)
axes[0].set_ylabel('Передаточный импеданс Z, Ом');axes[1].set_ylabel('Z двуслойной модели − Z FEM, Ом')
axes[0].set_title('Сравнение моделей при одних сопротивлениях');axes[1].set_title('Невязка по каждому размеру')
fig.suptitle(f"Одна поза: ρ₁ = {r['rho1_ohm_m']:.3f}, ρ₂ = {r['rho2_ohm_m']:.3f} Ом·м",fontsize=13)
fig.text(.02,.035,'h по FEM — параметр приближения кривой, а не анатомическая толщина.\n'
         'Остаток включает различие точечных и конечных электродов; он не выделяет отдельно влияние кривизны.',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.93])
path=here/f'fixed_rho_h_{args.tag}.png';fig.savefig(path,dpi=150);plt.close(fig);print(path)
