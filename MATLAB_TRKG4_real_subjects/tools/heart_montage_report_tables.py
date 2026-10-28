"""Reproducible report tables from completed, audited montage calculations."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np

from heart_montage_study_prepare import read, sha
from heart_shape_electrical_comparison import write_json, write_csv


SHAPES = ("STL", "G1", "L1")
SCENARIOS = ("diagnostic4", "commonSTLfit")
PRIMARY = "top_gap100"


def select(rows, **conditions):
    return [r for r in rows if all(str(r[k]) == str(v) for k,v in conditions.items())]


def one(rows, **conditions):
    rr=select(rows,**conditions)
    if len(rr)!=1: raise ValueError(f"Expected one row: {conditions}, got {len(rr)}")
    return rr[0]


def fmt(v, digits=4):
    if v is None: return "—"
    if isinstance(v, str): return v
    return f"{v:.{digits}g}".replace(".",",")


def main(analysis, compensation, newton, refinement, output):
    a,c,n,r,out=map(Path,(analysis,compensation,newton,refinement,output))
    out.mkdir(parents=True,exist_ok=True)
    inputs=[]
    def csvread(path):
        inputs.append(path)
        with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
    ap=a/"analysis.local.json";dp=a/"dataset.local.json";inputs.extend([ap,dp])
    if read(ap)["states"]!=720:raise ValueError("Require complete primary analysis")
    groups=read(dp)["groups"]
    gd={(d["dataset"]["context"]["shape"],d["dataset"]["context"]["scenario"],d["dataset"]["context"]["quadrature"]):d for d in groups}
    omission=csvread(a/"omission_units.csv");budgets=csvread(a/"exclusion_error_budgets.csv")
    joints=csvread(a/"joint_decomposition.csv");rho_checks=csvread(a/"rho_checks.csv")
    comp=csvread(c/"finite_compensation.csv");refined=csvread(n/"finite_compensation.csv")
    qref=csvread(r/"volume_refinement.csv")
    if len(comp)!=36 or len(refined)!=36 or len(qref)!=12 or len(groups)!=12:
        raise ValueError("Supplementary tables/groups are incomplete")
    sections=[]; numeric={}; summary={}
    def add(key,number,title,headers,rows):
        numeric[key]=rows
        text=["<details>",f"<summary>Таблица 18.{number}. {title}</summary>","",
              "| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]
        text.extend("| "+" | ".join(fmt(v) for v in row)+" |" for row in rows)
        text.extend(["","</details>",""])
        sections.append("\n".join(text))
    rows=[]
    for s in SCENARIOS:
        for h in SHAPES:
            d=gd[h,s,64];fine=gd[h,s,256];m=d["dataset"]["montage_ids"].index(PRIMARY)
            values={x["step_mm"]:1000*x["J"][m][0] for x in d["jacobians"]}
            vf=1000*fine["jacobians"][0]["J"][m][0]
            ve=None if h!="L1" else 1000*float(one(qref,shape=h,scenario=s,montage=PRIMARY,quadrature=1024)["K_V_ohm_per_ml"])
            rows.append([s,h,values[5.],values[2.5],values[1.],vf,ve])
    add("volume_derivative",2,"Объёмная чувствительность исходного монтажа, мОм/мл",
        ["Опора","Форма","q64: 5%","q64: 2,5%","q64: 1%","q256: 1%","q1024: 1%"],rows)
    rows=[]
    for s in SCENARIOS:
        for h in SHAPES:
            rr=select(omission,shape=h,scenario=s,montage=PRIMARY,quadrature=256)
            rows.append([s,h,*[float(one(rr,parameter=p)["equivalent_volume_at_diagnostic_amplitude"])
                               for p in ("x","y","z","rho_heart","rho_soft","rho_lung")]])
    add("isolated_errors",3,"Линейная эквивалентная ошибка объёма при исключении одного вклада, мл",
        ["Опора","Форма","X: 1 мм","Y: 1 мм","Z: 1 мм","ρH: 1%","ρS: 1%","ρL: 1%"],rows)
    rows=[]
    for s in SCENARIOS:
        for h in SHAPES:
            rr=select(budgets,shape=h,scenario=s,montage=PRIMARY,quadrature=256,illustrative_budget_ml="5.0")
            if not rr: rr=select(budgets,shape=h,scenario=s,montage=PRIMARY,quadrature=256,illustrative_budget_ml="5")
            values=[]
            for p in ("x","y","z","rho_heart","rho_soft","rho_lung"):
                cell=one(rr,parameter=p)
                field="isolated_abs_parameter_change" if p in "xyz" else "isolated_abs_rho_change_percent"
                values.append(float(cell[field]) if cell[field] else None)
            rows.append([s,h,*values])
    add("five_ml_budgets",4,"Изолированные ограничения для иллюстративного бюджета 5 мл",
        ["Опора","Форма","|δX|, мм","|δY|, мм","|δZ|, мм","|δρH|, %","|δρS|, %","|δρL|, %"],rows)
    rows=[]
    for s in SCENARIOS:
        for h in SHAPES:
            d=gd[h,s,256];j=np.asarray(d["jacobians"][0]["J"]);scale=np.array(d["dataset"]["amplitudes"])
            full=j[:,:5]*scale[:5];sv=np.linalg.svd(full,compute_uv=False)
            reduced=full[:,[0,4]];sr=np.linalg.svd(reduced,compute_uv=False)
            from heart_montage_analysis import volume_identified_given_nuisance
            test=volume_identified_given_nuisance(j[:,:5],scale[:5])
            rows.append([s,h,test["rank"],"да" if test["volume_identified"] else "нет",sv[-1]/sv[0],sr[0]/sr[-1]])
    add("identifiability",5,"Матрица трёх альтернативных монтажей при q256 и малом шаге",
        ["Опора","Форма","Ранг: 5 параметров","V определим при свободных XYZρH","s₃/s₁","κ для V и ρH при известных XYZ"],rows)
    rows=[]
    for s in SCENARIOS:
        for h in SHAPES:
            for sign in (-1.,1.):
                cell=one(joints,shape=h,scenario=s,montage=PRIMARY,sign=sign)
                rows.append([s,h,int(sign),1000*float(cell["sum_single_ohm"]),1000*float(cell["joint_delta_ohm"]),
                             100*float(cell["cancellation_fraction"]),1000*float(cell["aligned_interaction_ohm"])])
    add("joint",6,"Совместные изменения и компенсация отдельных вкладов, исходный монтаж",
        ["Опора","Форма","Знак","Σ отдельных ΔZ, мОм","Совместный ΔZ, мОм","Погашение суммы, %","Rвз, мОм"],rows)
    rows=[]
    for s in SCENARIOS:
        for h in SHAPES:
            old=select(comp,scenario=s,shape=h,montage=PRIMARY);new=select(refined,scenario=s,shape=h,montage=PRIMARY)
            others=[x for x in select(refined,scenario=s,shape=h) if x['montage']!=PRIMARY]
            maximum=lambda seq,key: max(abs(float(x[key])) for x in seq)
            rows.append([s,h,maximum(new,"delta_volume_ml"),100*maximum(new,"relative_delta_rho_heart"),
                1000*maximum(old,"compensated_delta_ohm"),1000*maximum(new,"compensated_delta_ohm"),
                1000*maximum(others,"compensated_delta_ohm")])
    add("finite_compensation",7,"Максимальные модули по двум знакам объёмной пробы",
        ["Опора","Форма","|ΔV|, мл","|ΔρH| после уточнения, %","Остаток до Ньютона, мОм","После: исходный, мОм","После: другие монтажи, мОм"],rows)
    cross=[];rows=[]
    for s in SCENARIOS:
        original=gd["STL",s,256]["dataset"]
        base=next(r for r in original["records"] if r["probe"]["kind"]=="base")
        for h in SHAPES:
            errors=[]
            j=np.asarray(gd[h,s,256]["jacobians"][0]["J"])
            for trial in sorted(original["records"],key=lambda x:x["probe"]["signed_step"]):
                if trial["probe"]["kind"]!="volume":continue
                dz=np.asarray(trial["impedance_ohm"])-base["impedance_ohm"]
                dv=trial["volume_ml"]-base["volume_ml"]
                for i,m in enumerate(original["montage_ids"]):
                    estimate=dz[i]/j[i,0]
                    cross.append(dict(scenario=s,form=h,montage=m,probe=trial["id"],true_delta_volume_ml=dv,
                                      estimate_ml=estimate,error_ml=estimate-dv,scope="local_volume_only_with_exact_other_parameters"))
                    if m==PRIMARY:errors.append(estimate-dv)
            rows.append([s,h,*errors,max(abs(x) for x in errors)])
    add("anatomy_volume_transfer",8,"Локальное восстановление объёмных проб STL по чувствительности выбранной формы",
        ["Опора","Форма коэффициента","Ошибка для −1%, мл","Ошибка для +1%, мл","max |ошибка|, мл"],rows)
    write_csv(out/"anatomy_volume_transfer.csv",cross)
    # Explicit checks and ranges; numerical approximation changes are not noise.
    summary["analytic_rho_finite_relative_error_max"]={str(step):max(abs(float(x['signed_error'])/float(x['analytic']))
        for x in rho_checks if float(x['rho_step_percent'])==step and float(x['analytic'])!=0) for step in (1.,5.)}
    summary["newton_target_max_absolute_residual_ohm"]=max(abs(float(x['compensated_delta_ohm'])) for x in refined if x['montage']==PRIMARY)
    summary["newton_other_max_absolute_residual_ohm"]=max(abs(float(x['compensated_delta_ohm'])) for x in refined if x['montage']!=PRIMARY)
    summary["newton_outside_prior_five_percent_rho_states"]=sum(x['outside_previously_probed_rho_fraction']=='True' for x in refined)
    with (out/"tables.md").open('x',encoding='utf-8') as f:f.write('\n'.join(sections))
    inputs.append(Path(__file__))
    write_json(out/"summary.local.json",dict(tables=numeric,summary=summary,source_sha256={str(p.resolve()):sha(p) for p in inputs},
        units_and_conditions="Tables are conditional computational controls; no clinical or noise accuracy",states_primary=720))
    print('Prepared',len(sections),'report tables')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','compensation','newton','refinement','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.analysis,a.compensation,a.newton,a.refinement,a.output)
