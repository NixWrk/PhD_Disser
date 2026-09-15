"""Invert independently regenerated FEM position and nonlinear pulse probes."""
import numpy as np
import pandas as pd
from halfspace_study import OUT,operators

def run():
    g,ref,sizes,ops,libs=operators(['m3h','m4h'])
    df=pd.read_csv(OUT/'fem_reference.csv');rows=[]
    for scenario,d in df.groupby('scenario'):
        y=d.sort_values('L_mm').Z.to_numpy()
        for name,op in ops.items():
            r,loss=op.fit(y)
            rows.append(dict(scenario=scenario,model=name,rho1=r[0],rho2=r[1],rho1_true=4.,rho2_true=16.,error1_pct=100*(r[0]/4-1),error2_pct=100*(r[1]/16-1),rmse_ohm=loss))
    r=pd.DataFrame(rows);base=r[r.scenario=='baseline'].set_index('model')
    r['change1_pct']=[100*(x.rho1/base.loc[x.model,'rho1']-1) for x in r.itertuples()]
    r['change2_pct']=[100*(x.rho2/base.loc[x.model,'rho2']-1) for x in r.itertuples()]
    r.to_csv(OUT/'placement_inverse.csv',index=False)
    p=pd.read_csv(OUT/'fem_pulse_nonlinearity.csv');rr=[]
    for scenario,d in p.groupby('scenario'):
        d=d.sort_values('L_mm');dy=d.nonlinear_response.to_numpy();dp=d[['delta_rho1_ohm_m','delta_rho2_ohm_m']].iloc[0].to_numpy()
        for name,op in ops.items():
            rhob=base.loc[name,['rho1','rho2']].to_numpy(float);_,J=op.field(rhob)
            est=np.linalg.lstsq(J,dy,rcond=None)[0]
            rr.append(dict(scenario=scenario,model=name,delta1_true=dp[0],delta2_true=dp[1],delta1_est=est[0],delta2_est=est[1],error1_ohm_m=est[0]-dp[0],error2_ohm_m=est[1]-dp[1],error1_pct=None if dp[0]==0 else 100*(est[0]/dp[0]-1),error2_pct=None if dp[1]==0 else 100*(est[1]/dp[1]-1),false_soft_when_absent=bool(dp[0]==0),false_lung_when_absent=bool(dp[1]==0)))
    pd.DataFrame(rr).to_csv(OUT/'pulse_nonlinear_inverse.csv',index=False)

if __name__=='__main__':run()
