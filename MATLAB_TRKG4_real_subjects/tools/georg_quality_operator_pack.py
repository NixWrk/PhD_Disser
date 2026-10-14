"""Fresh operator inputs from immutable coefficient tables; no FEM or fitting."""
from pathlib import Path
import argparse
import shutil
from georg_quality_fractions import plan
from scorpion_georg_fem_prepare import read, write, sha


def prepare(source,out,geometry,meshroot):
    source,out,geometry,meshroot=map(Path,[source,out,geometry,meshroot]);out.mkdir(exist_ok=False)
    points=read(source/'plan.local.json')['numerical_quadrature_points']
    for name in ['mesh.mat','mesh_quality.local.json',*(f'fractions_q{q}.mat' for q in points)]:
        shutil.copy2(source/name,out/name)
    plan(out,geometry,meshroot,points)
    data=read(out/'plan.local.json');data['coefficient_preparation_plan_sha256']=sha(source/'plan.local.json')
    data['coefficient_preparation_directory']=str(source.resolve())
    data['sources'].append(dict(path=str((source/'plan.local.json').resolve()),sha256=sha(source/'plan.local.json')))
    write(out/'plan.local.json',data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','out','geometry','meshroot']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.out,a.geometry,a.meshroot)
