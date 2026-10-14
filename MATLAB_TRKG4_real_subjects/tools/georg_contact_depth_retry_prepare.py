"""Retry the same final-pose refinement with a larger bounded promotion count.

Keeps the 1M-node cap, q>=0.01 and all geometry/material rules. No FEM solve.
"""
from pathlib import Path
import argparse
from georg_quality_contact_depth_rebuild import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','parent','out','parent-forward','ray-profile'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--promotion-rounds',type=int,default=6)
    a=p.parse_args()
    if not 2<a.promotion_rounds<=6:raise ValueError('Retry promotion count must be 3..6')
    prepare(a.candidate,a.parent,a.out,promotion_rounds=a.promotion_rounds,max_promoted=10000,
            parent_forward_file=a.parent_forward,ray_profile_file=a.ray_profile)
