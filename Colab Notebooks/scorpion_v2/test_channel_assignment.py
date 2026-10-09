import unittest
from copy import deepcopy
from build_channel_assignment_report import hypotheses, values


class AssignmentDiagnostics(unittest.TestCase):
    def test_switch_restores_synthetic_constant_reference_without_relabeling_input(self):
        records=[]
        for rid,lateral in zip(['004','007','008','009','010'],[80.,90.,100.,110.,120.]):
            channels=[]
            for ch in (1,2):
                z=lateral if ch==(2 if rid=='004' else 1) else 65.
                part={'stats':{'median':z}}
                channels.append({'channel':ch,'phases':[{'mode':m,'analysis':part,'full':part} for m in ('inspiration_hold','expiration_hold')]})
            records.append({'record_id':'SCV2-G-'+rid,'channels':channels})
        original=deepcopy(records)
        cases=hypotheses(records)
        switched=next(c for c in cases if (c['early'],c['late'],c['mode'])==(2,1,'inspiration_hold'))
        self.assertEqual(switched['ttrkg_range_ohm'],0.)
        self.assertTrue(switched['author_consistent'])
        self.assertFalse(switched['constant_number'])
        self.assertTrue(all(c['ttrkg_range_ohm']>0 for c in cases if c['early']==c['late']))
        self.assertEqual(records,original)

    def test_unavailable_stage_is_not_replaced_by_zero(self):
        records=[{'record_id':'SCV2-G-001','channels':[{'channel':1,'phases':[{'mode':'calm_breathing','analysis':None,'full':{'stats':{'median':70.}}}]}]}]
        self.assertEqual(values(records,'001',1),{})
        self.assertEqual(values(records,'001',1,'full'),{'calm_breathing':70.})


if __name__=='__main__':unittest.main()
