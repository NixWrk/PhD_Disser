import unittest
from copy import deepcopy
from build_channel_assignment_report import hypotheses, values
from scorpion_qc import channel_assignment
from lateral_solution_surfaces import definition


class AssignmentDiagnostics(unittest.TestCase):
    def test_qc_roles_agree_with_active_inverse_and_keep_unknowns(self):
        for item in definition(3):
            roles = channel_assignment(item['record_id'])['physical_channel_roles']
            self.assertTrue(roles[str(item['channel'])].startswith('lateral_'))
        for number in ('001', '002', '003'):
            self.assertEqual(channel_assignment('SCV2-G-'+number)['physical_channel_roles'],
                             {'1':'ttrkg','2':'lateral_horizontal'})
        self.assertEqual(channel_assignment('SCV2-G-006')['status'], 'author_confirmed_2026-10-09')
        self.assertEqual(channel_assignment('SCV2-G-011')['status'], 'late_block_continuity_protocol_assumption')
        self.assertEqual(channel_assignment('SCV2-G-012')['physical_channel_roles'], {'1':'unknown','2':'unknown'})

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
