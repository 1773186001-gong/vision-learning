"""Necessary artificial checks only: thresholds, ties, layers, risk and count units."""
import unittest
from nms_relation_core import relationship,build_queue,pair_summary,model_summary

STAGES=('decoded','pre_nms_topn','clipped','small_box_filtered','score_filtered','layer_nms','post_nms_topn')
def sample_rows():
    # Four GTs: common risk (a,c), IR-only risk (b), neither risk (d).
    rows=[]
    for stem,ip,fp,il,fl in (('a',True,True,False,True),('b',True,False,True,False),('c',True,True,True,True),('d',False,False,False,False)):
        for model,pre,loss in (('IR',ip,il),('Fusion',fp,fl)):
            for s in STAGES:
                covered=pre and (not loss or s not in ('layer_nms','post_nms_topn'))
                rows.append({'model':model,'stem':stem,'gt_index':'0','scene_id':'03','image_index':'0','gt_xyxy':'[0,0,10,10]',
                    'stage':s,'covered_075_float32_cpu':str(covered),'max_iou_float32_cpu':str(.8 if covered else .4),
                    'count_iou_ge_075_float32_cpu':str(int(covered))})
    return rows

class NecessaryTests(unittest.TestCase):
    def test_threshold_is_strict_not_greater_equal(self):
        r=relationship(True,.5,.6,.7,.7,.7)
        self.assertFalse(r['primary_possible_relation']);self.assertFalse(r['diagnostic_only_relation'])
        self.assertTrue(relationship(True,.5,.6,.700001,.700001,.700001)['primary_possible_relation'])
    def test_near_flag_does_not_relax_threshold(self):
        r=relationship(True,.5,.6,.6999998,.6999998,.6999998)
        self.assertTrue(r['near_threshold_diagnostic_only']);self.assertFalse(r['primary_possible_relation'])
    def test_exact_score_tie_separate_from_higher_and_lower(self):
        r=relationship(True,.8,.8,.9,.9,.9)
        self.assertTrue(r['score_tie']);self.assertEqual(r['score_relation'],'equal');self.assertTrue(r['primary_possible_relation'])
        r=relationship(True,.8,.7999999,.9,.9,.9)
        self.assertFalse(r['primary_possible_relation']);self.assertFalse(r['score_tie'])
    def test_cross_layer_overlap_is_not_relation_or_numerical_disagreement(self):
        r=relationship(False,.5,.6,.95,.95,0.)
        self.assertFalse(r['primary_possible_relation']);self.assertFalse(r['diagnostic_only_relation'])
        self.assertFalse(r['relation_predicate_disagreement'])
    def test_float64_only_does_not_enter_primary(self):
        r=relationship(True,.5,.6,.7,.70000001,.7)
        self.assertFalse(r['primary_possible_relation']);self.assertTrue(r['diagnostic_only_relation']);self.assertTrue(r['relation_predicate_disagreement'])
    def test_offset_diagnostic_does_not_enter_primary(self):
        r=relationship(True,.5,.6,.69999,.69999,.70001)
        self.assertFalse(r['primary_possible_relation']);self.assertTrue(r['diagnostic_only_relation'])
    def test_shared_risk_and_not_at_risk_are_distinct(self):
        events,paired=build_queue(sample_rows())
        self.assertEqual(len(events),4)
        row=next(p for p in paired if p['stem']=='b');self.assertEqual(row['Fusion_risk_state'],'not_at_risk')
        common=next(r for r in pair_summary(paired) if r['scene_id']=='ALL' and r['cohort']=='both_precovered')
        self.assertEqual((common['GT_denominator'],common['IR_lost_gt'],common['Fusion_lost_gt']),(2,1,2))
        whole=next(r for r in pair_summary(paired) if r['scene_id']=='ALL' and r['cohort']=='all_gt')
        self.assertEqual((whole['IR_precovered_gt'],whole['Fusion_precovered_gt']),(3,2))
    def test_duplicate_stage_rejected(self):
        rows=sample_rows()
        with self.assertRaises(ValueError):build_queue(rows+[rows[0]])
    def test_gt_candidate_rows_and_physical_candidates_not_conflated(self):
        events=[]
        for gi in (0,1):
            events.append(dict(model='IR',stem='a',scene_id='03',gt_index=gi,has_primary_relation=True,
                all_deleted_good_have_primary_relation=True,unexplained_deleted_good_count=0,
                has_primary_score_tie=False,has_eligible_score_tie=False,near_nms_threshold=False,
                relation_predicate_disagreement=False,numerically_sensitive=False,has_diagnostic_only_relation=False,
                any_related_retained_other_gt_overlap_larger=False))
        deleted=[dict(model='IR',scene_id='03',deleted_candidate_id='IR/a/10') for _ in events]
        related=[dict(model='IR',scene_id='03',deleted_candidate_id='IR/a/10',retained_candidate_id='IR/a/11') for _ in events]
        _,paired=build_queue(sample_rows())
        r=next(r for r in model_summary(events,deleted,related,paired) if r['scene_id']=='ALL' and r['model']=='IR')
        self.assertEqual((r['nms_lost_gt'],r['deleted_good_candidate_GT_rows'],r['unique_deleted_good_candidates']),(2,2,1))
        self.assertEqual((r['primary_relation_GT_rows'],r['unique_primary_candidate_pairs']),(2,1))
    def test_unexplained_flag_is_per_gt_not_number_of_missing_candidates(self):
        event=dict(model='IR',stem='a',scene_id='03',gt_index=0,has_primary_relation=True,
            all_deleted_good_have_primary_relation=False,unexplained_deleted_good_count=2,has_primary_score_tie=False,
            has_eligible_score_tie=False,near_nms_threshold=False,relation_predicate_disagreement=False,
            numerically_sensitive=False,has_diagnostic_only_relation=False,any_related_retained_other_gt_overlap_larger=False)
        _,paired=build_queue(sample_rows())
        r=next(r for r in model_summary([event],[],[],paired) if r['scene_id']=='ALL' and r['model']=='IR')
        self.assertEqual(r['gt_partly_or_fully_unexplained_primary'],1);self.assertEqual(r['gt_with_primary_relation'],1)

if __name__=='__main__':unittest.main()
