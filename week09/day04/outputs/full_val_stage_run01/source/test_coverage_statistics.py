"""Artificial geometry unit checks only; no LLVIP models or image forwards."""
import copy
import unittest
import torch
from coverage_statistics import STAGES,measure_image,summarize

def stage(boxes,ids=None):
    boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
    return {"boxes":boxes,"ids":torch.tensor(list(range(len(boxes))) if ids is None else ids,dtype=torch.int64)}

def fixture(gt,boxes,changes=None,stem="synthetic",scene="03",model="IR",catalogue_size=None):
    states={s:stage(boxes) for s in STAGES}
    for s,v in (changes or {}).items():states[s]=v
    size=catalogue_size if catalogue_size is not None else len(boxes)
    catalogue=[{"candidate_id":f"synthetic/{model}/{stem}/{i}"} for i in range(size)]
    return measure_image(model,stem,scene,0,torch.tensor(gt,dtype=torch.float32).reshape(-1,4),states,catalogue,(640,640))

def both_models(gt,boxes,changes=None,stem="synthetic",scene="03"):
    a,ai=fixture(gt,boxes,changes,stem,scene,"IR")
    b,bi=fixture(gt,boxes,changes,stem,scene,"Fusion")
    return a+b,[ai,bi]

class CoverageTests(unittest.TestCase):
    def test_multi_gt_multiple_boxes_not_multiple_gt_hits(self):
        rows,images=both_models([[0,0,10,10],[20,20,30,30]],[[0,0,10,10],[0,0,10,10]])
        self.assertEqual(len(rows),28)
        self.assertEqual(rows[0]["count_iou_ge_075_float32_cpu"],2)
        summary=summarize(rows,images)
        r=next(r for r in summary['stage_summary'] if r['scope']=='overall' and r['model']=='IR' and r['stage']=='decoded' and r['iou_convention']=='float32_cpu')
        self.assertEqual((r['covered_gt'],r['valid_gt'],r['coverage_rate']),(1,2,.5))
    def test_empty_gt_image_not_fabricated(self):
        rows,images=both_models([],[[0,0,10,10]])
        self.assertEqual(rows,[]);self.assertTrue(images[0]['empty_gt'])
        stats=summarize(rows,images)
        self.assertEqual(stats['gt_coverage_sequences'],[])
        self.assertTrue(all(r['coverage_rate'] is None and r['empty_gt_images']==1 for r in stats['stage_summary']))
    def test_empty_proposals_have_zero_and_no_best_id(self):
        rows,_=fixture([[0,0,10,10]],[])
        self.assertTrue(all(r['max_iou_float32_cpu']==0 and r['best_candidate_id_float32_cpu'] is None for r in rows))
    def test_partial_good_loss_keeps_gt_covered(self):
        one=stage([[0,0,10,10]],[1]);changes={s:one for s in STAGES[4:]}
        rows,_=fixture([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]],changes)
        r=rows[4]
        self.assertTrue(r['partial_high_loss_coverage_kept_float32_cpu'])
        self.assertFalse(r['coverage_lost_float32_cpu'])
    def test_all_good_loss_and_conditional_rate(self):
        changes={s:stage([]) for s in STAGES[4:]}
        rows,images=both_models([[0,0,10,10]],[[0,0,10,10]],changes)
        stats=summarize(rows,images)
        r=next(r for r in stats['stage_summary'] if r['scope']=='overall' and r['model']=='IR' and r['stage']=='score_filtered' and r['iou_convention']=='float32_cpu')
        self.assertEqual((r['coverage_lost_gt'],r['previous_covered_gt'],r['lost_rate_previous_covered']),(1,1,1.))
        self.assertEqual(stats['gt_coverage_sequences'][0]['coverage_sequence'],'1111000')
    def test_clipping_first_creates_coverage(self):
        changes={s:stage([[0,0,10,10]]) for s in STAGES[2:]}
        rows,images=both_models([[0,0,10,10]],[[-10,0,10,10]],changes)
        stats=summarize(rows,images)
        t=stats['gt_coverage_sequences'][0]
        self.assertEqual(t['coverage_sequence'],'0011111')
        self.assertEqual(t['first_covered_stage'],'clipped');self.assertFalse(t['never_covered'])
    def test_topn_loss_then_real_clip_recovery_recorded(self):
        # Best full-image box was removed; the surviving larger box becomes exact after clipping.
        changes={'pre_nms_topn':stage([[-10,0,10,10]],[1]),**{s:stage([[0,0,10,10]],[1]) for s in STAGES[2:]}}
        a,ai=fixture([[0,0,10,10]],[[0,0,10,10],[-10,0,10,10]],changes,model='IR')
        b,bi=fixture([[0,0,10,10]],[[0,0,10,10],[-10,0,10,10]],changes,model='Fusion')
        ts=summarize(a+b,[ai,bi])['gt_coverage_sequences']
        self.assertEqual(ts[0]['coverage_sequence'],'1011111');self.assertTrue(ts[0]['recovered_after_loss'])
    def test_gt_weighted_not_image_average(self):
        a,ai=both_models([[0,0,10,10]],[[0,0,10,10]],stem='a')
        b,bi=both_models([[0,0,10,10],[20,20,30,30],[40,40,50,50]],[],stem='b',scene='13')
        stats=summarize(a+b,ai+bi)
        r=next(r for r in stats['stage_summary'] if r['scope']=='overall' and r['stage']=='decoded' and r['model']=='IR' and r['iou_convention']=='float32_cpu')
        self.assertEqual(r['coverage_rate'],.25)
    def test_net_changes_are_arithmetic_with_gains_and_losses(self):
        a,ai=fixture([[0,0,10,10]],[[0,0,10,10]],{s:stage([]) for s in STAGES[4:]},model='IR')
        b,bi=fixture([[0,0,10,10]],[[-10,0,10,10]],{s:stage([[0,0,10,10]]) for s in STAGES[2:]},model='Fusion')
        stats=summarize(a+b,[ai,bi]);net=[r for r in stats['net_difference'] if r['scope']=='overall' and r['iou_convention']=='float32_cpu']
        self.assertEqual([r['IR_minus_Fusion'] for r in net],[1,1,0,0,-1,-1,-1])
        self.assertEqual(net[2]['net_change_from_previous'],-1);self.assertEqual(net[2]['IR_gains_minus_Fusion_gains'],-1)
        self.assertEqual(net[4]['Fusion_losses_minus_IR_losses'],-1)
    def test_threshold_strict_without_epsilon(self):
        a,_=fixture([[0,0,10,10]],[[0,0,7.5,10]])
        b,_=fixture([[0,0,10,10]],[[0,0,7.499999,10]])
        self.assertTrue(a[0]['covered_075_float32_cpu']);self.assertFalse(b[0]['covered_075_float32_cpu'])
    def test_equal_coordinates_keep_ids_and_stage_tie_order(self):
        changes={s:stage([[0,0,10,10],[0,0,10,10]],[1,0]) for s in STAGES}
        rows,_=fixture([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]],changes)
        self.assertTrue(rows[0]['best_candidate_id_float32_cpu'].endswith('/1'))
        self.assertEqual(rows[0]['count_iou_ge_075_float32_cpu'],2)
    def test_duplicate_and_pair_mismatch_rejected(self):
        rows,images=both_models([[0,0,10,10]],[[0,0,10,10]])
        with self.assertRaises(ValueError):summarize(rows+[rows[0]],images)
        bad=copy.deepcopy(rows)
        for r in bad:
            if r['model']=='Fusion':r['gt_xyxy']='[1,1,2,2]'
        with self.assertRaises(ValueError):summarize(bad,images)
    def test_zero_condition_denominator_is_none(self):
        rows,images=both_models([[0,0,10,10]],[])
        stats=summarize(rows,images)
        self.assertTrue(all(r['lost_rate_previous_covered'] is None for r in stats['stage_summary']))
    def test_invalid_gt_and_missing_stage_rejected(self):
        with self.assertRaises(ValueError):fixture([[0,0,0,10]],[])
        with self.assertRaises(ValueError):fixture([[-1,0,1,10]],[])

if __name__=='__main__':unittest.main()
