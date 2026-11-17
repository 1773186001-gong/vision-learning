"""Only necessary synthetic checks for the single L hypothesis."""
import unittest
import torch
from layer_support import measure,counts_by_level,make_pairs,direction_summary,background_summary

def catalogue(n=12):return [{'candidate_id':f'synthetic/{i}'} for i in range(n)]
def support(model,L,lost,stem='a',scene='03'):
    return dict(model=model,stem=stem,gt_index=0,scene_id=scene,gt_xyxy='[0,0,10,10]',L=L,nms_lost=lost)

class NecessaryTests(unittest.TestCase):
    def test_many_good_boxes_in_one_level_count_one(self):
        gt=torch.tensor([0.,0.,10.,10.]);b=gt.repeat(3,1)
        L,counts,ids,_=measure(gt,b,torch.tensor([0,0,0]),torch.tensor([3,5,8]),catalogue())
        self.assertEqual(L,1);self.assertEqual(counts,{'0':3,'1':0,'pool':0});self.assertEqual(len(ids['0']),3)
    def test_real_level_indices_not_size_or_flat_id_guess(self):
        gt=torch.tensor([0.,0.,10.,10.]);b=gt.repeat(3,1)
        L,counts,ids,_=measure(gt,b,torch.tensor([2,0,1]),torch.tensor([9,2,6]),catalogue())
        self.assertEqual(L,3);self.assertEqual(counts,{'0':1,'1':1,'pool':1})
        self.assertEqual(ids['pool'],['synthetic/9']);self.assertEqual(ids['0'],['synthetic/2'])
    def test_exact_075_is_included_without_tolerance(self):
        gt=torch.tensor([0.,0.,10.,10.])
        L,_,_,v=measure(gt,torch.tensor([[0.,0.,7.5,10.]]),torch.tensor([1]),torch.tensor([0]),catalogue())
        self.assertEqual(float(v[0]),.75);self.assertEqual(L,1)
        with self.assertRaisesRegex(ValueError,'L=0'):
            measure(gt,torch.tensor([[0.,0.,7.49999,10.]]),torch.tensor([1]),torch.tensor([0]),catalogue())
    def test_zero_support_in_mother_set_is_error(self):
        with self.assertRaisesRegex(ValueError,'L=0'):
            measure(torch.tensor([0.,0.,10.,10.]),torch.empty((0,4)),torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64),catalogue())
    def test_bad_layer_or_standard_pyramid_keys_rejected(self):
        with self.assertRaises(ValueError):counts_by_level(torch.tensor([True]),torch.tensor([3]))
        with self.assertRaises(ValueError):counts_by_level(torch.tensor([True]),torch.tensor([0]),('P2','P3','P4'))
    def test_failed_minus_survivor_direction_for_both_models(self):
        c=[dict(stem='a',gt_index='0')]
        ir=make_pairs([support('IR',1,True),support('Fusion',3,False)],c)[0]
        self.assertEqual((ir['failed_model'],ir['delta_L'],ir['direction']),('IR',-2,'fewer'))
        fu=make_pairs([support('IR',3,False),support('Fusion',1,True)],c)[0]
        self.assertEqual((fu['failed_model'],fu['delta_L'],fu['direction']),('Fusion',-2,'fewer'))
        opposite=make_pairs([support('IR',1,False),support('Fusion',2,True)],c)[0]
        self.assertEqual((opposite['delta_L'],opposite['direction']),(1,'more'))
    def test_both_lost_or_kept_do_not_create_asymmetric_delta(self):
        c=[dict(stem='a',gt_index='0')]
        for lost in (True,False):
            p=make_pairs([support('IR',1,lost),support('Fusion',2,lost)],c)[0]
            self.assertIsNone(p['delta_L']);self.assertIsNone(p['failed_model']);self.assertIsNone(p['direction'])
    def test_paired_counts_and_zero_denominator(self):
        rows=[support('IR',1,True,'a'),support('Fusion',2,False,'a'),
              support('IR',2,False,'b'),support('Fusion',2,True,'b'),
              support('IR',1,False,'c','13'),support('Fusion',2,True,'c','13')]
        pp=make_pairs(rows,[dict(stem=s,gt_index=0) for s in ('a','b','c')])
        ss=direction_summary(pp);overall=next(r for r in ss if r['scene_id']=='ALL' and r['failed_model']=='ALL')
        self.assertEqual((overall['fewer_GT'],overall['equal_GT'],overall['more_GT']),(1,1,1))
        empty=next(r for r in ss if r['scene_id']=='08' and r['failed_model']=='IR')
        self.assertEqual(empty['asymmetric_GT'],0);self.assertIsNone(empty['fewer_rate'])
    def test_background_by_L_keeps_risk_denominators(self):
        rr=[support('IR',1,True,'a'),support('IR',1,False,'b'),support('Fusion',2,False,'a'),support('Fusion',3,True,'b')]
        ss=background_summary(rr);one=next(r for r in ss if r['scene_id']=='ALL' and r['model']=='IR' and r['L']==1)
        self.assertEqual((one['risk_GT'],one['nms_lost_GT'],one['conditional_loss_rate']),(2,1,.5))
        empty=next(r for r in ss if r['scene_id']=='ALL' and r['model']=='IR' and r['L']==3)
        self.assertIsNone(empty['conditional_loss_rate'])
    def test_duplicate_GT_keys_rejected(self):
        with self.assertRaises(ValueError):
            make_pairs([support('IR',1,True),support('IR',1,True),support('Fusion',2,False)],[dict(stem='a',gt_index=0)])

if __name__=='__main__':unittest.main()
