"""Synthetic unit checks only; these are not LLVIP experiment results."""
import copy
import unittest
from unittest.mock import patch
import capture_remaining_val as remaining
from verify_merged_capture import transitions

class RemainingTests(unittest.TestCase):
    def test_fixed_order_and_indices(self):
        rows=[dict(stem=s,gt_index=str(g),image_index=str(i)) for s,g,i in remaining.FIXED]
        self.assertIs(remaining.selected_cases(rows),rows)
        bad=copy.deepcopy(rows);bad[1],bad[2]=bad[2],bad[1]
        with self.assertRaises(RuntimeError):remaining.selected_cases(bad)
        bad=copy.deepcopy(rows);bad[2]["gt_index"]="1"
        with self.assertRaises(RuntimeError):remaining.selected_cases(bad)
    def test_reused_case_has_no_live_adapter(self):
        with self.assertRaises(ValueError):remaining.adapter("030269",1)
        with self.assertRaises(ValueError):remaining.adapter("unknown",0)
    def test_private_bindings_preserve_single(self):
        import capture_single_val as original
        a=remaining.adapter("030308",1);b=remaining.adapter("030205",0)
        self.assertEqual((original.STEM,original.GT_INDEX),("030269",1))
        self.assertEqual((a.STEM,a.GT_INDEX),("030308",1))
        self.assertEqual((b.STEM,b.GT_INDEX),("030205",0))
        self.assertIs(a.capture_model.__wrapped__.__globals__,a.__dict__)
        self.assertIs(b.capture_model.__wrapped__.__globals__,b.__dict__)
    def test_clip_can_create_high_iou_and_later_loss(self):
        rows=[dict(stage=s,high=h) for s,h in (("decoded",0),("pre_nms_topn",0),("clipped",1),
             ("small_box_filtered",1),("score_filtered",1),("layer_nms",0),("post_nms_topn",0))]
        result=transitions(rows,"high")
        self.assertEqual(result["first_high"],"clipped")
        self.assertEqual(result["all_lost_at"],["layer_nms"])
        self.assertEqual(result["reappeared_or_first_appeared_after_decode"],["clipped"])
    def test_boundary_no_threshold_tolerance(self):
        x=.7500129342079163
        self.assertTrue(x>=.75);self.assertFalse(.749999999>=.75)
    def test_scope_export_uses_actual_case_without_forward(self):
        import tempfile
        from pathlib import Path
        import torch
        a=remaining.adapter("170455",1)
        with tempfile.TemporaryDirectory() as temp:
            a.save_capture(Path(temp),"IR",None,{"model_forward_calls":0})
            d=torch.load(Path(temp)/"IR_raw_capture.pth",weights_only=True)
            self.assertEqual(d["scope"],"170455/G1 only")
            self.assertEqual(d["records"],[])

if __name__=="__main__":unittest.main()
