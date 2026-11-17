"""Adapter export verification with artificial tensor data only; no model loading."""
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from capture_single_val import export_trace, legacy_iou, loss_transitions
from test_trace_rpn_stages import execute, NumericRPN

class AdapterTests(unittest.TestCase):
    def test_float32_cpu_and_float64_are_separate_and_exports_keep_ids(self):
        cap,_=execute()
        record,trace=cap.records[0],cap.traces[0]
        x=legacy_iou(record.final_boxes,record.gt)
        self.assertEqual(x.device.type,"cpu");self.assertEqual(x.dtype,torch.float32)
        self.assertEqual(trace.stages["post_nms_topn"].iou.dtype,torch.float64)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"checks") as d:
            out=Path(d);rows=export_trace(out,"IR",record,trace)
            self.assertEqual(len(rows),7)
            for r in rows:
                self.assertIn("max_iou_float32_cpu",r)
                self.assertIn("max_iou_float64_diagnostic",r)
            with (out/"IR_candidate_trajectory.csv").open(encoding="utf-8-sig",newline="") as f:
                histories=list(csv.DictReader(f))
            final_best=trace.stages["post_nms_topn"].summary["best_candidate_id"]
            self.assertTrue(any(r["candidate_id"]==final_best for r in histories))
            self.assertTrue((out/"IR_candidate_catalogue.csv").is_file())
    def test_export_all_empty_stages(self):
        cap,_=execute(torch.empty(0,4),torch.empty(0),(0,0,0),NumericRPN())
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"checks") as d:
            rows=export_trace(Path(d),"IR",cap.records[0],cap.traces[0])
            self.assertTrue(all(r["candidate_count"]==0 for r in rows))
            self.assertTrue(all(r["max_iou_float32_cpu"]==0 and r["best_id_float32_cpu"] is None for r in rows))
    def test_loss_transitions_do_not_ignore_recovery(self):
        rows=[dict(stage=s,n=n) for s,n in [("decoded",2),("topn",0),("clipped",1),("score",0)]]
        self.assertEqual(loss_transitions(rows,"n"),[{"from":"decoded","to":"topn"},{"from":"clipped","to":"score"}])
    def test_raw_save_includes_originals_when_replay_fails(self):
        from capture_single_val import save_capture
        cap,_=execute()
        cap.traces=[]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"checks") as d:
            save_capture(Path(d),"IR",cap,{"rpn_calls":0})
            data=torch.load(Path(d)/"IR_raw_capture.pth",weights_only=True)
            self.assertEqual(len(data["records"]),1);self.assertEqual(data["stages"],[])
            self.assertTrue(torch.equal(data["records"][0]["final_boxes"],cap.records[0].final_boxes))

if __name__=="__main__":unittest.main(verbosity=2)

