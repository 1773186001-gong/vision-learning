"""Artificial tensor unit verification ONLY. No LLVIP, weights, model or network."""
import contextlib
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
import trace_rpn_stages as trace_module
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import boxes as ops
from trace_rpn_stages import TraceCapture, replay, VerificationError, STAGES, _LOCK

class NumericRPN:
    """Plain Python receiver for the installed original numeric methods; not an nn.Module."""
    filter_proposals = RegionProposalNetwork.filter_proposals
    _get_top_n_idx = RegionProposalNetwork._get_top_n_idx
    def __init__(self, pre=2, post=3, minimum=.001, threshold=.05):
        self.pre,self.post=pre,post
        self.min_size,self.score_thresh,self.nms_thresh=minimum,threshold,.7
    def pre_nms_top_n(self):return self.pre
    def post_nms_top_n(self):return self.post

class HookTarget:
    def __init__(self, fail=False):self.callbacks={};self.fail=fail
    def register_forward_hook(self, callback):
        if self.fail:raise RuntimeError("synthetic registration failure")
        key=len(self.callbacks);self.callbacks[key]=callback
        owner=self
        class Handle:
            def remove(self):owner.callbacks.pop(key,None)
        return Handle()
    def emit(self):
        for callback in list(self.callbacks.values()):callback(self,(),None)

class ReadFaultRPN(NumericRPN):
    """Inject early call/lookup errors without creating a real model."""
    def __getattribute__(self,name):
        state=object.__getattribute__(self,"__dict__")
        if name==state.get("fail_read"):
            raise RuntimeError("synthetic lookup failure: "+name)
        return super().__getattribute__(name)
    def pre_nms_top_n(self):
        if self.__dict__.get("fail_call")=="pre_nms_top_n":
            raise RuntimeError("synthetic configuration failure: pre_nms_top_n")
        return super().pre_nms_top_n()
    def post_nms_top_n(self):
        if self.__dict__.get("fail_call")=="post_nms_top_n":
            raise RuntimeError("synthetic configuration failure: post_nms_top_n")
        return super().post_nms_top_n()

GT=torch.tensor([2.,2.,8.,8.])

def example():
    boxes=torch.tensor([[2.,2.,8.,8.],[0.,0.,1.,1.],[8.,8.,10.,10.],
                        [3.,3.,9.,9.],[0.,8.,1.,9.],[8.,0.,9.,1.],
                        [1.,2.,7.,8.],[0.,4.,1.,5.],[9.,4.,10.,5.]])
    logits=torch.logit(torch.tensor([.8,.7,.6,.9,.7,.6,.95,.7,.6]))
    return boxes,logits,(3,3,3)

def execute(boxes=None,logits=None,counts=None,rpn=None,gt=GT):
    if boxes is None:boxes,logits,counts=example()
    if rpn is None:rpn=NumericRPN()
    with TraceCapture(rpn,("0","1","pool"),gt) as cap:
        output=rpn.filter_proposals(boxes[None],logits,[(10,10)],list(counts))
        assert output is cap.returned_reference
    return cap,output

class TraceTests(unittest.TestCase):
    def _assert_early_failure_cleanup(self,rpn,message):
        self.assertFalse(_LOCK.locked())
        before=dict(rpn.__dict__);original_nms=ops.batched_nms
        original_ops=trace_module.rpn_module.box_ops
        target=HookTarget()
        target.register_forward_hook(lambda *a:None)
        old_hooks=dict(target.callbacks)
        b,l,c=example()
        cap=TraceCapture(rpn,("0","1","pool"),GT,hooks=((target,lambda *a:None),))
        with self.assertRaisesRegex(RuntimeError,message):
            with cap:
                rpn.filter_proposals(b[None],l,[(10,10)],list(c))
        self.assertFalse(_LOCK.locked())
        self.assertFalse(cap.active)
        self.assertEqual(rpn.__dict__,before)
        self.assertEqual(target.callbacks,old_hooks)
        self.assertIs(ops.batched_nms,original_nms)
        self.assertIs(trace_module.rpn_module.box_ops,original_ops)

    def _assert_normal_capture_after_failure(self,rpn):
        before=dict(rpn.__dict__)
        cap,_=execute(rpn=rpn)
        self.assertEqual(len(cap.traces),1)
        self.assertTrue(cap.traces[0].verified)
        self.assertFalse(_LOCK.locked())
        self.assertEqual(rpn.__dict__,before)

    def test_configuration_calls_failure_releases_lock_and_recovers(self):
        for name in ("pre_nms_top_n","post_nms_top_n"):
            with self.subTest(name=name):
                rpn=ReadFaultRPN();rpn.fail_call=name
                self._assert_early_failure_cleanup(rpn,"configuration failure")
                rpn.fail_call=None
                self._assert_normal_capture_after_failure(rpn)

    def test_configuration_attributes_failure_releases_lock_and_recovers(self):
        for name in ("min_size","score_thresh","nms_thresh"):
            with self.subTest(name=name):
                rpn=ReadFaultRPN();rpn.fail_read=name
                self._assert_early_failure_cleanup(rpn,"lookup failure")
                rpn.fail_read=None
                self._assert_normal_capture_after_failure(rpn)

    def test_top_method_reference_failure_releases_lock_and_recovers(self):
        rpn=ReadFaultRPN();rpn.fail_read="_get_top_n_idx"
        self._assert_early_failure_cleanup(rpn,"lookup failure")
        rpn.fail_read=None
        self._assert_normal_capture_after_failure(rpn)

    def test_nms_method_reference_failure_releases_lock_and_recovers(self):
        class BrokenOps:
            @property
            def batched_nms(self):
                raise RuntimeError("synthetic NMS reference failure")
        rpn=ReadFaultRPN()
        with patch.object(trace_module.rpn_module,"box_ops",BrokenOps()):
            self._assert_early_failure_cleanup(rpn,"NMS reference failure")
        self._assert_normal_capture_after_failure(rpn)

    def test_config_construction_failure_releases_lock_and_recovers(self):
        rpn=ReadFaultRPN()
        with patch.object(trace_module,"Config",side_effect=RuntimeError("synthetic Config failure")):
            self._assert_early_failure_cleanup(rpn,"Config failure")
        self._assert_normal_capture_after_failure(rpn)

    def test_original_filter_reference_failure_restores_hooks(self):
        # This lookup happens before acquiring the lock, during context entry.
        rpn=ReadFaultRPN();rpn.fail_read="filter_proposals"
        self._assert_early_failure_cleanup(rpn,"lookup failure")
        rpn.fail_read=None
        self._assert_normal_capture_after_failure(rpn)

    def test_three_levels_and_budgets(self):
        cap,_=execute()
        t=cap.traces[0]
        self.assertEqual(len(t.catalogue),9)
        self.assertEqual([(t.catalogue[i]["level_key"],t.catalogue[i]["index_within_level"])
                          for i in (0,3,6)], [("0",0),("1",0),("pool",0)])
        self.assertEqual(t.stages["pre_nms_topn"].summary["candidate_count"],6)
        self.assertEqual(t.stages["post_nms_topn"].summary["candidate_count"],3)
        self.assertEqual(t.stages["pre_nms_topn"].summary["counts_by_level"],{"0":2,"1":2,"pool":2})
        self.assertEqual(tuple(t.stages),STAGES)
        self.assertTrue(t.verified)

    def test_identical_coordinates_keep_distinct_identity(self):
        b=torch.tensor([[2.,2.,8.,8.],[2.,2.,8.,8.],[2.,2.,8.,8.]])
        cap,_=execute(b,torch.logit(torch.tensor([.9,.8,.7])),(2,0,1),NumericRPN(pre=3,post=3))
        t=cap.traces[0]
        self.assertEqual(t.stages["layer_nms"].ids.tolist(),[0,2])
        self.assertNotEqual(t.catalogue[0]["candidate_id"],t.catalogue[2]["candidate_id"])
        self.assertEqual(t.first_removed[1],"layer_nms")
        self.assertEqual(t.history(2)[-1]["kept"],True)

    def test_clip_inside_gt_never_decreases_and_preserves_id(self):
        b=torch.tensor([[-3.,-3.,12.,12.],[20.,20.,25.,25.],[2.,2.,8.,8.]])
        cap,_=execute(b,torch.tensor([3.,2.,1.]),(1,1,1),NumericRPN(pre=3))
        t=cap.traces[0]
        self.assertEqual(t.stages["clipped"].ids.tolist(),t.stages["pre_nms_topn"].ids.tolist())
        self.assertTrue(all(x["iou_after"]+1e-6>=x["iou_before"] for x in t.clip_deltas))
        self.assertTrue(next(x for x in t.clip_deltas if x["flat_index"]==1)["degenerate_after"])
        self.assertEqual(t.first_removed[1],"small_box_filtered")
        self.assertGreater(t.clip_deltas[0]["delta"],0)

    def test_min_size_and_score_boundaries(self):
        b=torch.tensor([[0.,0.,.001,1.],[0.,0.,.000999,1.],
                        [2.,2.,8.,8.],[1.,2.,7.,8.],[2.,1.,8.,7.]])
        logits=torch.logit(torch.tensor([.9,.9,.0499,.05,.0501]))
        cap,_=execute(b,logits,(2,2,1),NumericRPN(pre=5,post=10))
        t=cap.traces[0]
        self.assertIn(0,t.stages["small_box_filtered"].ids.tolist())
        self.assertNotIn(1,t.stages["small_box_filtered"].ids.tolist())
        self.assertNotIn(2,t.stages["score_filtered"].ids.tolist())
        self.assertIn(3,t.stages["score_filtered"].ids.tolist())
        self.assertIn(4,t.stages["score_filtered"].ids.tolist())

    def test_ties_keep_original_indices_and_order(self):
        b=torch.tensor([[0.,0.,1.,1.],[2.,0.,3.,1.],[4.,0.,5.,1.],
                        [0.,3.,1.,4.],[2.,3.,3.,4.],[4.,3.,5.,4.]])
        cap,_=execute(b,torch.ones(6),(3,2,1),NumericRPN(pre=2,post=10))
        r,t=cap.records[0],cap.traces[0]
        self.assertEqual(t.stages["pre_nms_topn"].ids.tolist(),r.top_indices.tolist())
        self.assertEqual(t.stages["layer_nms"].ids.tolist(),
                         t.stages["score_filtered"].ids[r.nms_indices].tolist())
        self.assertTrue(all(t.history(i)[-1]["score_rank_min"]==1
                            for i in t.stages["post_nms_topn"].ids.tolist()))

    def test_empty_candidates(self):
        cap,_=execute(torch.empty((0,4)),torch.empty(0),(0,0,0),NumericRPN())
        t=cap.traces[0]
        self.assertEqual(t.clip_deltas,[])
        for stage in t.stages.values():
            self.assertTrue(stage.summary["empty"])
            self.assertEqual(stage.summary["max_iou"],0)
            self.assertIsNone(stage.summary["best_candidate_id"])

    def test_full_640_image_bounds_include_letterbox_padding(self):
        rpn=NumericRPN(pre=2,post=3)
        gt=torch.tensor([1.,1.,20.,20.])
        b=torch.tensor([[0.,0.,30.,30.],[-10.,-10.,30.,30.]])
        with TraceCapture(rpn,("0","1","pool"),gt) as cap:
            rpn.filter_proposals(b[None],torch.tensor([2.,1.]),[(640,640)],[1,1,0])
        s=cap.traces[0].stages["clipped"]
        self.assertEqual(s.boxes[:,0].tolist(),[0.,0.])
        self.assertEqual(s.boxes[:,1].tolist(),[0.,0.])
        self.assertGreater(s.summary["max_iou"],0.)

    def test_real_torch_hook_handles_on_artificial_tensor(self):
        # Identity has no learned parameters; its input is not an image or LLVIP data.
        toy=torch.nn.Identity();before=dict(toy._forward_hooks);rpn=NumericRPN()
        events=[]
        with TraceCapture(rpn,("0","1","pool"),GT,hooks=((toy,lambda *a:events.append(1)),)):
            toy(torch.tensor([1.]))
        self.assertEqual(events,[1]);self.assertEqual(dict(toy._forward_hooks),before)
        def fail(*a):raise RuntimeError("synthetic hook failure")
        with self.assertRaisesRegex(RuntimeError,"hook failure"):
            with TraceCapture(rpn,("0","1","pool"),GT,hooks=((toy,fail),)):
                toy(torch.tensor([1.]))
        self.assertEqual(dict(toy._forward_hooks),before)

    def test_all_candidates_removed_at_score(self):
        cap,_=execute(torch.tensor([[2.,2.,8.,8.]]),torch.tensor([-10.]),(1,0,0),NumericRPN())
        self.assertEqual(cap.traces[0].stages["post_nms_topn"].summary["candidate_count"],0)
        self.assertEqual(cap.traces[0].first_removed[0],"score_filtered")

    def test_stage_best_switch_and_fixed_history(self):
        b=torch.tensor([[2.,2.,8.,8.],[1.,2.,8.,8.],[8.,8.,9.,9.]])
        cap,_=execute(b,torch.logit(torch.tensor([.1,.9,.8])),(2,0,1),NumericRPN(pre=1))
        t=cap.traces[0]
        self.assertEqual(t.stages["decoded"].summary["best_candidate_id"],t.catalogue[0]["candidate_id"])
        self.assertEqual(t.stages["pre_nms_topn"].summary["best_candidate_id"],t.catalogue[1]["candidate_id"])
        self.assertEqual(t.history(0)[1]["first_removed_stage"],"pre_nms_topn")
        self.assertFalse(t.history(0)[1]["kept"])
        self.assertNotIn(0,[x["flat_index"] for x in t.clip_deltas])

    def test_pure_deletion_monotonicity(self):
        t=execute()[0].traces[0]
        chain=["clipped","small_box_filtered","score_filtered","layer_nms","post_nms_topn"]
        for a,b in zip(chain,chain[1:]):
            self.assertLessEqual(t.stages[b].summary["max_iou"],t.stages[a].summary["max_iou"]+1e-6)
            self.assertLessEqual(t.stages[b].summary["count_iou_ge_075"],t.stages[a].summary["count_iou_ge_075"])

    def test_gt_outside_and_invalid_candidates_are_rejected(self):
        b,l,c=example()
        for gt in (torch.tensor([-1.,2.,8.,8.]),torch.tensor([2.,2.,11.,8.]),
                   torch.tensor([2.,2.,2.,8.])):
            with self.assertRaises(ValueError):execute(b,l,c,gt=gt)
        for bad in (torch.tensor([[2.,2.,2.,8.]]),torch.tensor([[float("nan"),0.,2.,3.]])):
            with self.assertRaises(ValueError):execute(bad,torch.ones(1),(1,0,0))

    def test_explicit_float_tolerance_recorded(self):
        cap,_=execute()
        self.assertEqual(cap.records[0].config.iou_atol,1e-6)
        replay(replace(cap.records[0],config=replace(cap.records[0].config,iou_atol=1e-12)))

    def test_fault_injected_coordinate_order_score_detected(self):
        r=execute()[0].records[0]
        bad=r.final_boxes.clone();bad[0,0]+=.01
        with self.assertRaises(VerificationError):replay(replace(r,final_boxes=bad))
        with self.assertRaises(VerificationError):replay(replace(r,final_boxes=r.final_boxes.flip(0),final_scores=r.final_scores.flip(0)))
        bad=r.final_scores.clone();bad[0]-=.01
        with self.assertRaises(VerificationError):replay(replace(r,final_scores=bad))
        bad=r.nms_boxes.clone();bad[0,0]+=.01
        with self.assertRaises(VerificationError):replay(replace(r,nms_boxes=bad))
        with self.assertRaises(VerificationError):replay(replace(r,top_indices=torch.zeros_like(r.top_indices)))

    def test_normal_cleanup_hooks_and_return_identity(self):
        rpn=NumericRPN();before=dict(rpn.__dict__);original=ops.batched_nms
        target=HookTarget();events=[]
        b,l,c=example()
        with TraceCapture(rpn,("0","1","pool"),GT,hooks=((target,lambda *a:events.append("event")),)) as cap:
            target.emit()
            out=rpn.filter_proposals(b[None],l,[(10,10)],list(c))
            self.assertIs(out,cap.returned_reference)
            self.assertIs(ops.batched_nms,original) # narrow scope already restored
        self.assertEqual(rpn.__dict__,before)
        self.assertEqual(target.callbacks,{})
        self.assertEqual(events,["event"])

    def test_context_body_exception_cleanup(self):
        rpn=NumericRPN();before=dict(rpn.__dict__);target=HookTarget();original=ops.batched_nms
        with self.assertRaisesRegex(RuntimeError,"body"):
            with TraceCapture(rpn,("0","1","pool"),GT,hooks=((target,lambda *a:None),)):
                raise RuntimeError("synthetic body")
        self.assertEqual(rpn.__dict__,before);self.assertEqual(target.callbacks,{})
        self.assertIs(ops.batched_nms,original)

    def test_original_nms_exception_cleanup(self):
        rpn=NumericRPN();before=dict(rpn.__dict__);original=ops.batched_nms;target=HookTarget()
        def fail(*a):raise RuntimeError("synthetic NMS failure")
        ops.batched_nms=fail
        try:
            b,l,c=example()
            with self.assertRaisesRegex(RuntimeError,"NMS failure"):
                with TraceCapture(rpn,("0","1","pool"),GT,hooks=((target,lambda *a:None),)):
                    rpn.filter_proposals(b[None],l,[(10,10)],list(c))
            self.assertIs(ops.batched_nms,fail)
            self.assertEqual(rpn.__dict__,before);self.assertEqual(target.callbacks,{})
            self.assertFalse(_LOCK.locked())
        finally:ops.batched_nms=original

    def test_existing_instance_override_restored(self):
        rpn=NumericRPN();original=rpn._get_top_n_idx
        custom=lambda *a,**k:original(*a,**k)
        rpn._get_top_n_idx=custom
        bound=rpn.filter_proposals;rpn.filter_proposals=lambda *a,**k:bound(*a,**k)
        before=dict(rpn.__dict__);b,l,c=example()
        with TraceCapture(rpn,("0","1","pool"),GT):
            rpn.filter_proposals(b[None],l,[(10,10)],list(c))
        self.assertEqual(rpn.__dict__,before)

    def test_partial_hook_registration_failure_cleanup(self):
        rpn=NumericRPN();before=dict(rpn.__dict__);a=HookTarget();b=HookTarget(fail=True)
        with self.assertRaisesRegex(RuntimeError,"registration"):
            with TraceCapture(rpn,("0","1","pool"),GT,hooks=((a,lambda *x:None),(b,lambda *x:None))):
                self.fail("must not enter")
        self.assertEqual(a.callbacks,{})
        self.assertEqual(rpn.__dict__,before)

    def test_replay_failure_cleanup(self):
        rpn=NumericRPN();bound=rpn.filter_proposals
        def corrupt(*a,**k):
            boxes,scores=bound(*a,**k)
            scores[0][0]-=.01
            return boxes,scores
        rpn.filter_proposals=corrupt
        before=dict(rpn.__dict__);original=ops.batched_nms;target=HookTarget()
        b,l,c=example()
        with self.assertRaises(VerificationError):
            with TraceCapture(rpn,("0","1","pool"),GT,hooks=((target,lambda *a:None),)):
                rpn.filter_proposals(b[None],l,[(10,10)],list(c))
        self.assertEqual(rpn.__dict__,before);self.assertEqual(target.callbacks,{})
        self.assertIs(ops.batched_nms,original);self.assertFalse(_LOCK.locked())

    def test_concurrent_capture_refused_and_outer_restored(self):
        rpn=NumericRPN();before=dict(rpn.__dict__);b,l,c=example()
        _LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,"Concurrent"):
                with TraceCapture(rpn,("0","1","pool"),GT):
                    rpn.filter_proposals(b[None],l,[(10,10)],list(c))
            self.assertTrue(_LOCK.locked()) # Failed acquire must not release the owner's lock.
        finally:_LOCK.release()
        self.assertEqual(rpn.__dict__,before)

if __name__=="__main__":
    unittest.main(verbosity=2)

