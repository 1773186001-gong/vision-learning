"""Artificial tensors and temporary ledgers only, never a real detector/checkpoint."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import boxes as ops
import agnostic_capture as ac
from capture_ledger import Ledger,UnknownExecution,atomic_torch

class NumericRPN:
    filter_proposals=RegionProposalNetwork.filter_proposals
    _get_top_n_idx=RegionProposalNetwork._get_top_n_idx
    def __init__(self):self.min_size=.001;self.score_thresh=.05;self.nms_thresh=.7
    def pre_nms_top_n(self):return 2
    def post_nms_top_n(self):return 3

def values():
    b=torch.tensor([[2.,2.,8.,8.],[0.,0.,1.,1.],[8.,8.,10.,10.],
        [3.,3.,9.,9.],[0.,8.,1.,9.],[8.,0.,9.,1.],
        [1.,2.,7.,8.],[0.,4.,1.,5.],[9.,4.,10.,5.]])
    return b,torch.logit(torch.tensor([.8,.7,.6,.9,.7,.6,.95,.7,.6]))
def execute(gt,rpn=None,hooks=()):
    rpn=rpn or NumericRPN();b,l=values()
    with ac.AgnosticCapture(rpn,('0','1','pool'),gt,'synthetic/IR/image',hooks) as cap:
        output=rpn.filter_proposals(b[None],l,[(10,10)],[3,3,3])
        assert output is cap.returned_reference
    return cap,output

class FullCaptureTests(unittest.TestCase):
    def test_multi_gt_and_empty_gt_without_placeholder(self):
        for gt in (torch.tensor([[2.,2.,8.,8.],[0.,0.,1.,1.]]),torch.empty((0,4))):
            cap,out=execute(gt);self.assertTrue(torch.equal(cap.records[0]['gt_all'],gt))
            self.assertEqual(tuple(cap.states[0]),ac.STAGES);self.assertEqual(len(cap.states[0]['pre_nms_topn']['ids']),6)
            self.assertEqual(len(out[0][0]),3);self.assertNotIn('gt',cap.records[0])
    def test_original_return_and_normal_hooks_attributes_restored(self):
        rpn=NumericRPN();original=rpn.filter_proposals;rpn.filter_proposals=original;before=dict(rpn.__dict__)
        module=torch.nn.Identity();old_hooks=dict(module._forward_hooks);nms=ops.batched_nms
        cap,_=execute(torch.empty((0,4)),rpn,((module,lambda *a:None),))
        self.assertEqual(rpn.__dict__,before);self.assertEqual(dict(module._forward_hooks),old_hooks)
        self.assertIs(ops.batched_nms,nms);self.assertFalse(ac.LOCK.locked());self.assertFalse(cap.active)
    def test_configuration_error_releases_lock_then_normal_capture(self):
        rpn=NumericRPN();before=dict(rpn.__dict__)
        with patch.object(rpn,'pre_nms_top_n',side_effect=RuntimeError('config')):
            with self.assertRaisesRegex(RuntimeError,'config'):execute(torch.empty((0,4)),rpn)
        self.assertEqual(rpn.__dict__,before);self.assertFalse(ac.LOCK.locked());execute(torch.empty((0,4)),rpn)
    def test_top_reference_error_releases_lock(self):
        class Fault(NumericRPN):
            def __getattribute__(self,name):
                if name=='_get_top_n_idx' and self.__dict__.get('fail'):raise RuntimeError('lookup')
                return super().__getattribute__(name)
        rpn=Fault();rpn.fail=True
        with self.assertRaisesRegex(RuntimeError,'lookup'):execute(torch.empty((0,4)),rpn)
        self.assertFalse(ac.LOCK.locked());rpn.fail=False;execute(torch.empty((0,4)),rpn)
    def test_original_nms_exception_restores_everything(self):
        rpn=NumericRPN();before=dict(rpn.__dict__);module=torch.nn.Identity();old=dict(module._forward_hooks)
        with patch.object(ops,'batched_nms',side_effect=RuntimeError('nms')) as fake:
            with self.assertRaisesRegex(RuntimeError,'nms'):execute(torch.empty((0,4)),rpn,((module,lambda *a:None),))
            self.assertIs(ops.batched_nms,fake)
        self.assertEqual(rpn.__dict__,before);self.assertEqual(dict(module._forward_hooks),old);self.assertFalse(ac.LOCK.locked())
    def test_replay_exception_preserves_original_record_and_restores(self):
        rpn=NumericRPN();b,l=values();before=dict(rpn.__dict__)
        cap=ac.AgnosticCapture(rpn,('0','1','pool'),torch.empty((0,4)),'synthetic')
        with patch.object(ac,'replay',side_effect=ValueError('replay')):
            with self.assertRaisesRegex(ValueError,'replay'):
                with cap:rpn.filter_proposals(b[None],l,[(10,10)],[3,3,3])
        self.assertEqual(len(cap.records),1);self.assertEqual(rpn.__dict__,before);self.assertFalse(ac.LOCK.locked())
    def test_body_exception_and_partial_hook_registration_cleanup(self):
        rpn=NumericRPN();before=dict(rpn.__dict__)
        with self.assertRaisesRegex(RuntimeError,'body'):
            with ac.AgnosticCapture(rpn,('0','1','pool'),torch.empty((0,4)),'synthetic'):raise RuntimeError('body')
        self.assertEqual(rpn.__dict__,before)
        module=torch.nn.Identity()
        class Bad:
            def register_forward_hook(self,*a):raise RuntimeError('hook')
        with self.assertRaisesRegex(RuntimeError,'hook'):execute(torch.empty((0,4)),rpn,((module,lambda *a:None),(Bad(),None)))
        self.assertEqual(dict(module._forward_hooks),{});self.assertFalse(ac.LOCK.locked())
    def test_concurrent_rejection_does_not_release_other_owner(self):
        ac.LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'Concurrent'):execute(torch.empty((0,4)))
            self.assertTrue(ac.LOCK.locked())
        finally:ac.LOCK.release()
        execute(torch.empty((0,4)))
    def test_atomic_commit_verified_skip_signature_and_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'chunk01/IR_example','signature');self.assertEqual(ledger.action(),'new')
            ledger.start({'forward_calls':1});self.assertRaises(UnknownExecution,ledger.action)
            ledger.commit({'signature':'signature','tensor':torch.ones(2)})
            self.assertEqual(ledger.action(),'recover_offline');ledger.accept({'passed':True});self.assertEqual(ledger.action(),'skip')
            self.assertRaises(RuntimeError,ledger.start,{})
            self.assertRaises(RuntimeError,Ledger(ledger.path,'changed').action)
            (ledger.path/'raw_capture.pth').write_bytes(b'corrupt');self.assertRaises(RuntimeError,ledger.action)
    def test_partial_serialization_is_unknown_not_retryable(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'IR_example','s');ledger.start({})
            with patch('capture_ledger.torch.save',side_effect=RuntimeError('disk')):
                self.assertRaisesRegex(RuntimeError,'disk',ledger.commit,{'signature':'s'})
            self.assertTrue((ledger.path/'raw_capture.pth.tmp').exists());self.assertFalse((ledger.path/'raw_capture.pth').exists())
            self.assertRaises(UnknownExecution,ledger.action)
    def test_block_restart_skips_verified_and_halts_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            first=Ledger(Path(td)/'block01/IR_a','s');first.start({});first.commit({'signature':'s'});first.accept({'passed':True})
            second=Ledger(Path(td)/'block01/Fusion_a','s');second.start({})
            self.assertEqual(first.action(),'skip');self.assertRaises(UnknownExecution,second.action)
            third=Ledger(Path(td)/'block02/IR_b','s');self.assertEqual(third.action(),'new')
    def test_captured_raw_can_be_recovered_without_forward(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'IR_example','s');ledger.start({});ledger.commit({'signature':'s','tensor':torch.ones(1)})
            restored=torch.load(ledger.path/'raw_capture.pth',weights_only=True)
            self.assertEqual(restored['signature'],'s');ledger.accept({'offline_recovered':True});self.assertEqual(ledger.action(),'skip')

if __name__=='__main__':unittest.main()
