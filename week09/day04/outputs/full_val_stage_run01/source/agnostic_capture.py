"""GT-independent, single-image read-only RPN capture. No fake/sentinel GT."""
from contextlib import ExitStack
import inspect
import threading
import torch
from torchvision.models.detection import rpn as rpn_module
from torchvision.ops import boxes as box_ops

LOCK=threading.Lock()
MISSING=object()
STAGES=('decoded','pre_nms_topn','clipped','small_box_filtered','score_filtered','layer_nms','post_nms_topn')

def exact(a,b,label='tensor'):
    if a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device or not torch.equal(a,b):raise ValueError(label+' differs')

def patch_attribute(stack,obj,key,value):
    old=obj.__dict__.get(key,MISSING)
    def restore():
        if old is MISSING:delattr(obj,key)
        else:setattr(obj,key,old)
    stack.callback(restore);setattr(obj,key,value)

def replay(r):
    b=r['boxes'];gt=r['gt_all'];h,w=r['image_size'];cfg=r['config'];keys=r['layer_keys'];counts=r['counts']
    if b.ndim!=2 or b.shape[1]!=4 or not bool(torch.isfinite(b).all()) or bool((b[:,2:]<=b[:,:2]).any()):raise ValueError('Invalid decoded boxes')
    if gt.ndim!=2 or gt.shape[1]!=4 or not bool(torch.isfinite(gt).all()) or bool((gt[:,2:]<=gt[:,:2]).any()):raise ValueError('Invalid all-GT')
    if bool((gt<0).any()) or bool((gt[:,[0,2]]>w).any()) or bool((gt[:,[1,3]]>h).any()):raise ValueError('GT outside image')
    if len(keys)!=len(counts) or sum(counts)!=len(b) or r['logits'].numel()!=len(b):raise ValueError('Level/count mismatch')
    levels=torch.repeat_interleave(torch.arange(len(keys),device=b.device),torch.tensor(counts,device=b.device))
    cat=[];offset=0
    for level,(key,n) in enumerate(zip(keys,counts)):
        for i in range(n):cat.append(dict(candidate_id=f"{r['prefix']}/{offset+i}",flat_index=offset+i,level_key=key,level_index=level,index_within_level=i))
        offset+=n
    ids=torch.arange(len(b),device=b.device);b=b.clone();scores=torch.sigmoid(r['logits'].reshape(-1));states={}
    def snapshot(name):states[name]={'name':name,'ids':ids.clone(),'boxes':b.clone(),'scores':scores.clone()}
    def select(idx,name):
        nonlocal ids,b,scores
        if idx.ndim!=1 or idx.dtype!=torch.int64 or len(idx.unique())!=len(idx) or bool(((idx<0)|(idx>=len(ids))).any()):raise ValueError('Invalid actual indices')
        ids,b,scores=ids[idx],b[idx],scores[idx];snapshot(name)
    snapshot('decoded');top=r['top_indices']
    select(top,'pre_nms_topn')
    if any(int((levels[ids]==i).sum())!=min(cfg['pre_nms'],n) for i,n in enumerate(counts)):raise ValueError('Per-level pre budget')
    scores=torch.sigmoid(r['logits'].reshape(-1)[top]);snapshot('pre_nms_topn')
    b=box_ops.clip_boxes_to_image(b,r['image_size']);snapshot('clipped')
    select(box_ops.remove_small_boxes(b,cfg['min_size']),'small_box_filtered')
    select(torch.where(scores>=cfg['score_thresh'])[0],'score_filtered')
    exact(b,r['nms_boxes'],'NMS input boxes');exact(scores,r['nms_scores'],'NMS input scores');exact(levels[ids],r['nms_levels'],'NMS levels')
    select(r['nms_indices'],'layer_nms')
    if len(scores) and bool((scores[1:]>scores[:-1]).any()):raise ValueError('NMS score order')
    select(torch.arange(min(cfg['post_nms'],len(ids)),device=b.device),'post_nms_topn')
    exact(b,r['final_boxes'],'final boxes/order');exact(scores,r['final_scores'],'final scores')
    return states,cat

class AgnosticCapture:
    def __init__(self,rpn,layer_keys,gt_all,prefix,hooks=()):
        self.rpn=rpn;self.keys=tuple(layer_keys);self.gt=gt_all;self.prefix=prefix;self.hooks=hooks
        self.records=[];self.states=[];self.catalogues=[];self.active=False;self.returned_reference=None
    def __enter__(self):
        if self.active:raise RuntimeError('Already active')
        stack=ExitStack()
        try:
            for module,fn in self.hooks:stack.callback(module.register_forward_hook(fn).remove)
            original=self.rpn.filter_proposals
            signature=inspect.signature(rpn_module.RegionProposalNetwork.filter_proposals)
            def wrapped(*args,**kwargs):
                argsmap=signature.bind(self.rpn,*args,**kwargs).arguments
                p,logits=argsmap['proposals'],argsmap['objectness']
                shapes,counts=argsmap['image_shapes'],argsmap['num_anchors_per_level']
                if len(shapes)!=1 or p.shape[0]!=1:raise ValueError('Single-image capture only')
                if not LOCK.acquire(blocking=False):raise RuntimeError('Concurrent capture rejected')
                try:
                    owner=threading.get_ident();tops=[];nms_calls=[]
                    cfg={'pre_nms':self.rpn.pre_nms_top_n(),'post_nms':self.rpn.post_nms_top_n(),
                        'min_size':self.rpn.min_size,'score_thresh':self.rpn.score_thresh,'nms_thresh':self.rpn.nms_thresh}
                    top_original=self.rpn._get_top_n_idx;nms_original=rpn_module.box_ops.batched_nms
                    def top(*a,**k):
                        if threading.get_ident()!=owner:raise RuntimeError('Other thread')
                        result=top_original(*a,**k);tops.append(result.detach().clone());return result
                    def nms(b,s,l,t):
                        if threading.get_ident()!=owner:raise RuntimeError('Other thread')
                        result=nms_original(b,s,l,t);nms_calls.append((b.detach().clone(),s.detach().clone(),l.detach().clone(),result.detach().clone()));return result
                    boxes=p.detach().clone();raw_logits=logits.detach().clone()
                    with ExitStack() as narrow:
                        patch_attribute(narrow,self.rpn,'_get_top_n_idx',top)
                        patch_attribute(narrow,rpn_module.box_ops,'batched_nms',nms)
                        output=original(*args,**kwargs)
                    self.returned_reference=output
                    if len(tops)!=1 or len(nms_calls)!=1 or len(output)!=2:raise ValueError('Unexpected original calls')
                    b,s,l,keep=nms_calls[0]
                    record=dict(boxes=boxes[0],logits=raw_logits.reshape(-1),counts=tuple(counts),image_size=tuple(shapes[0]),
                        layer_keys=self.keys,gt_all=self.gt.detach().clone(),config=cfg,top_indices=tops[0][0],
                        nms_boxes=b,nms_scores=s,nms_levels=l,nms_indices=keep,
                        final_boxes=output[0][0].detach().clone(),final_scores=output[1][0].detach().clone(),prefix=self.prefix)
                    self.records.append(record)
                    states,cat=replay(record);self.states.append(states);self.catalogues.append(cat)
                    return output
                finally:LOCK.release()
            patch_attribute(stack,self.rpn,'filter_proposals',wrapped)
            self.stack=stack;self.active=True;return self
        except BaseException:stack.close();raise
    def __exit__(self,*exc):
        try:return self.stack.__exit__(*exc)
        finally:self.active=False
