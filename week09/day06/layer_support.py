"""One prespecified measure: number of actual feature levels with IoU>=.75 support."""
from collections import Counter
import json
import torch
from torchvision.ops import box_iou

KEYS=('0','1','pool')
SCENES=('03','08','13','17')

def counts_by_level(good_mask,levels,keys=KEYS):
    if tuple(keys)!=KEYS:raise ValueError('Unexpected real feature keys')
    if good_mask.dtype!=torch.bool or levels.dtype!=torch.int64 or good_mask.shape!=levels.shape:raise ValueError('Mask/level shape or type')
    if bool(((levels<0)|(levels>=len(keys))).any()):raise ValueError('Invalid actual level index')
    return {key:int((good_mask & (levels==index)).sum()) for index,key in enumerate(keys)}

def measure(gt,boxes,levels,ids,catalogue,keys=KEYS):
    if gt.dtype!=torch.float32 or boxes.dtype!=torch.float32 or gt.device.type!='cpu' or boxes.device.type!='cpu':raise ValueError('CPU float32 required')
    if gt.shape!=(4,) or boxes.shape!=(len(ids),4):raise ValueError('Invalid geometry shape')
    if not bool(torch.isfinite(gt).all()) or not bool(torch.isfinite(boxes).all()):raise ValueError('Nonfinite geometry')
    if bool((gt[2:]<=gt[:2]).any()) or bool((boxes[:,2:]<=boxes[:,:2]).any()):raise ValueError('Nonpositive box')
    if ids.dtype!=torch.int64 or len(ids.unique())!=len(ids):raise ValueError('Invalid original IDs')
    values=box_iou(gt[None],boxes)[0]
    good=values>=.75
    counts=counts_by_level(good,levels,keys)
    identities={key:[catalogue[int(ids[p])]['candidate_id'] for p in torch.where(good & (levels==i))[0].tolist()] for i,key in enumerate(keys)}
    L=sum(n>0 for n in counts.values())
    if L==0:raise ValueError('L=0 in common-risk cohort; inspect input/convention, never drop')
    return L,counts,identities,values

def make_pairs(support,cohort):
    lookup={(r['model'],r['stem'],r['gt_index']):r for r in support}
    if len(lookup)!=len(support):raise ValueError('Duplicate model/GT support')
    pairs=[]
    for c in cohort:
        stem,gi=c['stem'],int(c['gt_index'])
        ir,fu=lookup['IR',stem,gi],lookup['Fusion',stem,gi]
        if ir['scene_id']!=fu['scene_id'] or ir['gt_xyxy']!=fu['gt_xyxy']:raise ValueError('Paired GT differs')
        il,fl=ir['nms_lost'],fu['nms_lost']
        category='both_lost' if il and fl else 'IR_only_lost' if il else 'Fusion_only_lost' if fl else 'neither_lost'
        failed='IR' if il and not fl else 'Fusion' if fl and not il else None
        survivor='Fusion' if failed=='IR' else 'IR' if failed=='Fusion' else None
        failure_L=ir['L'] if failed=='IR' else fu['L'] if failed=='Fusion' else None
        survivor_L=fu['L'] if failed=='IR' else ir['L'] if failed=='Fusion' else None
        delta=failure_L-survivor_L if failed else None
        pairs.append(dict(stem=stem,gt_index=gi,scene_id=ir['scene_id'],IR_L=ir['L'],Fusion_L=fu['L'],
            IR_nms_lost=il,Fusion_nms_lost=fl,loss_pair_category=category,
            failed_model=failed,surviving_model=survivor,failed_L=failure_L,surviving_L=survivor_L,
            delta_L=delta,direction='fewer' if delta is not None and delta<0 else 'equal' if delta==0 else 'more' if delta is not None else None))
    return pairs

def direction_summary(pairs):
    asym=[p for p in pairs if p['failed_model'] is not None]
    rows=[]
    for scene in ('ALL',)+SCENES:
      for failed in ('ALL','IR','Fusion'):
        rr=[p for p in asym if (scene=='ALL' or p['scene_id']==scene) and (failed=='ALL' or p['failed_model']==failed)]
        n=len(rr)
        counts={d:sum(p['direction']==d for p in rr) for d in ('fewer','equal','more')}
        rows.append(dict(scene_id=scene,failed_model=failed,asymmetric_GT=n,
            **{d+'_GT':v for d,v in counts.items()},**{d+'_rate':v/n if n else None for d,v in counts.items()},
            delta_histogram=json.dumps({str(d):sum(p['delta_L']==d for p in rr) for d in (-2,-1,0,1,2)})))
    return rows

def background_summary(support):
    rows=[]
    for scene in ('ALL',)+SCENES:
      for model in ('IR','Fusion'):
       for L in (1,2,3):
        rr=[r for r in support if r['model']==model and r['L']==L and (scene=='ALL' or r['scene_id']==scene)]
        n=len(rr);lost=sum(r['nms_lost'] for r in rr)
        rows.append(dict(scene_id=scene,model=model,L=L,risk_GT=n,nms_lost_GT=lost,
            nms_survived_GT=n-lost,conditional_loss_rate=lost/n if n else None,
            denominator_note='common-risk GT within this model/scene/L; no causal interpretation'))
    return rows
