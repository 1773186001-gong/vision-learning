"""Exact historical Day04 trace copied; source SHA recorded in package manifest."""
from fusion_model import ir
import torch

def trace(p, gt):
    """Instrument ir.match verbatim decisions; verify complete event sequence per image."""
    used=set(); owner={}; rows=[]
    for idx in torch.argsort(p['scores'], descending=True, stable=True).tolist():
        score=float(p['scores'][idx])
        overlaps=ir.box_iou(p['boxes'][idx:idx+1],gt)[0]
        selected=score>=0.5; hit=None
        if selected and len(gt):
            available=overlaps.clone()
            for j in used: available[j]=-1
            value,j=available.max(0)
            if float(value)>=0.5:
                hit=int(j); used.add(hit); owner[hit]=idx
        rows.append(dict(prediction_id=idx,score=score,box=p['boxes'][idx].tolist(),
                         selected=selected,status=('TP' if hit is not None else 'FP') if selected else 'below_score',
                         matched_gt=hit,ious=overlaps.tolist(),
                         max_iou=float(overlaps.max()) if len(gt) else None))
    events,total=ir.match([p],[gt],0.5,0.5)
    assert events==[(r['score'],int(r['status']=='TP')) for r in rows if r['selected']]
    for r in rows:
        r.update(fp_type=None,duplicate_links=[],reason=None,no_gt_fp=False)
        if r['status']!='FP': continue
        links=[dict(gt_id=j,tp_prediction_id=owner[j],fp_gt_iou=r['ious'][j],
                    tp_gt_iou=float(ir.box_iou(p['boxes'][owner[j]:owner[j]+1],gt)[0,j]))
               for j in sorted(owner) if r['ious'][j]>=0.5]
        if links: category='duplicate'
        elif not len(gt): category='low_overlap'; r['no_gt_fp']=True
        elif .1<=r['max_iou']<.5: category='localization'
        elif r['max_iou']<.1: category='low_overlap'
        else: category='other'; r['reason']='maxIoU >= .5 but no finally matched GT overlap; inspect ordering'
        r['fp_type']=category; r['duplicate_links']=links
    return rows

