"""Independent scalar binary32 geometry, actual-layer mapping and summary verification."""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import traceback
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
from torchvision.ops import boxes as ops

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
D4=ROOT/'week09/day04/outputs/full_val_stage_run01'
D5=ROOT/'week09/day05/outputs/full_val_nms_relations_run01'
KEYS=('0','1','pool')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def boolean(v):
    assert v in ('True','False');return v=='True'
def number(v,expected):
    if expected is None:assert v==''
    else:assert float(v)==expected,(v,expected)
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def scalar_iou32(a,b):
    """Independent area formula with binary32 rounding after every arithmetic operation."""
    aw=f32(a[2]-a[0]);ah=f32(a[3]-a[1]);bw=f32(b[2]-b[0]);bh=f32(b[3]-b[1])
    area_a=f32(aw*ah);area_b=f32(bw*bh)
    iw=max(0.,f32(min(a[2],b[2])-max(a[0],b[0])))
    ih=max(0.,f32(min(a[3],b[3])-max(a[1],b[1])))
    inter=f32(iw*ih);union=f32(f32(area_a+area_b)-inter)
    if union<=0:raise ValueError('Invalid positive-area source boxes')
    return f32(inter/union)

def verify(out):
    meta=load(out/'run_metadata.json');assert meta['status']=='completed' and meta['exit_code']==0
    assert meta['historical_sha256_before']==meta['historical_sha256_after']
    assert all(sha(p)==h for p,h in meta['historical_sha256_before'].items())
    assert all(sha(p)==h for p,h in meta['input_sha256'].items())
    cohort=read(D5/'common_risk_pairing.csv')
    assert len(cohort)==len({(c['stem'],c['gt_index']) for c in cohort})==297
    assert Counter(c['loss_pair_category'] for c in cohort)==Counter({'both_lost':10,'IR_only_lost':11,'Fusion_only_lost':20,'neither_lost':256})
    support=read(out/'per_gt_support.csv');layers=read(out/'per_layer_support.csv');trace=read(out/'good_candidate_trace.csv')
    S={(r['model'],r['stem'],int(r['gt_index'])):r for r in support}
    T={(r['model'],r['stem'],int(r['gt_index']),r['layer_key']):r for r in layers}
    C={(r['model'],r['stem'],int(r['gt_index']),int(r['nms_input_position_0based'])):r for r in trace}
    assert len(S)==len(support)==594;assert len(T)==len(layers)==1782;assert len(C)==len(trace)
    ledger={(r['model'],r['stem']):r for r in read(D4/'forward_ledger.csv')}
    groups=defaultdict(list)
    for c in cohort:groups[c['stem']].append(c)
    actual={};expected_trace=set();capture_reads=0;maxerror=0.
    for stem,cs in sorted(groups.items()):
      for model in ('IR','Fusion'):
        e=ledger[model,stem];path=D4/e['relative_raw'];assert sha(path)==e['raw_sha256']
        d=torch.load(path,map_location='cpu',weights_only=True);capture_reads+=1
        r=d['records'][0];pre=d['stages'][0]['score_filtered'];post=d['stages'][0]['layer_nms'];cat=d['catalogues'][0]
        gt=d['observed']['transform_gt_all'];assert tuple(r['layer_keys'])==KEYS
        b=r['nms_boxes'];levels=r['nms_levels'];keep=r['nms_indices'];ids=pre['ids']
        assert torch.equal(pre['boxes'],b) and torch.equal(pre['scores'],r['nms_scores'])
        for field in ('ids','boxes','scores'):assert torch.equal(post[field],pre[field][keep])
        ends=[];total=0
        for n in r['counts']:total+=n;ends.append(total)
        assert total==len(r['boxes'])
        for pos,flat in enumerate(ids.tolist()):
            index=bisect_right(ends,flat);assert index==int(levels[pos])
            assert cat[flat]['flat_index']==flat and cat[flat]['level_key']==KEYS[index]
            assert cat[flat]['candidate_id']==f"{r['prefix']}/{flat}"
        boxes=b.tolist()
        for c in cs:
            gi=int(c['gt_index']);key=model,stem,gi;row=S[key];g=gt[gi].tolist()
            assert json.loads(row['gt_xyxy'])==g and row['scene_id']==c['scene_id']
            values=[scalar_iou32(g,bb) for bb in boxes]
            good=[p for p,v in enumerate(values) if v>=.75]
            assert good,'Unexpected L=0; no dropping'
            level_positions={name:[p for p in good if int(levels[p])==i] for i,name in enumerate(KEYS)}
            L=sum(bool(pp) for pp in level_positions.values());assert L in (1,2,3)
            number(row['L'],L)
            assert json.loads(row['supported_layer_keys'])==[k for k in KEYS if level_positions[k]]
            for layer,pp in level_positions.items():
                lr=T[model,stem,gi,layer];expected_ids=[cat[int(ids[p])]['candidate_id'] for p in pp]
                number(row['count_'+layer],len(pp));number(lr['good_candidate_count'],len(pp));number(lr['L'],L)
                assert json.loads(lr['candidate_ids'])==expected_ids
                for pos in pp:
                    tkey=model,stem,gi,pos;expected_trace.add(tkey);tr=C[tkey];flat=int(ids[pos])
                    assert tr['candidate_id']==cat[flat]['candidate_id'] and int(tr['flat_index'])==flat and tr['layer_key']==layer
                    assert json.loads(tr['candidate_xyxy'])==boxes[pos] and json.loads(tr['gt_xyxy'])==g
                    err=abs(float(tr['iou_float32_cpu'])-values[pos]);maxerror=max(maxerror,err);assert err==0.
            lost=not any(values[int(p)]>=.75 for p in keep)
            assert boolean(row['nms_lost'])==lost and boolean(row['nms_survived'])==(not lost)
            assert lost==boolean(c[model+'_nms_lost']) and (not lost)==boolean(c[model+'_after_nms_covered'])
            actual[key]={'L':L,'lost':lost,'scene':c['scene_id']}
        print('RAW_L_VERIFIED',model,stem,'GT',len(cs),flush=True)
    assert expected_trace==set(C)
    assert set(actual)==set(S) and capture_reads==len(groups)*2==280
    provenance=load(out/'capture_provenance.json')
    assert len(provenance)==280 and len({(p['model'],p['stem']) for p in provenance})==280
    for p in provenance:
        e=ledger[p['model'],p['stem']]
        assert p['raw_sha256']==e['raw_sha256'] and p['relative_raw']==e['relative_raw'] and p['raw_deserializations']==1
    pairs=read(out/'all_common_gt_pairs.csv')
    P={(p['stem'],int(p['gt_index'])):p for p in pairs};assert len(P)==len(pairs)==297
    expected_asym=[]
    for c in cohort:
        stem,gi=c['stem'],int(c['gt_index']);p=P[stem,gi];a,b=actual['IR',stem,gi],actual['Fusion',stem,gi]
        number(p['IR_L'],a['L']);number(p['Fusion_L'],b['L'])
        assert boolean(p['IR_nms_lost'])==a['lost'] and boolean(p['Fusion_nms_lost'])==b['lost']
        category='both_lost' if a['lost'] and b['lost'] else 'IR_only_lost' if a['lost'] else 'Fusion_only_lost' if b['lost'] else 'neither_lost'
        assert p['loss_pair_category']==c['loss_pair_category']==category
        if a['lost']!=b['lost']:
            failed='IR' if a['lost'] else 'Fusion';surviving='Fusion' if failed=='IR' else 'IR'
            fail=actual[failed,stem,gi]['L'];survive=actual[surviving,stem,gi]['L'];delta=fail-survive
            assert p['failed_model']==failed and p['surviving_model']==surviving
            number(p['failed_L'],fail);number(p['surviving_L'],survive);number(p['delta_L'],delta)
            assert p['direction']==('fewer' if delta<0 else 'equal' if delta==0 else 'more')
            expected_asym.append(p)
        else:
            assert all(p[k]=='' for k in ('failed_model','surviving_model','failed_L','surviving_L','delta_L','direction'))
    asymmetric=read(out/'asymmetric_pairs.csv')
    assert expected_asym==asymmetric and len(asymmetric)==31
    assert Counter(p['failed_model'] for p in asymmetric)==Counter({'IR':11,'Fusion':20})
    summaries=read(out/'direction_summary.csv');assert len(summaries)==15
    for s in summaries:
        rr=[p for p in asymmetric if (s['scene_id']=='ALL' or p['scene_id']==s['scene_id']) and (s['failed_model']=='ALL' or p['failed_model']==s['failed_model'])]
        n=len(rr);number(s['asymmetric_GT'],n)
        for direction in ('fewer','equal','more'):
            count=sum(p['direction']==direction for p in rr)
            number(s[direction+'_GT'],count);number(s[direction+'_rate'],count/n if n else None)
        assert json.loads(s['delta_histogram'])=={str(v):sum(int(p['delta_L'])==v for p in rr) for v in (-2,-1,0,1,2)}
    background=read(out/'model_scene_L_summary.csv');assert len(background)==30
    for s in background:
        rr=[r for (model,stem,gi),r in actual.items() if model==s['model'] and r['L']==int(s['L']) and (s['scene_id']=='ALL' or r['scene']==s['scene_id'])]
        n=len(rr);lost=sum(r['lost'] for r in rr)
        number(s['risk_GT'],n);number(s['nms_lost_GT'],lost);number(s['nms_survived_GT'],n-lost)
        number(s['conditional_loss_rate'],lost/n if n else None)
    for model,lost_expected in (('IR',21),('Fusion',30)):
        rr=[s for s in background if s['scene_id']=='ALL' and s['model']==model]
        assert sum(int(s['risk_GT']) for s in rr)==297 and sum(int(s['nms_lost_GT']) for s in rr)==lost_expected
    return {'status':'passed','primary_metric':'L only','mother_GT':297,'model_GT_support_rows':594,'layer_rows':1782,
        'trace_rows_verified':len(trace),'raw_capture_reads':capture_reads,'asymmetric_GT':31,
        'directions':dict(Counter(p['direction'] for p in asymmetric)),
        'independent_method':'Scalar area with binary32 rounding at each operation; membership/counts/IDs/L/deltas exact; no IoU tolerance',
        'max_trace_iou_difference':maxerror,'historical_files_unchanged':len(meta['historical_sha256_before']),
        'model_forward_performed':False,'checkpoint_loaded':False,'NMS_executed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(HERE):raise ValueError('Day06 only')
    args.output.mkdir(exist_ok=False,parents=True)
    before={str(p):sha(p) for p in args.analysis_dir.rglob('*') if p.is_file()}
    code=1
    with (args.output/'run.log').open('x',encoding='utf-8') as log:
        class Tee:
            def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
            def flush(self):sys.__stdout__.flush();log.flush()
        from contextlib import redirect_stdout
        with redirect_stdout(Tee()):
            try:
                torch.set_num_threads(2)
                with patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forbidden')), \
                     patch.object(ops,'nms',side_effect=RuntimeError('NMS forbidden')),patch.object(ops,'batched_nms',side_effect=RuntimeError('NMS forbidden')):
                    info=verify(args.analysis_dir)
                assert before=={str(p):sha(p) for p in args.analysis_dir.rglob('*') if p.is_file()};code=0
            except BaseException:info={'status':'failed','error':traceback.format_exc()}
            info.update(exit_code=code,command=subprocess.list2cmdline(sys.orig_argv),verification_source_sha256=sha(Path(__file__)))
            with (args.output/'verification.json').open('x',encoding='utf-8') as f:json.dump(info,f,ensure_ascii=False,indent=2)
            print(json.dumps(info,ensure_ascii=False,indent=2))
    return code
if __name__=='__main__':sys.exit(main())
