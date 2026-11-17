"""Independent saved-coordinate/export verification. No model, checkpoint, image or NMS execution."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
sys.dont_write_bytecode=True
import torch
from torchvision.ops import box_iou

HERE=Path(__file__).resolve().parent
STAGES=('decoded','pre_nms_topn','clipped','small_box_filtered','score_filtered','layer_nms','post_nms_topn')
CONVENTIONS=('float32_cpu','float64_diagnostic')
def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def boolean(s):
    assert s in ('True','False');return s=='True'
def equal_number(actual,expected,atol=0.):
    if expected is None:assert actual==''
    else:assert abs(float(actual)-expected)<=atol,(actual,expected)
def ratio(a,b):return a/b if b else None
def scalar_iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union

def verify(source):
    meta=load(source/'run_metadata.json');assert meta['status']=='completed' and meta['exit_code']==0
    assert meta['historical_sha256_before']==meta['historical_sha256_after']
    assert all(sha(p)==h for p,h in meta['historical_sha256_before'].items())
    evidence=load(source/'reuse_evidence.json');rows=csv_rows(source/'per_gt_stage.csv');images=csv_rows(source/'per_image.csv')
    keyed={(r['model'],r['stem'],int(r['gt_index']),r['stage']):r for r in rows};assert len(keyed)==len(rows)
    assert len(images)==len(evidence['captures'])
    seen=set();max64error=0.
    for capture in evidence['captures']:
        path=Path(capture['raw_path']);assert sha(path)==capture['raw_sha256']
        d=torch.load(path,weights_only=True,map_location='cpu');gts=d['observed']['transform_gt_all'];states=d['stages'][0];catalogue=d['catalogues'][0]
        model,stem=capture['model'],capture['stem']
        im=next(i for i in images if (i['model'],i['stem'])==(model,stem))
        assert int(im['gt_count'])==len(gts)==capture['all_gt_count'];assert boolean(im['empty_gt'])==(len(gts)==0)
        assert json.loads(im['proposal_counts'])=={s:len(states[s]['ids']) for s in STAGES}
        for stage in STAGES:
            state=states[stage];b=state['boxes'];f32=box_iou(gts,b);blist=b.tolist()
            for gi,g in enumerate(gts.tolist()):
                key=(model,stem,gi,stage);r=keyed[key];seen.add(key)
                assert r['scene_id']==capture['scene_id'] and int(r['image_index'])==capture['image_index']
                assert json.loads(r['gt_xyxy'])==g and int(r['proposal_count'])==len(b)
                v64=[scalar_iou(g,bb) for bb in blist]
                for convention,values in ((CONVENTIONS[0],f32[gi].tolist()),(CONVENTIONS[1],v64)):
                    best=max(range(len(values)),key=lambda i:values[i]) if values else None
                    maximum=values[best] if best is not None else 0.;count=sum(v>=.75 for v in values)
                    error=abs(float(r['max_iou_'+convention])-maximum)
                    if convention==CONVENTIONS[0]:assert error==0.
                    else:max64error=max(max64error,error);assert error<=1e-12
                    assert int(r['count_iou_ge_075_'+convention])==count
                    assert boolean(r['covered_075_'+convention])==(count>0)
                    expected=catalogue[int(state['ids'][best])]['candidate_id'] if best is not None else ''
                    assert r['best_candidate_id_'+convention]==expected
    assert seen==set(keyed)
    bygt=defaultdict(dict)
    for r in rows:bygt[r['model'],r['stem'],int(r['gt_index'])][r['stage']]=r
    for ss in bygt.values():
        for convention in CONVENTIONS:
            prev=None
            for stage in STAGES:
                r=ss[stage];h=boolean(r['covered_075_'+convention]);c=int(r['count_iou_ge_075_'+convention]);m=float(r['max_iou_'+convention])
                assert (r['previous_covered_'+convention]=='') if prev is None else boolean(r['previous_covered_'+convention])==prev[0]
                assert boolean(r['coverage_lost_'+convention])==bool(prev and prev[0] and not h)
                assert boolean(r['coverage_gained_'+convention])==bool(prev and not prev[0] and h)
                assert boolean(r['partial_high_loss_coverage_kept_'+convention])==bool(prev and c<prev[1] and h)
                equal_number(r['max_iou_change_'+convention],m-prev[2] if prev else None)
                prev=h,c,m
    stage_summary=csv_rows(source/'stage_summary.csv')
    for s in stage_summary:
        scope=lambda r:s['scope']=='overall' or r['scene_id']==s['scene_id']
        rr=[r for r in rows if scope(r) and r['model']==s['model'] and r['stage']==s['stage']]
        ims=[i for i in images if scope(i) and i['model']==s['model']]
        conv=s['iou_convention'];n=sum(int(i['gt_count']) for i in ims);assert len(rr)==n
        covered=sum(boolean(r['covered_075_'+conv]) for r in rr)
        lost=sum(boolean(r['coverage_lost_'+conv]) for r in rr);gain=sum(boolean(r['coverage_gained_'+conv]) for r in rr)
        prev=sum(r['previous_covered_'+conv]=='True' for r in rr) if s['stage']!='decoded' else None
        for k,v in dict(images=len(ims),empty_gt_images=sum(boolean(i['empty_gt']) for i in ims),valid_gt=n,
            covered_gt=covered,coverage_lost_gt=lost,coverage_gained_gt=gain,
            previous_covered_gt=prev,coverage_rate=ratio(covered,n),
            lost_rate_all_gt=ratio(lost,n) if prev is not None else None,
            lost_rate_previous_covered=ratio(lost,prev) if prev is not None else None,
            partial_high_loss_coverage_kept_gt=sum(boolean(r['partial_high_loss_coverage_kept_'+conv]) for r in rr)).items():equal_number(s[k],v)
    trajectories=csv_rows(source/'gt_coverage_sequences.csv')
    assert len(trajectories)==len(bygt)*2
    for t in trajectories:
        ss=bygt[t['model'],t['stem'],int(t['gt_index'])];conv=t['iou_convention']
        hs=[boolean(ss[s]['covered_075_'+conv]) for s in STAGES]
        losses=[s for i,s in enumerate(STAGES) if i and hs[i-1] and not hs[i]]
        gains=[s for i,s in enumerate(STAGES) if i and not hs[i-1] and hs[i]]
        first=next((s for s,h in zip(STAGES,hs) if h),'')
        assert t['coverage_sequence']==''.join(str(int(v)) for v in hs)
        assert boolean(t['never_covered'])==(not any(hs));assert t['first_covered_stage']==first
        assert t['first_all_loss_stage']==(losses[0] if losses else '')
        assert json.loads(t['loss_stages'])==losses and json.loads(t['gain_stages'])==gains
        assert boolean(t['first_coverage_after_decoded'])==bool(first and first!='decoded')
        assert boolean(t['recovered_after_loss'])==any(STAGES.index(g)>STAGES.index(l) for g in gains for l in losses)
    assert [t for t in trajectories if boolean(t['never_covered'])]==csv_rows(source/'never_covered_gt.csv')
    for n in csv_rows(source/'never_covered_summary.csv'):
        ts=[t for t in trajectories if t['model']==n['model'] and t['iou_convention']==n['iou_convention'] and (n['scope']=='overall' or t['scene_id']==n['scene_id'])]
        count=sum(boolean(t['never_covered']) for t in ts);equal_number(n['never_covered_gt'],count)
        equal_number(n['valid_gt'],len(ts));equal_number(n['never_covered_rate'],ratio(count,len(ts)))
    paired=csv_rows(source/'paired_per_gt.csv')
    assert len(paired)==len(rows)
    for p in paired:
        ir=keyed['IR',p['stem'],int(p['gt_index']),p['stage']];fu=keyed['Fusion',p['stem'],int(p['gt_index']),p['stage']]
        a=boolean(ir['covered_075_'+p['iou_convention']]);b=boolean(fu['covered_075_'+p['iou_convention']])
        assert boolean(p['IR_covered'])==a and boolean(p['Fusion_covered'])==b
        assert p['pair_category']==('both' if a and b else 'IR_only' if a else 'Fusion_only' if b else 'neither')
    for p in csv_rows(source/'paired_summary.csv'):
        pp=[r for r in paired if r['stage']==p['stage'] and r['iou_convention']==p['iou_convention'] and (p['scope']=='overall' or r['scene_id']==p['scene_id'])]
        equal_number(p['valid_gt'],len(pp))
        for cat in ('both','IR_only','Fusion_only','neither'):
            n=sum(r['pair_category']==cat for r in pp);equal_number(p[cat],n);equal_number(p[cat+'_rate'],ratio(n,len(pp)))
    previous={}
    for n in csv_rows(source/'net_difference.csv'):
        ss=[s for s in stage_summary if (s['scope'],s['scene_id'],s['stage'],s['iou_convention'])==(n['scope'],n['scene_id'],n['stage'],n['iou_convention'])]
        ir=next(s for s in ss if s['model']=='IR');fu=next(s for s in ss if s['model']=='Fusion')
        difference=int(ir['covered_gt'])-int(fu['covered_gt']);key=n['scope'],n['scene_id'],n['iou_convention']
        equal_number(n['IR_minus_Fusion'],difference)
        if key in previous:
            delta=difference-previous[key];loss=int(fu['coverage_lost_gt'])-int(ir['coverage_lost_gt']);gain=int(ir['coverage_gained_gt'])-int(fu['coverage_gained_gt'])
            equal_number(n['net_change_from_previous'],delta);equal_number(n['Fusion_losses_minus_IR_losses'],loss);equal_number(n['IR_gains_minus_Fusion_gains'],gain)
            assert delta==loss+gain
        else:assert n['net_change_from_previous']==''
        previous[key]=difference
    historic=load(source/'verification.json')
    assert all(c['actual_difference']==0 for c in historic['day01_comparisons'])
    return {'status':'passed','scope':'six_saved_images_all_gt_pilot_not_full_validation','per_gt_stage_rows_verified':len(rows),
        'model_gt_items':len(bygt),'stage_summary_rows_verified':len(stage_summary),'max_independent_float64_error':max64error,
        'float64_atol':1e-12,'historical_files_unchanged':len(meta['historical_sha256_before']),
        'no_model_forward':True,'no_checkpoint_load':True,'no_nms_execution':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(HERE):raise ValueError('Day04 output only')
    args.output.mkdir(parents=True,exist_ok=False);code=1
    before={str(p):sha(p) for p in args.capture_dir.rglob('*') if p.is_file()}
    try:
        info=verify(args.capture_dir)
        assert before=={str(p):sha(p) for p in args.capture_dir.rglob('*') if p.is_file()}
        code=0
    except BaseException:info={'status':'failed','error':traceback.format_exc()}
    info.update(command=subprocess.list2cmdline(sys.orig_argv),exit_code=code,verification_source_sha256=sha(Path(__file__)))
    text=json.dumps(info,ensure_ascii=False,indent=2)
    with (args.output/'verification.json').open('x',encoding='utf-8') as f:f.write(text+'\n')
    with (args.output/'run.log').open('x',encoding='utf-8') as f:f.write(text+'\n')
    print(text);return code
if __name__=='__main__':sys.exit(main())
