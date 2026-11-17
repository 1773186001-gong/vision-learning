"""Independent scalar-area/indices/GT-summary audit; no model or NMS execution."""
import argparse
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

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'week09/day04/outputs/full_val_stage_run01'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def boolean(v):
    assert v in ('True','False');return v=='True'
def equal(v,expected,tol=0.):
    if expected is None:assert v==''
    else:assert abs(float(v)-expected)<=tol,(v,expected)
def scalar_iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
TH32=struct.unpack('f',struct.pack('f',.7))[0]

def verify(out):
    meta=load(out/'run_metadata.json');assert meta['status']=='completed' and meta['exit_code']==0
    assert meta['historical_sha256_before']==meta['historical_sha256_after']
    assert all(sha(p)==h for p,h in meta['historical_sha256_before'].items())
    manifest=load(SOURCE/'artifact_manifest.json');assert all(sha(SOURCE/p)==h for p,h in manifest.items())
    source=defaultdict(dict)
    for r in read(SOURCE/'per_gt_stage.csv'):source[r['model'],r['stem'],int(r['gt_index'])][r['stage']]=r
    selected={k for k,ss in source.items() if ss['score_filtered']['covered_075_float32_cpu']=='True' and ss['layer_nms']['covered_075_float32_cpu']=='False'}
    events=read(out/'gt_events.csv');ev={(e['model'],e['stem'],int(e['gt_index'])):e for e in events}
    assert len(ev)==len(events)==58 and set(ev)==selected
    assert Counter(k[0] for k in ev)==Counter({'IR':24,'Fusion':34})
    assert len({k[1:] for k in ev})==48
    deleted=read(out/'deleted_good_candidates.csv');relations=read(out/'possible_related_retained.csv');audit=read(out/'pair_audit.csv')
    diagnostic=read(out/'diagnostic_only_relations.csv');inventory=read(out/'retained_inventory.csv');overlaps=read(out/'gt_overlap_audit.csv')
    assert len({(r['event_id'],r['deleted_candidate_id']) for r in deleted})==len(deleted)
    assert len({(r['event_id'],r['deleted_candidate_id'],r['retained_candidate_id']) for r in audit})==len(audit)
    assert [r for r in audit if boolean(r['primary_possible_relation'])]==relations
    assert [r for r in audit if boolean(r['diagnostic_only_relation'])]==diagnostic
    assert [r for r in audit if boolean(r['near_threshold_diagnostic_only']) or boolean(r['relation_predicate_disagreement']) or (boolean(r['eligible_same_layer_score']) and boolean(r['score_tie']))]==read(out/'numerical_audit.csv')
    ledger={(r['model'],r['stem']):r for r in read(SOURCE/'forward_ledger.csv')}
    groups=defaultdict(list)
    for key in selected:groups[key[:2]].append(key)
    expected_good=set();expected_overlap_keys=set();maxerror=0.;near_count=0;disagree_count=0;load_count=0
    for model,stem in sorted(groups):
        entry=ledger[model,stem];raw=SOURCE/entry['relative_raw'];assert sha(raw)==entry['raw_sha256']
        d=torch.load(raw,map_location='cpu',weights_only=True);load_count+=1
        r=d['records'][0];states=d['stages'][0];pre=states['score_filtered'];keep=r['nms_indices']
        b,s,l=r['nms_boxes'],r['nms_scores'],r['nms_levels'];gt=d['observed']['transform_gt_all'];cat=d['catalogues'][0]
        assert torch.equal(pre['boxes'],b) and torch.equal(pre['scores'],s)
        for field in ('ids','boxes','scores'):assert torch.equal(states['layer_nms'][field],pre[field][keep])
        for field in ('ids','boxes','scores'):assert torch.equal(states['post_nms_topn'][field],states['layer_nms'][field][:150])
        assert torch.equal(states['post_nms_topn']['boxes'],r['final_boxes'])
        assert torch.equal(d['observed']['rpn_forward_boxes'],r['final_boxes'])
        offsets=l.to(b)*(b.max()+1);shifted=b+offsets[:,None]
        positions={int(flat):i for i,flat in enumerate(pre['ids'].tolist())};kept={int(i):k for k,i in enumerate(keep.tolist())}
        inv=[row for row in inventory if (row['model'],row['stem'])==(model,stem)]
        assert [int(row['flat_index']) for row in inv]==pre['ids'][keep].tolist()
        for k,row in enumerate(inv):
            j=int(keep[k]);assert int(row['nms_input_position_0based'])==j and int(row['nms_output_position_0based'])==k
            assert json.loads(row['xyxy'])==b[j].tolist() and float(row['score'])==float(s[j])
            assert row['candidate_id']==cat[int(pre['ids'][j])]['candidate_id']
            assert boolean(row['passes_final_budget'])==(k<len(r['final_boxes']))
        g32=ops.box_iou(gt,b)
        for _,_,gi in sorted(groups[model,stem]):
            event=ev[model,stem,gi];eid=event['event_id'];good=torch.where(g32[gi]>=.75)[0]
            assert len(good)>0 and not any(int(i) in kept for i in good)
            assert not bool((g32[gi,keep]>=.75).any())
            assert event['gt_xyxy']==source[model,stem,gi]['score_filtered']['gt_xyxy']
            assert json.loads(event['gt_xyxy'])==gt[gi].tolist()
            drows=[row for row in deleted if row['event_id']==eid];arows=[row for row in audit if row['event_id']==eid]
            assert {int(row['deleted_nms_input_position_0based']) for row in drows}==set(good.tolist())
            assert len(arows)==len(good)*len(keep)
            pair32=ops.box_iou(b[good],b[keep]);off32=ops.box_iou(shifted[good],shifted[keep])
            qindex={int(i):q for q,i in enumerate(good.tolist())};seenpairs=set();related_j=set()
            for row in arows:
                i,j=int(row['deleted_nms_input_position_0based']),int(row['retained_nms_input_position_0based'])
                assert i in qindex and j in kept and (i,j) not in seenpairs;seenpairs.add((i,j))
                for role,pos in (('deleted',i),('retained',j)):
                    flat=int(pre['ids'][pos]);assert int(row[role+'_flat_index'])==flat
                    assert row[role+'_candidate_id']==cat[flat]['candidate_id']
                    assert json.loads(row[role+'_xyxy'])==b[pos].tolist()
                    assert float(row[role+'_score'])==float(s[pos])
                    assert row[role+'_level_key']==r['layer_keys'][int(l[pos])]
                    equal(row[role+'_gt_iou_float32_cpu'],float(g32[gi,pos]))
                    g64=scalar_iou(gt[gi].tolist(),b[pos].tolist())
                    error=abs(float(row[role+'_gt_iou_float64_diagnostic'])-g64);maxerror=max(maxerror,error);assert error<=1e-12
                    equal(row[role+'_nms_output_position_0based'],kept.get(pos))
                    assert boolean(row[role+'_passes_final_budget'])==(pos in kept and kept[pos]<len(r['final_boxes']))
                q,k=qindex[i],kept[j];v32=float(pair32[q,k]);v64=scalar_iou(b[i].tolist(),b[j].tolist());offset32=float(off32[q,k])
                equal(row['mutual_iou_float32_cpu'],v32);equal(row['mutual_iou_shifted_float32_diagnostic'],offset32)
                error=abs(float(row['mutual_iou_float64_diagnostic'])-v64);maxerror=max(maxerror,error);assert error<=1e-12
                same=bool(l[i]==l[j]);higher=bool(s[j]>s[i]);tie=bool(s[j]==s[i]);eligible=same and (higher or tie)
                booleans=(v32>TH32,v64>.7,offset32>TH32)
                primary=eligible and booleans[0];diag=eligible and not booleans[0] and any(booleans[1:])
                near=eligible and min(abs(v32-TH32),abs(v64-.7),abs(offset32-TH32))<=1e-6
                disagreement=eligible and len(set(booleans))>1
                assert boolean(row['same_level'])==same and boolean(row['score_tie'])==tie
                assert row['score_relation']==('equal' if tie else 'higher' if higher else 'lower')
                for col,value in {'eligible_same_layer_score':eligible,'raw_float32_strict_overlap':booleans[0],
                        'raw_float64_strict_overlap':booleans[1],'shifted_float32_strict_overlap':booleans[2],
                        'primary_possible_relation':primary,'diagnostic_only_relation':diag,
                        'near_threshold_diagnostic_only':near,'relation_predicate_disagreement':disagreement}.items():assert boolean(row[col])==value
                near_count+=near;disagree_count+=disagreement
                others=[t for t in range(len(gt)) if t!=gi];maximum=max((float(g32[t,j]) for t in others),default=None)
                equal(row['retained_other_gt_max_iou_float32'],maximum)
                maximum64=max((scalar_iou(gt[t].tolist(),b[j].tolist()) for t in others),default=None)
                equal(row['retained_other_gt_max_iou_float64'],maximum64,1e-12)
                assert json.loads(row['retained_max_other_gt_indices'])==[t for t in others if float(g32[t,j])==maximum]
                assert boolean(row['retained_other_gt_overlap_larger'])==(maximum is not None and maximum>float(g32[gi,j]))
                assert boolean(row['retained_specified_gt_iou_worse'])==(float(g32[gi,j])<float(g32[gi,i]))
                if primary or diag:related_j.add(j)
            for dr in drows:
                candidate=dr['deleted_candidate_id'];expected_good.add((eid,candidate))
                rs=[row for row in arows if row['deleted_candidate_id']==candidate]
                n=sum(boolean(row['primary_possible_relation']) for row in rs)
                equal(dr['primary_relation_count'],n);equal(dr['diagnostic_only_relation_count'],sum(boolean(row['diagnostic_only_relation']) for row in rs))
                assert boolean(dr['unexplained_primary'])==(n==0)
                assert dr['candidate_gt_key']==candidate+f'|G{gi}'
                i=int(dr['deleted_nms_input_position_0based'])
                assert json.loads(dr['deleted_xyxy'])==b[i].tolist() and float(dr['deleted_score'])==float(s[i])
                disagree=(float(g32[gi,i])>=.75)!=(scalar_iou(gt[gi].tolist(),b[i].tolist())>=.75)
                assert boolean(dr['GT_threshold_classification_disagreement'])==disagree
            er=[row for row in arows if boolean(row['primary_possible_relation'])]
            di=[row for row in arows if boolean(row['diagnostic_only_relation'])]
            event_fields={'deleted_good_candidate_count':len(good),'primary_relation_count':len(er),'diagnostic_only_relation_count':len(di),
                'unexplained_deleted_good_count':sum(boolean(d['unexplained_primary']) for d in drows),
                'nms_input_count':len(b),'original_nms_output_count':len(keep),'final_proposal_count':len(r['final_boxes'])}
            for col,value in event_fields.items():equal(event[col],value)
            flags={'has_primary_relation':bool(er),'all_deleted_good_have_primary_relation':all(int(d['primary_relation_count'])>0 for d in drows),
                'has_primary_score_tie':any(boolean(row['score_tie']) for row in er),
                'has_eligible_score_tie':any(boolean(row['eligible_same_layer_score']) and boolean(row['score_tie']) for row in arows),
                'near_nms_threshold':any(boolean(row['near_threshold_diagnostic_only']) for row in arows),
                'relation_predicate_disagreement':any(boolean(row['relation_predicate_disagreement']) for row in arows),
                'has_diagnostic_only_relation':bool(di),
                'GT_float32_float64_coverage_disagreement':not any(scalar_iou(gt[gi].tolist(),bb)>=.75 for bb in b.tolist()) or any(scalar_iou(gt[gi].tolist(),b[int(j)].tolist())>=.75 for j in keep),
                'any_related_retained_other_gt_overlap_larger':any(boolean(row['retained_other_gt_overlap_larger']) for row in er)}
            flags['numerically_sensitive']=flags['near_nms_threshold'] or flags['relation_predicate_disagreement'] or flags['GT_float32_float64_coverage_disagreement'] or any(boolean(d['GT_threshold_classification_disagreement']) for d in drows)
            for col,value in flags.items():assert boolean(event[col])==value,(eid,col)
            for j in related_j:
                cid=cat[int(pre['ids'][j])]['candidate_id']
                for t in range(len(gt)):expected_overlap_keys.add((eid,cid,t))
            for row in [r for r in overlaps if r['event_id']==eid]:
                j=positions[int(row['retained_flat_index'])];t=int(row['compared_gt_index'])
                assert j in related_j and json.loads(row['candidate_xyxy'])==b[j].tolist() and json.loads(row['gt_xyxy'])==gt[t].tolist()
                equal(row['iou_float32_cpu'],float(g32[t,j]));equal(row['iou_float64_diagnostic'],scalar_iou(gt[t].tolist(),b[j].tolist()),1e-12)
    assert len(groups)==load_count==57
    assert expected_good=={(r['event_id'],r['deleted_candidate_id']) for r in deleted}
    assert expected_overlap_keys=={(r['event_id'],r['retained_candidate_id'],int(r['compared_gt_index'])) for r in overlaps}
    assert len(expected_overlap_keys)==len(overlaps)
    unique=read(out/'unique_deleted_candidates.csv')
    assert {r['candidate_id'] for r in unique}=={r['deleted_candidate_id'] for r in deleted}
    for r in unique:
        dd=[d for d in deleted if d['deleted_candidate_id']==r['candidate_id']]
        equal(r['candidate_GT_row_count'],len(dd));assert json.loads(r['GT_indices'])==sorted({int(d['gt_index']) for d in dd})
    paired=read(out/'all_gt_pairing.csv');assert len(paired)==416 and len({(r['stem'],r['gt_index']) for r in paired})==416
    for p in paired:
        key=p['stem'],int(p['gt_index'])
        states={}
        for model in ('IR','Fusion'):
            pre=source[model,*key]['score_filtered']['covered_075_float32_cpu']=='True'
            post=source[model,*key]['layer_nms']['covered_075_float32_cpu']=='True'
            lost=pre and not post;states[model]=pre,lost
            assert boolean(p[model+'_precovered'])==pre and boolean(p[model+'_nms_lost'])==lost
            assert p[model+'_risk_state']==('not_at_risk' if not pre else 'lost' if lost else 'still_covered')
        assert boolean(p['both_precovered'])==(states['IR'][0] and states['Fusion'][0])
        il,fl=states['IR'][1],states['Fusion'][1]
        assert p['loss_pair_category']==('both_lost' if il and fl else 'IR_only_lost' if il else 'Fusion_only_lost' if fl else 'neither_lost')
    common=[r for r in paired if boolean(r['both_precovered'])]
    assert common==read(out/'common_risk_pairing.csv') and len(common)==297
    for r in read(out/'paired_summary.csv'):
        pp=[p for p in paired if (r['scene_id']=='ALL' or p['scene_id']==r['scene_id']) and (r['cohort']=='all_gt' or boolean(p['both_precovered']))]
        equal(r['GT_denominator'],len(pp))
        counts={m:sum(boolean(p[m+'_nms_lost']) for p in pp) for m in ('IR','Fusion')}
        for model in ('IR','Fusion'):
            pre=sum(boolean(p[model+'_precovered']) for p in pp);equal(r[model+'_precovered_gt'],pre)
            equal(r[model+'_lost_gt'],counts[model]);equal(r[model+'_conditional_loss_rate'],counts[model]/pre if pre else None)
        for cat in ('both_lost','IR_only_lost','Fusion_only_lost','neither_lost'):equal(r[cat],sum(p['loss_pair_category']==cat for p in pp))
        equal(r['Fusion_minus_IR_lost_gt'],counts['Fusion']-counts['IR'])
        equal(r['same_risk_loss_rate_difference'],(counts['Fusion']-counts['IR'])/len(pp) if r['cohort']=='both_precovered' and pp else None)
    for r in read(out/'model_scene_summary.csv'):
        selected=lambda x:x['model']==r['model'] and (r['scene_id']=='ALL' or x['scene_id']==r['scene_id'])
        ee=[e for e in events if selected(e)];dd=[d for d in deleted if selected(d)];rr=[q for q in relations if selected(q)]
        pp=[p for p in paired if r['scene_id']=='ALL' or p['scene_id']==r['scene_id']]
        pre=sum(boolean(p[r['model']+'_precovered']) for p in pp)
        numeric={'all_valid_gt':len(pp),'nms_precovered_gt':pre,'nms_lost_gt':len(ee),'conditional_loss_rate':len(ee)/pre if pre else None,
            'deleted_good_candidate_GT_rows':len(dd),'unique_deleted_good_candidates':len({d['deleted_candidate_id'] for d in dd}),
            'primary_relation_GT_rows':len(rr),'unique_primary_candidate_pairs':len({(q['deleted_candidate_id'],q['retained_candidate_id']) for q in rr})}
        for col,value in numeric.items():equal(r[col],value)
        fields={'gt_with_primary_relation':'has_primary_relation','gt_every_deleted_good_has_primary_relation':'all_deleted_good_have_primary_relation',
            'gt_without_any_primary_relation':None,'gt_with_primary_score_tie':'has_primary_score_tie','gt_with_eligible_score_tie':'has_eligible_score_tie',
            'gt_near_nms_threshold':'near_nms_threshold','gt_relation_predicate_disagreement':'relation_predicate_disagreement',
            'gt_numerically_sensitive':'numerically_sensitive','gt_with_diagnostic_only_relation':'has_diagnostic_only_relation',
            'gt_with_related_retained_other_gt_overlap_larger':'any_related_retained_other_gt_overlap_larger'}
        for col,field in fields.items():equal(r[col],sum(boolean(e[field]) for e in ee) if field else sum(not boolean(e['has_primary_relation']) for e in ee))
        equal(r['gt_partly_or_fully_unexplained_primary'],sum(int(e['unexplained_deleted_good_count'])>0 for e in ee))
    assert maxerror<=1e-12
    return {'status':'passed','GT_model_events':len(events),'distinct_GT':48,'capture_loads':load_count,'candidate_GT_rows':len(deleted),
        'primary_relation_rows':len(relations),'pair_audit_rows':len(audit),'other_GT_overlap_rows':len(overlaps),
        'near_threshold_rows':near_count,'predicate_disagreement_rows':disagree_count,'independent_float64_atol':1e-12,
        'max_independent_float64_error':maxerror,'historical_files_unchanged':len(meta['historical_sha256_before']),
        'actual_GPU_suppression_pairs_proven':0,'no_model_or_NMS_execution':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(HERE):raise ValueError('Day05 only')
    args.output.mkdir(parents=True,exist_ok=False);before={str(p):sha(p) for p in args.analysis_dir.rglob('*') if p.is_file()};code=1
    try:
        torch.set_num_threads(2)
        with patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forbidden')), \
             patch.object(ops,'nms',side_effect=RuntimeError('NMS forbidden')),patch.object(ops,'batched_nms',side_effect=RuntimeError('NMS forbidden')):
            info=verify(args.analysis_dir)
        assert before=={str(p):sha(p) for p in args.analysis_dir.rglob('*') if p.is_file()};code=0
    except BaseException:info={'status':'failed','error':traceback.format_exc()}
    info.update(exit_code=code,command=subprocess.list2cmdline(sys.orig_argv),verification_source_sha256=sha(Path(__file__)))
    text=json.dumps(info,ensure_ascii=False,indent=2)
    with (args.output/'verification.json').open('x',encoding='utf-8') as f:f.write(text+'\n')
    with (args.output/'run.log').open('x',encoding='utf-8') as f:f.write(text+'\n')
    print(text);return code
if __name__=='__main__':sys.exit(main())
