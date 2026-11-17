"""One independent full-input class-NMS replay of saved Day04 CPU candidates."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,datetime,hashlib,inspect,json,math,platform,struct,subprocess,traceback
from collections import Counter
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from pathlib import Path
from unittest.mock import patch
import torch,torchvision
from torchvision.ops import box_iou,boxes as box_ops
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
D4=ROOT/'week10/day04/outputs/roi_trace_run01'
GOOD_FLATS=(8571,8576,9156,12831,2576)
FOCUS=GOOD_FLATS+(8877,)
COUNTS=dict(candidate_capture_loads=0,main_replays=0,full_output_verifications=0,checkpoint_loads=0,model_forwards=0,model_restores=0,existing_nms_operator_calls=0,counterfactual_replays=0,server_connections=0,test_runs=0,training_runs=0)

def require(ok,note):
    if not ok:raise ValueError(note)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,data):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows,fields=None):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
def q(v):return struct.unpack('f',struct.pack('f',v))[0]
def scalar_iou(a,b,binary32=True):
    f=q if binary32 else float
    aa=f(f(a[2]-a[0])*f(a[3]-a[1]));bb=f(f(b[2]-b[0])*f(b[3]-b[1]))
    inter=f(max(0.,f(min(a[2],b[2])-max(a[0],b[0])))*max(0.,f(min(a[3],b[3])-max(a[1],b[1]))))
    return f(inter/f(f(aa+bb)-inter))

def greedy_replay(boxes,scores,labels,threshold):
    """Algorithm inputs ONLY. No GT, source layer, IDs, focus labels or saved output."""
    require(boxes.device.type=='cpu' and boxes.dtype==scores.dtype==torch.float32,'CPU float32 inputs required')
    bb=boxes.tolist();ss=scores.tolist();cc=labels.tolist()
    maximum=max(max(b) for b in bb);stride=q(maximum+q(1.))
    offsets=[q(q(c)*stride) for c in cc]
    shifted=[[q(v+offsets[i]) for v in b] for i,b in enumerate(bb)]
    order=sorted(range(len(ss)),key=lambda i:(-ss[i],i))
    active=order.copy();kept=[];steps=[];decisions=[];comparisons=[]
    while active:
        a=active[0];step=len(kept);kept.append(a)
        decisions.append(dict(input_position=a,action='keep',step=step,selected_position=a,path_iou=None))
        removed=[];remaining=[]
        for b in active[1:]:
            value=scalar_iou(shifted[a],shifted[b]);drop=value>threshold
            comparisons.append(dict(step=step,selected_position=a,other_position=b,path_iou=value,deleted=drop))
            if drop:
                removed.append(b);decisions.append(dict(input_position=b,action='delete',step=step,selected_position=a,path_iou=value))
            else:remaining.append(b)
        steps.append(dict(step=step,selected_position=a,new_deleted_positions=removed,remaining_positions=remaining))
        active=remaining
    return dict(keep=kept,order=order,steps=steps,decisions=decisions,comparisons=comparisons,shifted=shifted,offsets=offsets,max_coordinate=maximum,offset_stride=stride)

def analyze(out):
    provenance=[]
    def record(p,expected=None,origin='current read-only input'):
        h=sha(p)
        if expected is not None:require(h==expected,'SHA mismatch: '+str(p))
        provenance.append(dict(path=str(p),sha256=h,expected_sha256=expected,origin=origin))
    manifest=load(D4/'artifact_manifest.json');record(D4/'artifact_manifest.json')
    # SHA verification is file-integrity work, not another numerical experiment.
    for name,h in manifest.items():record(D4/name,h,'Day04 artifact manifest')
    record(ROOT/'week10/day04/README.md')
    require(load(D4/'verification.json')['status']=='passed','Day04 not verified')
    previous=load(D4/'run_metadata.json')['environment']
    require(previous['torch']==torch.__version__ and previous['torchvision']==torchvision.__version__,'Day04/current library versions differ')
    source=Path(inspect.getfile(box_ops));source_text=source.read_text(encoding='utf-8')
    record(source);(out/'boxes_source.py.txt').write_text(source_text,encoding='utf-8')
    roi_source=Path(next(x['path'] for x in load(D4/'provenance.json') if x.get('role')=='actual installed RoI implementation'))
    record(roi_source,load(D4/'model_checks.json')['roi_source_sha'],'Day04 actual RoI source')
    require('boxes.numel() > (4000' in source_text and 'boxes + offsets[:, None]' in source_text and 'idxs.to(boxes)' in source_text,'Unexpected batched_nms implementation')
    require(not torch.jit.is_tracing() and not torch.jit.is_scripting(),'Tracing/scripting outside captured path')
    raw_path=D4/'raw_capture.pth'
    COUNTS['candidate_capture_loads']+=1
    raw=torch.load(raw_path,weights_only=True,map_location='cpu')
    require(raw['run_identity']=='roi_trace_run01/IR/030129','Unexpected capture identity')
    args=raw['nms']['args'];require(len(args)==4 and not raw['nms']['kwargs'],'Unexpected original call signature')
    boxes,scores,labels,threshold=args
    require(boxes.shape==(44,4) and scores.shape==labels.shape==(44,),'Expected full 44-item input')
    require(labels.dtype==torch.int64 and bool((labels==1).all()) and threshold==.5,'Original class labels/threshold')
    require(bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(scores).all()) and bool((boxes[:,2:]>boxes[:,:2]).all()),'Invalid input tensor')
    require(boxes.numel()<=4000,'Input not in small CPU coordinate-trick path')
    # Generate the whole trajectory before reading original saved keep indices.
    COUNTS['main_replays']+=1
    replay=greedy_replay(boxes,scores,labels,threshold)

    # Identity reconstruction is based on original stage indices, never nearest-box matching.
    proposal_flats=raw['rpn_stages'][0]['post_nms_topn']['ids'].tolist()
    probabilities=raw['softmax']['result'];person=raw['clip']['result'][:,1]
    scoreidx=torch.where(probabilities[:,1]>raw['configuration']['roi']['score'])[0]
    rows=scoreidx[raw['small']['result']]
    require(exact(person[scoreidx],raw['small']['args'][0]),'Score-filter reconstruction vs actual small input')
    require(exact(person[rows],boxes) and exact(probabilities[rows,1],scores),'Mapped actual RoI input boxes/scores differ')
    require(exact(labels,torch.ones(len(rows),dtype=torch.int64)),'Mapped labels differ')
    savedrows=[r for r in read(D4/'stage_candidates.csv') if r['stage']=='small_filtered']
    require(len(savedrows)==44,'Day04 stage export size')
    catalogue=raw['catalogues'][0];flat=[proposal_flats[int(i)] for i in rows]
    ids=[catalogue[f]['candidate_id']+'/person' for f in flat]
    for i,r in enumerate(savedrows):
        require((int(r['stage_index']),int(r['proposal_row']),int(r['flat']),int(r['class_label']))==(i,int(rows[i]),flat[i],int(labels[i])),'Day04 identity mapping')
        require(json.loads(r['xyxy'])==boxes[i].tolist() and float(r['classification_score'])==float(scores[i]),'Day04 actual input export values')
    bb=boxes.tolist();ss=scores.tolist();cc=labels.tolist()
    gt=raw['gt_all'][0];gt_iou=box_iou(boxes,gt[None])[:,0].tolist()
    require(set(flat[i] for i,x in enumerate(gt_iou) if x>=.75)==set(GOOD_FLATS),'Unexpected five good branches')
    require(gt_iou[flat.index(8877)]<.75,'8877 is reference, not a stage-good box')
    for i,r in enumerate(savedrows):require(float(r['gt_iou'])==gt_iou[i],'Day04 GT IoU differs')

    original=raw['nms']['result'];original_list=original.tolist()
    require(original.dtype==torch.int64 and len(set(original_list))==len(original_list) and all(0<=i<44 for i in original_list),'Invalid saved original indices')
    generated=replay['keep'];gen=torch.tensor(generated,dtype=torch.int64)
    COUNTS['full_output_verifications']+=1
    nms_match=generated==original_list
    maximum=raw['configuration']['roi']['detections'];require(maximum==100,'Original final budget')
    post=raw['actual_post_output'];model=raw['model_output']
    post_match=all(exact(x[gen[:maximum]],y[0]) for x,y in zip((boxes,scores,labels),post))
    model_match=all(exact(x[gen[:maximum]],model[k]) for x,k in zip((boxes,scores,labels),('boxes','scores','labels')))
    old_post_match=all(exact(x[original[:maximum]],y[0]) for x,y in zip((boxes,scores,labels),post))
    original_facts_match=old_post_match and all(exact(x[original[:maximum]],model[k]) for x,k in zip((boxes,scores,labels),('boxes','scores','labels')))
    source_export=read(D4/'final_sources.csv')
    require([int(r['flat']) for r in source_export]==[flat[i] for i in original_list[:maximum]],'Original final source export mapping')
    output_rows=[]
    for rank in range(max(len(original_list),len(generated))):
        a=original_list[rank] if rank<len(original_list) else None;b=generated[rank] if rank<len(generated) else None
        output_rows.append(dict(output_position=rank,original_nms_input_position=a,replay_nms_input_position=b,original_id=ids[a] if a is not None else None,replay_id=ids[b] if b is not None else None,original_flat=flat[a] if a is not None else None,replay_flat=flat[b] if b is not None else None,original_box=bb[a] if a is not None else None,replay_box=bb[b] if b is not None else None,original_classification_score=ss[a] if a is not None else None,replay_classification_score=ss[b] if b is not None else None,original_label=cc[a] if a is not None else None,replay_label=cc[b] if b is not None else None,all_fields_equal=a==b,within_final100=rank<maximum))
    output_verification=dict(original_count=len(original_list),replay_count=len(generated),original_indices=original_list,replay_indices=generated,full_index_sequence_equal=nms_match,identity_box_score_class_equal=all(r['all_fields_equal'] for r in output_rows),original_post_and_model_consistent=original_facts_match,replay_postprocess_exact=post_match,replay_budget_and_model_exact=model_match)
    save(out/'output_verification.json',output_verification)

    # Independent geometric audit only: no extra greedy or NMS runs.
    shifted=(boxes+labels.to(boxes)[:,None]*(boxes.max()+boxes.new_tensor(1.)))
    require(shifted.tolist()==replay['shifted'],'Independent tensor float32 offset construction differs')
    raw_matrix=box_iou(boxes,boxes);path_matrix=box_iou(shifted,shifted)
    audits=[];ties=[];sensitive=[];unrounded64=boxes.double()+labels.double()[:,None]*(boxes.double().max()+1.)
    for a in range(44):
        for b in range(a+1,44):
            raw32=scalar_iou(bb[a],bb[b]);path32=scalar_iou(replay['shifted'][a],replay['shifted'][b])
            require(raw32==float(raw_matrix[a,b]) and path32==float(path_matrix[a,b]),'Independent binary32/CPU tensor IoU mismatch')
            raw64=scalar_iou(bb[a],bb[b],False);path64=scalar_iou(replay['shifted'][a],replay['shifted'][b],False);offset64=scalar_iou(unrounded64[a].tolist(),unrounded64[b].tolist(),False)
            raw_delete=cc[a]==cc[b] and raw32>threshold;path_delete=path32>threshold
            disagree=raw_delete!=path_delete or path_delete!=(path64>threshold) or raw_delete!=(cc[a]==cc[b] and raw64>threshold)
            near=any(abs(x-threshold)<=1e-6 for x in [raw32,raw64,path32,path64,offset64])
            row=dict(a_position=a,b_position=b,a_id=ids[a],b_id=ids[b],a_label=cc[a],b_label=cc[b],a_score=ss[a],b_score=ss[b],score_tie=ss[a]==ss[b],raw_iou_float32=raw32,raw_iou_float64=raw64,offset_path_iou_float32=path32,offset_rounded_coordinates_iou_float64=path64,full_float64_offset_iou=offset64,raw_deletion_condition=raw_delete,path_deletion_condition=path_delete,near_threshold_1e_6=near,predicate_disagreement=disagree)
            audits.append(row)
            if row['score_tie']:ties.append(row)
            if near or disagree:sensitive.append(row)
    decision_count=Counter(x['input_position'] for x in replay['decisions'])
    require(decision_count==Counter(range(44)),'Each input must receive exactly one keep/delete decision')
    require([d['input_position'] for d in replay['decisions'] if d['action']=='keep']==generated,'Decision/keep order differs')
    matched=nms_match and post_match and model_match and original_facts_match
    tie_ambiguity=any(r['a_label']==r['b_label'] and r['path_deletion_condition'] for r in ties)
    relevant_comparisons={tuple(sorted((c['selected_position'],c['other_position']))) for c in replay['comparisons']}
    relevant_precision=any((r['a_position'],r['b_position']) in relevant_comparisons and (r['predicate_disagreement'] or r['near_threshold_1e_6']) for r in sensitive)
    attribution_valid=matched and not tie_ambiguity and not relevant_precision
    original_positions={i:j for j,i in enumerate(original_list)}
    decision_lookup={d['input_position']:d for d in replay['decisions']}
    rank={p:i for i,p in enumerate(replay['order'])}
    inputs=[]
    for i in range(44):
        inputs.append(dict(nms_input_position=i,roi_proposal_row=int(rows[i]),source_rpn_flat=flat[i],person_branch_id=ids[i],source_candidate_id=catalogue[flat[i]]['candidate_id'],class_label=cc[i],source_fpn_layer=catalogue[flat[i]]['level_key'],source_layer_used_for_grouping=False,xyxy=bb[i],classification_score=ss[i],gt_iou=gt_iou[i],stage_good_for_G0=gt_iou[i]>=.75,sort_position=rank[i],class_coordinate_offset=replay['offsets'][i],shifted_xyxy=replay['shifted'][i],original_nms_kept=i in original_positions,original_output_position=original_positions.get(i),original_final100_kept=i in original_positions and original_positions[i]<maximum))
    for d in replay['decisions']:
        i=d['input_position'];a=d['selected_position'];d.update(branch_id=ids[i],source_flat=flat[i],roi_proposal_row=int(rows[i]),classification_score=ss[i],class_label=cc[i],gt_iou=gt_iou[i],selected_id=ids[a],selected_flat=flat[a],original_nms_kept=i in original_positions,replay_agrees_with_original=(d['action']=='keep')==(i in original_positions),attribution_validated=attribution_valid)
    for s in replay['steps']:
        a=s['selected_position'];s.update(selected_id=ids[a],selected_flat=flat[a],selected_classification_score=ss[a],selected_class_label=cc[a],new_deleted_ids=[ids[i] for i in s['new_deleted_positions']],new_deleted_flats=[flat[i] for i in s['new_deleted_positions']],remaining_ids=[ids[i] for i in s['remaining_positions']])
    for c in replay['comparisons']:
        a,b=c['selected_position'],c['other_position'];c.update(selected_id=ids[a],other_id=ids[b],selected_flat=flat[a],other_flat=flat[b],selected_score=ss[a],other_score=ss[b],selected_label=cc[a],other_label=cc[b],raw_iou=scalar_iou(bb[a],bb[b]),strict_threshold=threshold)
    focus_rows=[]
    for f in FOCUS:
        i=flat.index(f);d=decision_lookup[i];a=d['selected_position']
        focus_rows.append(dict(role='good_for_G0_at_NMS_input' if f in GOOD_FLATS else 'reference_8877_not_good_at_NMS_input',source_flat=f,roi_proposal_row=int(rows[i]),nms_input_position=i,branch_id=ids[i],class_label=cc[i],classification_score=ss[i],gt_iou=gt_iou[i],xyxy=bb[i],original_nms_kept=i in original_positions,replay_action=d['action'],first_delete_step=d['step'] if d['action']=='delete' else None,first_deleter_position=a if d['action']=='delete' else None,first_deleter_flat=flat[a] if d['action']=='delete' else None,first_deleter_id=ids[a] if d['action']=='delete' else None,first_deleter_label=cc[a] if d['action']=='delete' else None,first_deleter_score=ss[a] if d['action']=='delete' else None,first_deleter_gt_iou=gt_iou[a] if d['action']=='delete' else None,raw_pair_iou=scalar_iou(bb[a],bb[i]) if d['action']=='delete' else None,path_pair_iou=d['path_iou'],original_and_replay_agree=d['replay_agrees_with_original'],attribution_validated=attribution_valid))
    relations=[]
    for a in original_list:
        for f in FOCUS:
            i=flat.index(f);raw_iou=scalar_iou(bb[a],bb[i]);path_iou=scalar_iou(replay['shifted'][a],replay['shifted'][i]);eligible=a!=i and cc[a]==cc[i] and ss[a]>ss[i]
            relations.append(dict(retained_position=a,retained_id=ids[a],retained_flat=flat[a],retained_label=cc[a],retained_score=ss[a],retained_gt_iou=gt_iou[a],target_position=i,target_id=ids[i],target_flat=f,target_label=cc[i],target_score=ss[i],target_gt_iou=gt_iou[i],target_is_good=f in GOOD_FLATS,raw_iou=raw_iou,path_iou=path_iou,same_class_higher_score=eligible,possible_raw_geometry=eligible and raw_iou>threshold,possible_offset_path=eligible and path_iou>threshold,replay_first_deleter=decision_lookup[i]['action']=='delete' and decision_lookup[i]['selected_position']==a,original_retained_kept=True,original_target_kept=i in original_positions,attribution_validated=attribution_valid))
    possible=[r for r in relations if r['possible_raw_geometry'] or r['possible_offset_path']]
    save(out/'implementation_evidence.json',dict(original_day04_environment=previous,current_environment=environment(),source_path=str(source),source_sha256=sha(source),applicable_path='CPU 44x4=176 elements <=4000; not tracing; _batched_nms_coordinate_trick',grouping='actual class labels, not source FPN levels',all_class_labels=sorted(set(cc)),threshold=threshold,comparison='strict >, no tolerance',sorting='descending original classification score; input position ascending for ties, predetermined',max_coordinate=replay['max_coordinate'],offset_stride=replay['offset_stride'],offsets=replay['offsets'],all_same_class_offset_still_applied=True,arithmetic='offset and coordinate addition in float32; area/intersection/union/division each rounded binary32',kernel_event_trace_available=False,kernel_source_not_embedded=True))
    tables=[('nms_inputs',inputs),('replay_steps',replay['steps']),('candidate_decisions',replay['decisions']),('replay_comparisons',replay['comparisons']),('output_comparison',output_rows),('focus_attributions',focus_rows),('retained_focus_all_pairs',relations),('possible_retained_relations',possible),('all_pair_numeric_audit',audits),('score_ties',ties),('threshold_precision_audit',sensitive)]
    for name,rr in tables:csvout(out/(name+'.csv'),rr,list(audits[0]) if name in ['score_ties','threshold_precision_audit'] else list(relations[0]) if name=='possible_retained_relations' else None)
    geometry=render(out,raw,inputs,focus_rows,relations,attribution_valid,record)
    save(out/'figure_geometry.json',geometry);save(out/'input_provenance.json',provenance)
    write_report(out,inputs,focus_rows,possible,output_verification,replay,audits,ties,sensitive,attribution_valid)
    status=matched and not tie_ambiguity and not relevant_precision
    result=dict(status='passed' if status else 'failed',scope='030129/IR/G0 RoI class NMS only',**COUNTS,nms_input_count=44,original_output_count=len(original_list),replay_output_count=len(generated),full_output_match=matched,all_input_identity_mappings_verified=True,unique_decisions_verified=True,decision_rows=len(replay['decisions']),actual_pair_comparisons=len(replay['comparisons']),all_input_pair_audit_rows=len(audits),float32_offset_tensor_check=1,independent_float32_pair_checks=len(audits)*2,GT_IoU_rows=44,score_tie_pairs=len(ties),threshold_precision_audit_rows=len(sensitive),tie_ambiguity=tie_ambiguity,relevant_precision_ambiguity=relevant_precision,attribution_validated=attribution_valid,focus_count=6,good_count=5,possible_relation_count=len(possible),core_expected_final_sources_verified=[flat[i] for i in original_list]==[8862,8546,2571])
    print('COMPLETE_OUTPUT',output_verification,flush=True)
    for f in focus_rows:print('FOCUS_ATTRIBUTION',json.dumps(f,ensure_ascii=False),flush=True)
    return result

def environment():return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform(),device='CPU',threads=torch.get_num_threads())

def render(out,raw,inputs,focus,relations,valid,record):
    # Only original image/XML reading and original geometric helper, no model restoration imports.
    sys.path.insert(0,str(ROOT/'week08/day01'));sys.path.insert(0,str(ROOT/'week08/day02'))
    from llvip_dataset import _load_infrared,_parse_voc_annotation
    from paired_letterbox import PairedLetterbox
    previous={Path(x['path']):x['sha256'] for x in load(D4/'provenance.json')}
    image_path=ROOT/'data/LLVIP_1000/images/val/infrared/030129.jpg';xml_path=ROOT/'data/LLVIP_1000/annotations_voc/val/030129.xml'
    for p in [image_path,xml_path,ROOT/'data/LLVIP_1000/manifest.csv',ROOT/'week08/day01/llvip_dataset.py',ROOT/'week08/day02/paired_letterbox.py']:
        record(p,previous[p],'Day04 recorded image/XML/geometry source SHA')
    original,size=_load_infrared(image_path,3);target,xml_size=_parse_voc_annotation(xml_path,0,'030129')
    letterbox=PairedLetterbox(640);_,ir,transformed=letterbox(original,original,target)
    require(size==xml_size==(1280,1024) and exact(transformed['boxes'],raw['gt_all']),'Full original GT Letterbox geometry')
    params=letterbox.parameters(1024,1280);require(params.padding_ltrb==(0,64,0,64),'Letterbox padding')
    image=Image.fromarray(ir.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy());gt=raw['gt_all'].tolist()
    lookup={r['source_rpn_flat']:r for r in inputs}
    kept=[r for r in inputs if r['original_nms_kept']]
    selected=[lookup[f] for f in FOCUS]+kept
    boxes=[b for b in gt]+[r['xyxy'] for r in selected]
    bounds=(max(0,math.floor(min(b[0] for b in boxes))-15),max(0,math.floor(min(b[1] for b in boxes))-15),min(640,math.ceil(max(b[2] for b in boxes))+15),min(640,math.ceil(max(b[3] for b in boxes))+15))
    def text(d,xy,s,size=21,color='#26354a'):d.text(xy,s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size),fill=color)
    def color(r):return '#e98b32' if r['original_nms_kept'] else '#a076d2' if r['source_rpn_flat']==8877 else '#00a89e'
    context=image.copy();d=ImageDraw.Draw(context)
    for j,b in enumerate(gt):d.rectangle(b,outline='#ffd24a' if j==0 else '#aab3be',width=3);text(d,(b[0],max(0,b[1]-23)),f'G{j}',18)
    d.rectangle(bounds,outline='#ffffff',width=2);context.save(out/'image_context.png')
    cw,ch=500,520;canvas=Image.new('RGB',(1500,135+math.ceil(len(selected)/3)*ch),'#f1f5fa');d=ImageDraw.Draw(canvas)
    text(d,(22,15),'030129/IR/G0：回归裁剪后的RoI框，不是原RPN框',29)
    text(d,(22,60),'黄G0 / 灰其他GT / 青5个好框 / 紫8877参照 / 橙原保留框',22)
    text(d,(22,95),'所有候选类别=person；来源层不用于分组；虚框表示原NMS删除。',20)
    for k,r in enumerate(selected):
        x=(k%3)*cw;y=135+(k//3)*ch;role='原保留框' if r['original_nms_kept'] else '8877参照（非好框）' if r['source_rpn_flat']==8877 else 'NMS输入好框'
        text(d,(x+12,y+4),f'{role} flat={r["source_rpn_flat"]}',22,color(r))
        crop=image.crop(bounds);cd=ImageDraw.Draw(crop);left,top,right,bottom=bounds
        def tr(b):return [b[0]-left,b[1]-top,b[2]-left,b[3]-top]
        for j,b in enumerate(gt):cd.rectangle(tr(b),outline='#ffd24a' if j==0 else '#aab3be',width=1)
        b=tr(r['xyxy'])
        if r['original_nms_kept']:cd.rectangle(b,outline=color(r),width=2)
        else:dash_rectangle(cd,b,color(r),1,4)
        scale=min(470/(right-left),300/(bottom-top));ww,hh=round((right-left)*scale),round((bottom-top)*scale)
        canvas.paste(crop.resize((ww,hh),Image.Resampling.BILINEAR),(x+(cw-ww)//2,y+46))
        text(d,(x+12,y+357),f'分类分数={r["classification_score"]:.9f}',20)
        text(d,(x+12,y+389),f'G0 IoU={r["gt_iou"]:.9f}  类别={r["class_label"]}',20)
        text(d,(x+12,y+421),f'RoI行={r["roi_proposal_row"]}  NMS输入#{r["nms_input_position"]}',20)
        text(d,(x+12,y+453),f'原NMS {"保留" if r["original_nms_kept"] else "删除"}；来源层{r["source_fpn_layer"]}',20)
    canvas.save(out/'key_candidates.png')
    canvas=Image.new('RGB',(1600,180+len(focus)*170),'#f1f5fa');d=ImageDraw.Draw(canvas)
    text(d,(25,16),'类别NMS独立重放：六个重点候选的首次删除',30)
    text(d,(25,65),'红实线箭头＝该重放首次删除；灰虚线标记＝其他可能几何关系。',22)
    text(d,(25,105),'与Day04原CPU完整输出一致' if valid else '未通过无歧义核验：以下仅为待核验重放观察',22)
    for j,f in enumerate(focus):
        y=160+j*170;a=lookup[f['first_deleter_flat']] if f['first_deleter_flat'] is not None else None
        if a is None:continue
        d.rectangle((25,y,410,y+120),outline='#e98b32',width=3)
        text(d,(38,y+7),f'保留flat {a["source_rpn_flat"]}  类别1',23)
        text(d,(38,y+43),f'score={a["classification_score"]:.9f}',20)
        text(d,(38,y+76),f'GT IoU={a["gt_iou"]:.9f}',20)
        d.line((425,y+60,800,y+60),fill='#db5365',width=4);d.polygon([(800,y+60),(780,y+50),(780,y+70)],fill='#db5365')
        text(d,(435,y+4),f'重放step={f["first_delete_step"]}（从0开始）',20)
        text(d,(435,y+76),f'原IoU {f["raw_pair_iou"]:.7f}',19)
        text(d,(435,y+106),f'偏移路径IoU {f["path_pair_iou"]:.7f}',19)
        dash_rectangle(d,(820,y,1560,y+120),'#00a89e' if f['source_flat']!=8877 else '#a076d2',3,10)
        text(d,(836,y+5),f'删除flat {f["source_flat"]}：'+('好框' if f['source_flat'] in GOOD_FLATS else '参照，非本阶段好框'),23)
        text(d,(836,y+42),f'score={f["classification_score"]:.9f}  GT IoU={f["gt_iou"]:.9f}',20)
        other=[r['retained_flat'] for r in relations if r['target_flat']==f['source_flat'] and (r['possible_raw_geometry'] or r['possible_offset_path']) and not r['replay_first_deleter']]
        for xx in range(839,875,12):d.line((xx,y+96,xx+6,y+96),fill='#8292a6',width=2)
        text(d,(887,y+82),'其他可能保留关系：'+str(other)+'（非首次归因）',19)
    text(d,(25,canvas.height-35),'箭头不是原CPU内核事件记录；全输入轨迹、全部可能关系和原索引事实分别保存在CSV。',21)
    canvas.save(out/'replay_process.png')
    return dict(original_wh=list(size),letterbox_wh=[640,640],resized_wh=list(params.resized_wh),scale=params.scale,padding_ltrb=list(params.padding_ltrb),coordinates='640x640 including padding for GT and actual RoI input',GT_all=gt,GT_exact_match=True,shared_zoom_xyxy=list(bounds),crop_for_display_only=True)

def dash_rectangle(d,box,color,width=2,dash=8):
    x1,y1,x2,y2=box
    for a,b in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
        length=math.dist(a,b)
        if not length:continue
        for k in range(0,math.ceil(length),dash*2):
            e=min(k+dash,length);d.line([(a[0]+(b[0]-a[0])*k/length,a[1]+(b[1]-a[1])*k/length),(a[0]+(b[0]-a[0])*e/length,a[1]+(b[1]-a[1])*e/length)],fill=color,width=width)

def write_report(out,inputs,focus,possible,verification,replay,audits,ties,sensitive,valid):
    goods=[f for f in focus if f['source_flat'] in GOOD_FLATS];reference=next(f for f in focus if f['source_flat']==8877)
    summary='；'.join(f'flat {f["source_flat"]}在第{f["first_delete_step"]}步由flat {f["first_deleter_flat"]}首次删除' if f['replay_action']=='delete' else f'flat {f["source_flat"]}在该重放中保留（与原删除记录不同）' for f in goods)
    concentrated=len({f['first_deleter_flat'] for f in goods})==1
    lines=['# Week10 Day05：030129/IR最终类别NMS的独立重放核查','',
           '本轮只读Day04保存的44个实际RoI类别NMS输入，独立生成全部过程后才对照原输出。以下步号从0开始，分数是person分类分数，不是RPN objectness；flat和来源FPN层仅用于身份说明。','',
           ('完整核验通过，以下为与Day04原CPU完整输出一致的独立重放解释。' if valid else '本次无歧义核验未通过，以下过程只能作为待验证观察；不能写成已核验删除解释。'),' ',
           '## 先回答固定问题','',summary+'。',
           f'5个好框的首次删除归因是否集中在同一个保留框：{concentrated}。8877作为非好框参照，在第{reference["first_delete_step"]}步由flat {reference["first_deleter_flat"]}首次删除。它在此之前回归后对G0已不达0.75，不能混入5个本阶段好框。','',
           '这次解释的关键是：分类NMS按分类分数排列候选，不按它们与G0的IoU排列。与G0贴合更好的低分框可能被先选中的高分框排除。GT仅在过程结束后用于解释，未参与排序或删除。','',
           '## 六个重点候选及首次删除归因','',
           '|角色|flat|RoI行|NMS输入位置|类别|分类分数|G0 IoU|重放删除步|首次删除者flat|删除者分数|删除者G0 IoU|原坐标相互IoU|偏移路径相互IoU|原/重放一致|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for f in focus:lines.append(f'|{"好框" if f["source_flat"] in GOOD_FLATS else "8877参照，非好框"}|{f["source_flat"]}|{f["roi_proposal_row"]}|{f["nms_input_position"]}|{f["class_label"]}|{f["classification_score"]:.9f}|{f["gt_iou"]:.9f}|{f["first_delete_step"]}|{f["first_deleter_flat"]}|{f["first_deleter_score"]}|{f["first_deleter_gt_iou"]}|{f["raw_pair_iou"]}|{f["path_pair_iou"]}|{f["original_and_replay_agree"]}|')
    best=next(f for f in focus if f['source_flat']==2576)
    deletion_explanation=(f'该重放中先选中保留的flat {best["first_deleter_flat"]}分类分数为{best["first_deleter_score"]:.9f}，对G0 IoU={best["first_deleter_gt_iou"]:.9f}。二者类别均为person，偏移路径相互IoU={best["path_pair_iou"]:.9f}>0.5，因此在这次重放中2576被删除。' if best['replay_action']=='delete' else '2576在本次重放中保留，与原删除事实不一致，本项归因未通过核验。')
    lines+=['', '## 2576：定位最贴合，与排序最靠前不同','',
            f'2576的person框对G0 IoU={best["gt_iou"]:.9f}，是44个输入中最贴合G0的框；但分类分数仅{best["classification_score"]:.9f}。'+deletion_explanation,
            '这里的两个IoU不能混淆：一个比较候选与指定GT，另一个比较候选与候选。分类分数和定位贴合度也不是同一种量。对G0不达标不表示背景或假阳性，图中还有另一个person GT；本轮未进行正式一对一TP/FP匹配。','',
            '## 全部原保留框与六个候选的可能关系','',
            'retained_focus_all_pairs.csv包含原3个保留框×6个重点候选的全部18个配对，可能关系表不按预期删选；下表列同类别、原分类分数严格更高，且原坐标或实际偏移路径IoU>0.5的全部关系。两个条件单列，保留与主首次归因不同的关系。','',
            '|原保留flat|目标flat|原IoU|偏移路径IoU|满足原坐标条件|满足路径条件|该重放首次删除者|','|---|---|---:|---:|---|---|---|']
    for r in possible:lines.append(f'|{r["retained_flat"]}|{r["target_flat"]}|{r["raw_iou"]:.9f}|{r["path_iou"]:.9f}|{r["possible_raw_geometry"]}|{r["possible_offset_path"]}|{r["replay_first_deleter"]}|')
    lines+=['', '## 完整轨迹与输出核验','',
            '|步号|选中flat|选中分类分数|本步新删除flat|','|---:|---:|---:|---|']
    for s in replay['steps']:lines.append(f'|{s["step"]}|{s["selected_flat"]}|{s["selected_classification_score"]}|{s["new_deleted_flats"]}|')
    lines+=['', '完整核验结果：`'+json.dumps(verification,ensure_ascii=False)+'`。',
            '先生成完整保留索引，再比较原索引序列、对应身份/框/分类分数/类别；同时逐位核对实际postprocess返回、原100预算切片和完整模型输出。每个输入恰好一次保留或删除决定，44项没有遗漏或重复。不是只比集合，也不是只核对6个重点框。','',
            '## 实现路径、数值审计与证据边界','',
            '当前torch/torchvision与Day04均为2.14.1+cpu / 0.29.1+cpu。已保存完整boxes.py快照与SHA，并核对Day04的RoI源码SHA。44×4=176个元素，非tracing，落入CPU小输入坐标偏移路径：类别标签乘(max_coordinate+1)后加到坐标上，再统一按分类分数贪心处理。全为类别1仍执行float32偏移。来源FPN层从未传入重放。',
            '主删除严格IoU>0.5；排序预先固定为分类分数降序、同分输入位置升序。同分审计覆盖全部输入，未尝试第二种排序，未修改分数/阈值/容差。主函数只接收boxes、scores、labels和threshold，没有读取原答案或GT。',
            f'全部44个输入共{len(audits)}个无序候选对；同分{len(ties)}对，阈值/精度敏感{len(sensitive)}对。独立CPU tensor IoU对照验证原坐标和偏移坐标的float32标量结果，float64仅作诊断；1e-6仅作附近标记，不参与主删除条件。非零数值差原样保存在完整审计表，不能把“无判定歧义”写成“没有浮点变化”。',
            '原记录事实：44个实际调用参数、原返回的3个完整索引及最终输出、原索引确认的候选去留、真实RoI行/flat/person映射。经核对的重建：分数过滤和小框索引组成的NMS输入身份链，与实际输入和Day04表逐项相符。独立重放解释：每次选中框、首次删除者、步号与活动集合；可能几何关系：全部原保留框与重点候选的同类别高分重叠条件。四者不混用。',
            '即使完整输出匹配，也不是直接记录了原CPU内核逐次抑制事件。保存的Python实现调用编译内核，本轮没有内核内部事件跟踪；完整输出一致支持固定输入下的独立过程解释，不能据此证明模型形成这些分类分数/回归框的根因。未做删候选、改分数或改阈值反事实实验，不推广总体比例或AP75提升。',
            '模型前向0次、checkpoint加载0次、模型创建/恢复0次。仅加载候选捕获1次（weights_only=True，CPU）；主独立重放1次、完整输出核验1次。额外计算为946对数值审计、float32偏移张量对照、原/偏移IoU矩阵与GT几何核验，没有第二次NMS重放。实际比较次数及各类核验行数见verification.json。',
            '图像与框均在640×640含Letterbox padding坐标系，完整GT与原XML及原PairedLetterbox精确核对；裁剪仅展示。历史和输入SHA前后核验见history_hashes.json、input_provenance.json；实际命令、环境、日志与退出码见run_metadata.json、run.log及同级*_process.json/log。','',
            '![关键候选](key_candidates.png)','![简化过程](replay_process.png)','', '本限定步骤完成后停止，供另一聊天审阅和教学。']
    (out/'case_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
    require(out.is_relative_to(HERE) and out!=HERE,'Output must stay inside day05');out.mkdir(parents=True,exist_ok=False)
    folders=[ROOT/'week08',ROOT/'week09']+[ROOT/f'week10/day{i:02}' for i in range(1,5)]
    extras=[ROOT/'AGENTS.md',ROOT/'data/LLVIP_1000/manifest.csv',ROOT/'data/LLVIP_1000/images/val/infrared/030129.jpg',ROOT/'data/LLVIP_1000/annotations_voc/val/030129.xml',Path(inspect.getfile(box_ops))]
    def snapshot():return {str(p):sha(p) for p in sorted(set(extras+[p for folder in folders for p in folder.rglob('*') if p.is_file()]))}
    before=snapshot();save(out/'history_before.json',before);result={};code=1;started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    class Tee:
        def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
        def flush(self):log.flush()
    with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
        try:
            torch.set_num_threads(2)
            with ExitStack() as guards:
                guards.enter_context(patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forward forbidden')))
                for obj,k in [(box_ops,'nms'),(box_ops,'batched_nms'),(torchvision.ops,'nms'),(torchvision.ops,'batched_nms')]:guards.enter_context(patch.object(obj,k,side_effect=RuntimeError('Existing NMS operator forbidden')))
                result=analyze(out)
            code=0 if result['status']=='passed' else 1
        except BaseException:
            traceback.print_exc();save(out/'failure.json',dict(error=traceback.format_exc(),counts=COUNTS))
        finally:
            after=snapshot();unchanged=before==after
            save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
            if not unchanged:code=1
            result.update(status='passed' if code==0 else 'failed',counts=COUNTS,historical_files_unchanged=unchanged,protected_file_count=len(before),exit_code=code)
            save(out/'verification.json',result)
            save(out/'run_metadata.json',dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment=environment(),source_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'README.md',HERE/'run_replay.ps1']},counts=COUNTS,exit_code=code))
            print(json.dumps(result,ensure_ascii=False),flush=True);print('EXIT',code,flush=True)
    save(out/'artifact_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return code

if __name__=='__main__':sys.exit(main())
