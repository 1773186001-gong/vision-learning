"""One actual 030078/IR forward; original RoI operations; saved-table comparison only."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,datetime,hashlib,inspect,json,math,platform,subprocess,traceback
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from pathlib import Path
from unittest.mock import patch
import torch,torchvision
from torchvision.models.detection import roi_heads as roi_module
from torchvision.ops import box_iou,FrozenBatchNorm2d
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
D3=ROOT/'week10/day03/outputs/rpn_final_comparison_run01'
D4=ROOT/'week10/day04/outputs/roi_trace_run01'
D5=ROOT/'week10/day05/outputs/roi_nms_replay_run01'
STEM='030078'
COUNTS=dict(checkpoint_loads=0,strict_state_loads=0,model_restores=0,full_forward_attempts=0,full_model_forward_calls=0,structure_probes=0,reference_capture_loads=0,independent_nms_replays=0,original_rpn_batched_nms_calls=0,original_roi_batched_nms_calls=0,server_connections=0,test_inference=0,training=0,fusion_forwards=0,additional_030129_forwards=0)

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
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]) if rows else fields or ['empty']);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def cp(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cp(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [cp(v) for v in x]
    return x
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
def normalized(x):return json.loads(json.dumps(x))
def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def config(d):
    return dict(rpn=dict(pre=d.rpn.pre_nms_top_n(),post=d.rpn.post_nms_top_n(),score=d.rpn.score_thresh,nms=d.rpn.nms_thresh,min_size=d.rpn.min_size,anchors=d.rpn.anchor_generator.sizes,ratios=d.rpn.anchor_generator.aspect_ratios),roi=dict(score=d.roi_heads.score_thresh,nms=d.roi_heads.nms_thresh,detections=d.roi_heads.detections_per_img,featmaps=d.roi_heads.box_roi_pool.featmap_names,bbox_weights=d.roi_heads.box_coder.weights),transform=dict(min=d.transform.min_size,max=d.transform.max_size,mean=d.transform.image_mean,std=d.transform.image_std,size_divisible=d.transform.size_divisible))
def environment():return dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,executable=sys.executable,platform=platform.platform(),threads=torch.get_num_threads(),seed=42,device='CPU')

def capture(out,provenance):
    def record(p,h=None,origin='current read-only source'):
        actual=sha(p)
        if h is not None:require(actual==h,'SHA mismatch '+str(p))
        provenance.append(dict(path=str(p),sha256=actual,expected_sha256=h,origin=origin))
    source=Path(inspect.getfile(roi_module));record(source,load(D4/'model_checks.json')['roi_source_sha'],'Day04 installed RoI implementation')
    source_text=source.read_text(encoding='utf-8');require('scores > self.score_thresh' in source_text and 'min_size=1e-2' in source_text,'Unexpected RoI source')
    (out/'roi_heads_source.py.txt').write_text(source_text,encoding='utf-8')
    for base,provname in [(D3,'input_provenance.json'),(D4,'provenance.json'),(D5,'input_provenance.json')]:
        require(load(base/'verification.json')['status']=='passed','Predecessor verification')
        # Integrity-check the recorded provenance; these bytes do not drive model inference.
        entries=load(base/provname)
        for e in entries:require(sha(e['path'])==e['sha256'],'Recorded predecessor input changed '+e['path'])
        record(base/provname)
        mf=load(base/'artifact_manifest.json');record(base/'artifact_manifest.json')
        for name in ['case_report_zh.md','verification.json','run_metadata.json']:record(base/name,mf[name],'Predecessor artifact manifest')
    d3manifest=load(D3/'artifact_manifest.json');record(D3/'IR_030078_raw_capture.pth',d3manifest['IR_030078_raw_capture.pth'],'Day03 reference capture')
    record(D3/'model_checks.json',d3manifest['model_checks.json'],'Day03 persistent state/config')
    COUNTS['reference_capture_loads']+=1
    old=torch.load(D3/'IR_030078_raw_capture.pth',map_location='cpu',weights_only=True)
    require(old['stem']==STEM and old['model']=='IR','Reference identity')
    expected_env=load(D3/'run_metadata.json')['environment']
    require(expected_env['torch']==torch.__version__ and expected_env['torchvision']==torchvision.__version__ and expected_env['python']==sys.version,'Current/Day03 environment differs')
    # Only original restore/data components and capture class are imported, no historic main script.
    sys.path.insert(0,str(ROOT/'week08/day03'));sys.path.insert(0,str(ROOT/'week09/day04'))
    import ir_common as ir
    from agnostic_capture import AgnosticCapture
    d3inputs={Path(e['path']):e['sha256'] for e in load(D3/'input_provenance.json')}
    for n in ['week08/day03/ir_common.py','week08/day01/llvip_dataset.py','week08/day02/paired_letterbox.py','week08/day02/paired_horizontal_flip.py','week09/day04/agnostic_capture.py','data/LLVIP_1000/manifest.csv']:
        p=ROOT/n;record(p,d3inputs[p],'Day03 actual source/input SHA')
    checkpoint=ROOT/'week08/day03/outputs/ir_baseline_run01/best.pth';expected_sha='9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2'
    record(checkpoint,expected_sha,'User-specified IR best SHA')
    ds=ir.LLVIPDataset(ROOT/'data/LLVIP_1000',split='val',modality='infrared',infrared_channels=3,transforms=ir.PairedLetterboxAndFlip('val'))
    index=next(i for i,r in enumerate(ds.records) if r.stem==STEM);rec=ds.get_record(index)
    for p in [rec.infrared_path,rec.visible_path,rec.annotation_path]:record(p,d3inputs[p],'Day03 original image/XML')
    image,target=ds[index];require(eq(target['boxes'],old['gt_all']) and bool((target['labels']==1).all()),'GT order/geometry/person label')
    with Image.open(rec.infrared_path) as im:original_wh=im.size
    params=ds.transforms.letterbox.parameters(original_wh[1],original_wh[0]);require(params.original_wh==(1280,1024) and params.padding_ltrb==(0,64,0,64),'Letterbox geometry')
    torch.set_num_threads(2);torch.manual_seed(42)
    original_load=torch.load;original_state=torch.nn.Module.load_state_dict
    def ckload(p,*a,**kw):
        require(Path(p).resolve()==checkpoint.resolve() and COUNTS['checkpoint_loads']==0,'Unexpected checkpoint load')
        COUNTS['checkpoint_loads']+=1;return original_load(p,*a,**kw)
    def strict(m,state,*a,**kw):
        require(kw.get('strict') is True,'strict=True restore required');COUNTS['strict_state_loads']+=1;return original_state(m,state,*a,**kw)
    def restore_probe(m,args):
        if type(m).__name__=='BackboneWithFPN':COUNTS['structure_probes']+=1
    with patch.object(torch,'load',ckload),patch.object(torch.nn.Module,'load_state_dict',strict),patch.object(torch.hub,'download_url_to_file',side_effect=RuntimeError('Download forbidden')):
        handle=torch.nn.modules.module.register_module_forward_pre_hook(restore_probe)
        try:COUNTS['model_restores']+=1;model,ck=ir.checkpoint_model(checkpoint,'cpu')
        finally:handle.remove()
    del ck;model.eval();cfg=config(model);initial=digest(model)
    require(normalized(cfg)==normalized(old['configuration']),'Actual model/Day03 config differs')
    oldstate=next(x['state_before'] for x in load(D3/'model_checks.json') if x['model']=='IR' and x['stem']==STEM)
    require(initial==oldstate,'Restored persistent state differs from Day03')
    require(cfg['rpn']['score']==.05 and cfg['rpn']['nms']==.7 and cfg['rpn']['pre']==cfg['rpn']['post']==150 and cfg['roi']['score']==.001 and cfg['roi']['nms']==.5 and cfg['roi']['detections']==100,'Actual thresholds/budgets')
    require(COUNTS['structure_probes']==0 and not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()) and any(isinstance(m,FrozenBatchNorm2d) for m in model.modules()),'IR FrozenBN or unexpected restoration forward')
    hooks0=[(m,dict(m._forward_hooks),dict(m._forward_pre_hooks)) for m in model.modules()]
    roi=model.roi_heads;cap=None;data={};original_post=roi.postprocess_detections
    wrapped_targets=[(model.rpn,k) for k in ['filter_proposals','_get_top_n_idx']]+[(roi,'postprocess_detections'),(roi.box_coder,'decode'),(roi_module.F,'softmax')]+[(roi_module.box_ops,k) for k in ['clip_boxes_to_image','remove_small_boxes','batched_nms']]
    attrs0=[(obj,k,k in obj.__dict__,obj.__dict__.get(k)) for obj,k in wrapped_targets]
    def post(logits,regression,proposals,shapes):
        require('class_logits' not in data,'Repeated RoI postprocess forbidden')
        data.update(class_logits=cp(logits),box_regression=cp(regression),post_proposals=cp(proposals),image_shapes=cp(shapes))
        def wrapped(obj,name,key):
            original=getattr(obj,name)
            def call(*a,**kw):
                require(key not in data,'Repeated RoI operation '+key)
                result=original(*a,**kw);data[key]=dict(args=cp(a),kwargs=cp(kw),result=cp(result))
                if key=='nms':COUNTS['original_roi_batched_nms_calls']+=1
                return result
            return call
        with ExitStack() as narrow:
            for obj,name,key in [(roi.box_coder,'decode','decode'),(roi_module.F,'softmax','softmax'),(roi_module.box_ops,'clip_boxes_to_image','clip'),(roi_module.box_ops,'remove_small_boxes','small'),(roi_module.box_ops,'batched_nms','nms')]:narrow.enter_context(patch.object(obj,name,wrapped(obj,name,key)))
            output=original_post(logits,regression,proposals,shapes)
        data['actual_post_output']=cp(output);return output
    try:
        with ExitStack() as stack:
            def modelpre(m,args):
                COUNTS['full_model_forward_calls']+=1;require(COUNTS['full_model_forward_calls']==1,'Only one full model forward authorized')
            def rpnpre(m,args):
                nonlocal cap
                images,features,targets=args;require(cap is None and images.image_sizes==[(640,640)] and tuple(features)==('0','1','pool'),'RPN actual geometry/features/call count')
                cap=stack.enter_context(AgnosticCapture(m,tuple(features),target['boxes'],f'{out.name}/IR/{STEM}'))
            def rpnpost(m,args,output):data['actual_rpn_return']=cp(output[0][0])
            def roipre(m,args):data['actual_roi_input']=cp(args[1]);data['actual_roi_shapes']=cp(args[2])
            def predictor(m,args,output):data['predictor_output']=cp(output)
            def transform(m,args,output):data['transform_sizes']=cp(output[0].image_sizes)
            for m,kind,fn in [(model,'pre',modelpre),(model.rpn,'pre',rpnpre),(model.rpn,'post',rpnpost),(roi,'pre',roipre),(roi.box_predictor,'post',predictor),(model.transform,'post',transform)]:
                h=m.register_forward_pre_hook(fn) if kind=='pre' else m.register_forward_hook(fn);stack.callback(h.remove)
            stack.enter_context(patch.object(roi,'postprocess_detections',post))
            COUNTS['full_forward_attempts']+=1
            with torch.inference_mode():prediction=model([image])[0]
        require(cap is not None and len(cap.records)==1,'One actual RPN capture required')
        COUNTS['original_rpn_batched_nms_calls']=len(cap.records)
        data.update(model_output=cp(prediction),gt_all=cp(target['boxes']),gt_labels=cp(target['labels']),input_image=cp(image),rpn_records=cp(cap.records),rpn_stages=cp(cap.states),catalogues=cp(cap.catalogues),configuration=normalized(cfg),run_identity=f'{out.name}/IR/{STEM}',model='IR',stem=STEM,gt_index=0,coordinates=dict(original_wh=list(original_wh),letterbox_wh=[640,640],resized_wh=list(params.resized_wh),padding_ltrb=list(params.padding_ltrb),scale=params.scale,internal_transform_wh=[640,640],final_output_wh=[640,640],output_coordinates='Letterbox input canvas, no repeated scaling or unpadding'))
        with (out/'raw_capture.pth').open('xb') as f:torch.save(data,f)
    finally:
        after_state=digest(model)
        hooks_restored=all(dict(m._forward_hooks)==a and dict(m._forward_pre_hooks)==b for m,a,b in hooks0)
        attrs_restored=all((k in obj.__dict__,obj.__dict__.get(k))==(present,value) for obj,k,present,value in attrs0)
        check=dict(state_before=initial,state_after=after_state,state_equal=initial==after_state,day03_initial_state_equal=initial==oldstate,configuration=normalized(cfg),configuration_unchanged=config(model)==cfg,day03_configuration_equal=normalized(cfg)==normalized(old['configuration']),hooks_restored=hooks_restored,temporary_methods_restored=attrs_restored,checkpoint_sha_before=expected_sha,checkpoint_sha_after=sha(checkpoint),checkpoint_unchanged=sha(checkpoint)==expected_sha,all_modules_eval=all(not m.training for m in model.modules()),frozen_bn_modules=sum(isinstance(m,FrozenBatchNorm2d) for m in model.modules()),counts=COUNTS)
        save(out/'model_checks.json',check)
    require(check['state_equal'] and check['configuration_unchanged'] and hooks_restored and attrs_restored and check['checkpoint_unchanged'],'Model/config/checkpoint/hooks verification failed')
    save(out/'input_provenance.json',provenance)
    del model
    return data,old,check

def analyze(out,data,old):
    proposal=data['actual_roi_input'][0];r=data['rpn_records'][0];flats=data['rpn_stages'][0]['post_nms_topn']['ids'].tolist();N=len(flats);g=data['gt_all'][0]
    require(eq(proposal,r['final_boxes']) and eq(proposal,data['actual_rpn_return']) and eq(proposal,data['post_proposals'][0]),'Actual RPN to actual RoI row identity')
    require(data['transform_sizes']==data['actual_roi_shapes']==data['image_shapes']==[[640,640]],'Same internal coordinate canvas')
    require(N==len(set(flats)) and bool((data['gt_labels']==1).all()),'Unique flat/person identity')
    logits=data['class_logits'];reg=data['box_regression'];prob=data['softmax']['result'];decoded=data['decode']['result'];clipped=data['clip']['result'];cfg=data['configuration']
    require(logits.shape==prob.shape==(N,2) and reg.shape==(N,8) and decoded.shape==clipped.shape==(N,2,4),'Class count/branch shape/order')
    require(eq(logits,data['predictor_output'][0]) and eq(reg,data['predictor_output'][1]) and eq(data['decode']['args'][0],reg) and eq(data['decode']['args'][1][0],proposal),'Actual predictor/decode row order')
    require(eq(data['softmax']['args'][0],logits) and eq(data['clip']['args'][0],decoded),'Actual softmax/clip input chain')
    expanded=clipped[:,1:].reshape(-1,4);scores=prob[:,1:].reshape(-1);labels=torch.ones(N,dtype=torch.int64);ids=torch.arange(N)
    scoreidx=torch.where(scores>cfg['roi']['score'])[0]
    require(eq(expanded[scoreidx],data['small']['args'][0]) and data['small']['kwargs']=={'min_size':.01},'Offline score index equals actual small filter input')
    smallidx=data['small']['result'];nids=scoreidx[smallidx];nb,ns,nl,threshold=data['nms']['args']
    require(eq(expanded[nids],nb) and eq(scores[nids],ns) and eq(labels[nids],nl) and threshold==.5,'Actual class NMS input chain')
    nmsidx=data['nms']['result'];kept=nids[nmsidx];finalids=kept[:cfg['roi']['detections']];post=data['actual_post_output'];pred=data['model_output']
    require(all(eq(x[finalids],y[0]) for x,y in zip((expanded,scores,labels),post)),'Original NMS indices/budget to actual postprocess return')
    require(all(eq(x,pred[k]) for x,k in zip((post[0][0],post[1][0],post[2][0]),('boxes','scores','labels'))),'Actual postprocess to final model coordinate/order chain')
    for k in ['boxes','scores']:require(bool(torch.isfinite(pred[k]).all()),'Finite model outputs')
    catalogue=data['catalogues'][0]
    def branchid(f,c=1):return catalogue[f]['candidate_id']+f'/class{c}'
    def overlaps(b):return box_iou(b,g[None])[:,0] if len(b) else torch.empty(0)
    branches=[]
    for i,f in enumerate(flats):
        for c in range(2):branches.append(dict(model='IR',stem=STEM,run_identity=data['run_identity'],proposal_row=i,flat=f,source_candidate_id=catalogue[f]['candidate_id'],branch_id=branchid(f,c),class_label=c,source_fpn_layer=catalogue[f]['level_key'],rpn_box=proposal[i].tolist(),rpn_objectness=float(r['final_scores'][i]),rpn_gt_iou=float(overlaps(proposal[i:i+1])[0]),class_logits=logits[i].tolist(),box_regression=reg[i].tolist(),background_probability=float(prob[i,0]),person_probability=float(prob[i,1]),branch_probability=float(prob[i,c]),decoded_box=decoded[i,c].tolist(),decoded_gt_iou=float(overlaps(decoded[i,c:c+1])[0]),clipped_box=clipped[i,c].tolist(),clipped_gt_iou=float(overlaps(clipped[i,c:c+1])[0])))
    stage_defs=[('rpn_proposals',proposal,ids,'actual RPN return / actual RoI input'),('person_decoded',decoded[:,1],ids,'actual decode capture, person branch slice'),('person_clipped',clipped[:,1],ids,'actual clip capture, person branch slice'),('background_removed_expanded',expanded,ids,'offline background removal/reshape; checked with next actual input'),('score_filtered',expanded[scoreidx],scoreidx,'offline strict score indices; equals actual small input'),('small_filtered',nb,nids,'actual small filter indices / actual NMS input'),('nms_all_output',nb[nmsidx],kept,'actual full class NMS return indices'),('detections_budget',post[0][0],finalids,'original budget slice checked against actual postprocess'),('model_output',pred['boxes'],finalids,'actual full model return')]
    records=[];coverage=[];indices={}
    for stage,boxes,rows,evidence in stage_defs:
        ov=overlaps(boxes);qualified=[flats[int(rows[j])] for j in range(len(rows)) if float(ov[j])>=.75];maxval=float(ov.max()) if len(ov) else None
        maxflats=[flats[int(rows[j])] for j in range(len(rows)) if float(ov[j])==maxval] if len(ov) else []
        coverage.append(dict(model='IR',stem=STEM,stage=stage,count=len(rows),max_gt_iou=maxval,covered=bool(qualified),qualifying_count=len(qualified),qualifying_flats=qualified,qualifying_branch_ids=[branchid(f) for f in qualified],maximum_iou_flats=maxflats,evidence=evidence))
        indices[stage]=dict(proposal_rows=rows.tolist(),flats=[flats[i] for i in rows.tolist()],evidence=evidence)
        for j,i in enumerate(rows.tolist()):records.append(dict(model='IR',stem=STEM,run_identity=data['run_identity'],stage=stage,stage_index=j,proposal_row=i,flat=flats[i],branch_id=branchid(flats[i]),class_label=1,xyxy=boxes[j].tolist(),gt_iou=float(ov[j]),covered=float(ov[j])>=.75,classification_score=None if stage=='rpn_proposals' else float(prob[i,1]),person_branch_score=float(prob[i,1]),rpn_objectness=float(r['final_scores'][i]),evidence=evidence))
    final_sources=[]
    for j,i in enumerate(finalids.tolist()):
        f=flats[i]
        final_sources.append(dict(model='IR',stem=STEM,run_identity=data['run_identity'],output_index=j,proposal_row=i,flat=f,source_candidate_id=catalogue[f]['candidate_id'],person_branch_id=branchid(f),class_label=int(pred['labels'][j]),classification_score=float(pred['scores'][j]),xyxy=pred['boxes'][j].tolist(),gt_iou=float(overlaps(pred['boxes'][j:j+1])[0]),final_good_for_G0=float(overlaps(pred['boxes'][j:j+1])[0])>=.75,rpn_xyxy=proposal[i].tolist(),rpn_gt_iou=float(overlaps(proposal[i:i+1])[0]),rpn_objectness=float(r['final_scores'][i]),person_decoded_xyxy=decoded[i,1].tolist(),person_decoded_gt_iou=float(overlaps(decoded[i,1:2])[0]),person_clipped_gt_iou=float(overlaps(clipped[i,1:2])[0])))
    final_good=[s for s in final_sources if s['final_good_for_G0']]
    evergood={f for c in coverage for f in c['qualifying_flats']};focus=sorted(evergood|{9141}|{s['flat'] for s in final_good})
    traces=[]
    for f in focus:
        require(f in flats,'Expected focus flat missing from new RPN')
        i=flats.index(f)
        for stage,_,_,evidence in stage_defs:
            hit=next((r for r in records if r['stage']==stage and r['flat']==f),None)
            traces.append(dict(model='IR',stem=STEM,flat=f,branch_id=branchid(f),proposal_row=i,stage=stage,present=hit is not None,stage_index=hit['stage_index'] if hit else None,xyxy=hit['xyxy'] if hit else None,gt_iou=hit['gt_iou'] if hit else None,covered=hit['covered'] if hit else False,classification_score=hit['classification_score'] if hit else None,person_branch_score=float(prob[i,1]),rpn_objectness=float(r['final_scores'][i]),evidence=evidence))
    outcomes=[]
    for f in sorted(evergood):
        i=flats.index(f);nmask=torch.where(nids==i)[0].tolist();kpos=torch.where(kept==i)[0].tolist();fpos=next((s['output_index'] for s in final_sources if s['flat']==f),None)
        absent=next((t['stage'] for t in traces if t['flat']==f and not t['present']),None)
        outcomes.append(dict(model='IR',stem=STEM,branch_id=branchid(f),flat=f,proposal_row=i,rpn_gt_iou=float(overlaps(proposal[i:i+1])[0]),decoded_gt_iou=float(overlaps(decoded[i,1:2])[0]),clipped_gt_iou=float(overlaps(clipped[i,1:2])[0]),person_score=float(prob[i,1]),good_at_rpn=float(overlaps(proposal[i:i+1])[0])>=.75,good_after_decode=float(overlaps(decoded[i,1:2])[0])>=.75,nms_input_position=nmask[0] if nmask else None,nms_kept=bool(kpos),nms_output_position=kpos[0] if kpos else None,final_output_index=fpos,final_good=any(s['flat']==f and s['final_good_for_G0'] for s in final_sources),first_absent_stage=absent))
    top_pos=int(ns.argmax()) if len(ns) else None;top_row=int(nids[top_pos]) if top_pos is not None else None
    top_detail=None
    if top_row is not None:
        f=flats[top_row];top_detail=dict(branch_id=branchid(f),flat=f,roi_row=top_row,nms_input_position=top_pos,classification_score=float(ns[top_pos]),gt_iou=float(overlaps(nb[top_pos:top_pos+1])[0]),good_for_G0=float(overlaps(nb[top_pos:top_pos+1])[0])>=.75,nms_kept=top_row in kept.tolist(),final_output_index=next((s['output_index'] for s in final_sources if s['flat']==f),None))
    precov=next(c for c in coverage if c['stage']=='small_filtered');max_pre=set(precov['maximum_iou_flats']);kept_flats={flats[i] for i in kept.tolist()}
    summary=dict(model='IR',stem=STEM,run_identity=data['run_identity'],proposal_count=N,final_count=len(final_sources),rpn_good_flats=coverage[0]['qualifying_flats'],decoded_good_flats=coverage[1]['qualifying_flats'],final_good_sources=final_good,final_best_iou_source=max(final_sources,key=lambda s:s['gt_iou']) if final_sources else None,final_top_score_source=max(final_sources,key=lambda s:s['classification_score']) if final_sources else None,nms_top_score_candidate=top_detail,nms_input_max_iou=precov['max_gt_iou'],nms_input_max_iou_flats=precov['maximum_iou_flats'],nms_keeps_any_pre_max_iou_branch=bool(max_pre&kept_flats),nms_keeps_all_pre_max_iou_branches=max_pre<=kept_flats,final_covered=bool(final_good),trace_9141=[t for t in traces if t['flat']==9141])
    reproduction=compare_day03(out,data,old)
    summary['day03_exact_reproduced']=reproduction['exact_reproduced']
    require(COUNTS['full_model_forward_calls']==COUNTS['checkpoint_loads']==COUNTS['strict_state_loads']==1 and COUNTS['structure_probes']==0,'Execution count constraint')
    for name,rr in [('proposal_branches',branches),('stage_candidates',records),('stage_coverage',coverage),('focus_traces',traces),('trace_9141',[t for t in traces if t['flat']==9141]),('final_good_source_traces',[t for t in traces if any(s['flat']==t['flat'] for s in final_good)]),('good_branch_outcomes',outcomes),('final_sources',final_sources)]:csvout(out/(name+'.csv'),rr)
    save(out/'stage_indices.json',dict(flat_sequence=flats,score_filter=scoreidx.tolist(),small_filter=smallidx.tolist(),nms_full=nmsidx.tolist(),final_proposal_rows=finalids.tolist(),stages=indices));save(out/'success_summary.json',summary)
    return summary,coverage,records,branches,traces,outcomes,final_sources,reproduction

def compare_day03(out,data,old):
    new=data['rpn_records'][0];previous=old['records'][0];ids=data['rpn_stages'][0]['post_nms_topn']['ids'].tolist();oldids=old['stages'][0]['post_nms_topn']['ids'].tolist();lookup={f:i for i,f in enumerate(oldids)};newlookup={f:i for i,f in enumerate(ids)}
    diffs=[]
    for f in sorted(set(ids)|set(oldids)):
        i=newlookup.get(f);j=lookup.get(f);both=i is not None and j is not None
        diffs.append(dict(model='IR',stem=STEM,flat=f,current_row=i,day03_row=j,order_position_equal=both and i==j,box_equal=both and eq(new['final_boxes'][i],previous['final_boxes'][j]),score_equal=both and eq(new['final_scores'][i],previous['final_scores'][j]),current_box=new['final_boxes'][i].tolist() if i is not None else None,day03_box=previous['final_boxes'][j].tolist() if j is not None else None,coordinate_delta=(new['final_boxes'][i]-previous['final_boxes'][j]).tolist() if both else None,current_objectness=float(new['final_scores'][i]) if i is not None else None,day03_objectness=float(previous['final_scores'][j]) if j is not None else None,objectness_delta=float(new['final_scores'][i]-previous['final_scores'][j]) if both else None))
    pred=data['model_output'];oldpred=old['model_output'];finaldiff=[]
    for i in range(max(len(pred['boxes']),len(oldpred['boxes']))):
        both=i<len(pred['boxes']) and i<len(oldpred['boxes'])
        finaldiff.append(dict(output_index=i,boxes_equal=both and eq(pred['boxes'][i],oldpred['boxes'][i]),scores_equal=both and eq(pred['scores'][i],oldpred['scores'][i]),labels_equal=both and eq(pred['labels'][i],oldpred['labels'][i]),current_box=pred['boxes'][i].tolist() if i<len(pred['boxes']) else None,day03_box=oldpred['boxes'][i].tolist() if i<len(oldpred['boxes']) else None,box_delta=(pred['boxes'][i]-oldpred['boxes'][i]).tolist() if both else None,current_score=float(pred['scores'][i]) if i<len(pred['scores']) else None,day03_score=float(oldpred['scores'][i]) if i<len(oldpred['scores']) else None,score_delta=float(pred['scores'][i]-oldpred['scores'][i]) if both else None))
    result=dict(rpn_current_count=len(ids),rpn_day03_count=len(oldids),flat_set_equal=set(ids)==set(oldids),flat_sequence_equal=ids==oldids,rpn_boxes_scores_equal_by_flat=all(r['box_equal'] and r['score_equal'] for r in diffs),final_current_count=len(pred['boxes']),final_day03_count=len(oldpred['boxes']),final_boxes_scores_labels_order_exact=all(eq(pred[k],oldpred[k]) for k in ['boxes','scores','labels']),GT_all_exact=eq(data['gt_all'],old['gt_all']),internal_coordinates_equal=data['transform_sizes']==old['observed']['transform_sizes'],configuration_equal=normalized(data['configuration'])==normalized(old['configuration']),tolerance_used=False)
    result['exact_reproduced']=all(result[k] for k in ['flat_sequence_equal','rpn_boxes_scores_equal_by_flat','final_boxes_scores_labels_order_exact','GT_all_exact','internal_coordinates_equal','configuration_equal'])
    csvout(out/'day03_flat_comparison.csv',diffs);csvout(out/'day03_final_comparison.csv',finaldiff);save(out/'day03_reproduction.json',result)
    return result

def make_comparison(out,summary,coverage,records):
    # Failure case is read from tables only. No model creation, forward, or NMS replay here.
    bases=[(D4,'stage_coverage.csv'),(D4,'stage_candidates.csv'),(D4,'proposal_branches.csv'),(D4,'final_sources.csv'),(D4,'trace_8877.csv'),(D5,'nms_inputs.csv'),(D5,'focus_attributions.csv')]
    sources=[]
    for base,name in bases:
        expected=load(base/'artifact_manifest.json')[name];require(sha(base/name)==expected,'Comparison table changed '+name);sources.append(dict(path=str(base/name),sha256=expected))
    save(out/'comparison_sources.json',sources)
    failure_cov=read(D4/'stage_coverage.csv');failure_rows=read(D4/'stage_candidates.csv');failure_final=read(D4/'final_sources.csv');failure_nms=read(D5/'nms_inputs.csv');failure_trace=read(D4/'trace_8877.csv')
    def table_summary(stem,cov,rr,final,nms,original_good):
        is_failure=stem=='030129'
        def cv(stage):return next(c for c in cov if c['stage']==stage)
        def intlist(v):return json.loads(v) if isinstance(v,str) else v
        def bf(f):return f['candidate_id']+'/class1' if is_failure else f['person_branch_id']
        final_best=max(final,key=lambda f:float(f['gt_iou'])) if final else None
        final_good=[f for f in final if float(f['gt_iou'])>=.75]
        top=max(nms,key=lambda r:float(r['classification_score'])) if nms else None
        topflat=int(top['source_rpn_flat'] if is_failure else top['flat']) if top else None
        top_out=next((int(f['output_index']) for f in final if int(f['flat'])==topflat),None)
        good_rpn=intlist(cv('rpn_proposals')['qualifying_flats']);good_dec=intlist(cv('person_decoded')['qualifying_flats']);good_nms=intlist(cv('small_filtered')['qualifying_flats']);good_after=intlist(cv('nms_all_output')['qualifying_flats'])
        maxflats=[int(r['flat']) for r in nms if float(r['gt_iou'])==float(cv('small_filtered')['max_gt_iou'])] if not is_failure else [int(r['source_rpn_flat']) for r in nms if float(r['gt_iou'])==float(cv('small_filtered')['max_gt_iou'])]
        finalflats={int(f['flat']) for f in final}
        original_records=[r for r in rr if int(r['flat'])==original_good]
        original_dec=next(r for r in original_records if r['stage']=='person_decoded')
        original_nms=next((r for r in original_records if r['stage']=='small_filtered'),None)
        original_final=next((f for f in final if int(f['flat'])==original_good),None)
        full=lambda f:f'IR/{stem}/{f}'
        return dict(model='IR',stem=stem,original_run_identity='roi_trace_run01/IR/030129' if is_failure else summary['run_identity'],rpn_count=int(cv('rpn_proposals')['count']),rpn_max_iou=float(cv('rpn_proposals')['max_gt_iou']),rpn_good_identities=[full(f) for f in good_rpn],decoded_max_iou=float(cv('person_decoded')['max_gt_iou']),decoded_good_count=len(good_dec),decoded_good_identities=[full(f) for f in good_dec],nms_input_count=int(cv('small_filtered')['count']),nms_before_good_count=len(good_nms),nms_before_good_identities=[full(f) for f in good_nms],nms_after_good_count=len(good_after),nms_after_good_identities=[full(f) for f in good_after],nms_highest_iou_identities=[full(f) for f in maxflats],nms_highest_iou_any_retained=any(f in finalflats for f in maxflats),nms_top_score_identity=full(topflat) if top else None,nms_top_classification_score=float(top['classification_score']) if top else None,nms_top_gt_iou=float(top['gt_iou']) if top else None,nms_top_kept=top_out is not None,nms_top_final_index=top_out,final_count=len(final),final_covered=bool(final_good),final_good_identities=[full(int(f['flat'])) for f in final_good],final_best_iou_identity=full(int(final_best['flat'])) if final_best else None,final_best_iou=float(final_best['gt_iou']) if final_best else None,final_best_iou_score=float(final_best['score'] if is_failure else final_best['classification_score']) if final_best else None,original_rpn_good_identity=full(original_good),original_rpn_good_decoded_iou=float(original_dec['gt_iou']),original_rpn_good_person_score=float(original_dec['classification_score']),original_rpn_good_present_before_nms=original_nms is not None,original_rpn_good_final_index=int(original_final['output_index']) if original_final else None,original_rpn_good_final_iou=float(original_final['gt_iou']) if original_final else None,source_tables='Day04 and Day05 saved tables only' if is_failure else 'Day06 actual capture exported tables')
    newfinal=read(out/'final_sources.csv');newrecords=read(out/'stage_candidates.csv');newcov=read(out/'stage_coverage.csv');newnms=[r for r in newrecords if r['stage']=='small_filtered']
    result=[table_summary(STEM,newcov,newrecords,newfinal,newnms,9141),table_summary('030129',failure_cov,failure_rows,failure_final,failure_nms,8877)]
    # The known failure target table is cross-checked rather than simply trusted in prose.
    require(next(r for r in failure_trace if r['stage']=='person_decoded')['gt_iou']==str(result[1]['original_rpn_good_decoded_iou']),'Failure reference trace/table disagreement')
    csvout(out/'two_case_comparison.csv',result)
    return result,failure_cov,failure_final,failure_rows

def figures(out,data,summary,coverage,records,traces,comparison,failure_rows,failure_final):
    def text(d,xy,s,size=21,color='#26354a'):d.text(xy,s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size),fill=color)
    image=Image.fromarray(data['input_image'].mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy());gts=data['gt_all'].tolist()
    rpn=next(r for r in records if r['stage']=='rpn_proposals' and r['flat']==9141);dec=next(r for r in records if r['stage']=='person_decoded' and r['flat']==9141)
    final_records=[r for r in records if r['stage']=='model_output' and r['covered']]
    best_pre=next(r for r in records if r['stage']=='small_filtered' and r['gt_iou']==summary['nms_input_max_iou']) if summary['nms_input_max_iou'] is not None else None
    chosen=[('9141原RPN',rpn),('9141 person解码',dec)]+[(f'最终达标来源{r["flat"]}',r) for r in final_records]
    if best_pre is not None:chosen.append((f'NMS前最高IoU来源{best_pre["flat"]}',best_pre))
    bb=gts+[r['xyxy'] for _,r in chosen];bounds=(max(0,math.floor(min(b[0] for b in bb))-18),max(0,math.floor(min(b[1] for b in bb))-18),min(640,math.ceil(max(b[2] for b in bb))+18),min(640,math.ceil(max(b[3] for b in bb))+18))
    def drawcard(canvas,xy,img,GT,box,title,lines,bounds,color='#00b4a5'):
        x,y=xy;d=ImageDraw.Draw(canvas);text(d,(x+14,y+5),title,23,color)
        crop=img.crop(bounds);cd=ImageDraw.Draw(crop);left,top,right,bottom=bounds
        for j,b in enumerate(GT):cd.rectangle((b[0]-left,b[1]-top,b[2]-left,b[3]-top),outline='#ffd24a' if j==0 else '#a8b4c4',width=1)
        cd.rectangle((box[0]-left,box[1]-top,box[2]-left,box[3]-top),outline=color,width=2)
        scale=min(450/(right-left),320/(bottom-top));w,h=round((right-left)*scale),round((bottom-top)*scale);canvas.paste(crop.resize((w,h),Image.Resampling.BILINEAR),(x+(480-w)//2,y+45))
        for j,line in enumerate(lines):text(d,(x+14,y+379+j*31),line,19)
    canvas=Image.new('RGB',(1440,120+math.ceil(len(chosen)/3)*530),'#f1f5fa');d=ImageDraw.Draw(canvas)
    text(d,(22,15),'030078/IR/G0：真实分支身份；黄G0 / 灰其他GT / 青RPN / 粉RoI',27)
    text(d,(22,63),'每张卡单独画框；所有坐标均为640×640含padding，局部裁剪仅展示。',21)
    for j,(title,r) in enumerate(chosen):
        lines=[f'完整来源 IR/030078/{r["flat"]}',f'G0 IoU={r["gt_iou"]:.9f}',f'objectness={r["rpn_objectness"]:.9f}' if r['stage']=='rpn_proposals' else f'分类分数={r["classification_score"]:.9f}',f'阶段 {r["stage"]}']
        drawcard(canvas,((j%3)*480,110+(j//3)*530),image,gts,r['xyxy'],title,lines,bounds,'#00b4a5' if r['stage']=='rpn_proposals' else '#e36f9a')
    canvas.save(out/'key_candidates.png')
    canvas=Image.new('RGB',(1480,120+len(coverage)*92),'#f1f5fa');d=ImageDraw.Draw(canvas)
    text(d,(22,15),'030078/IR：全体person分支覆盖与真实索引链',29)
    for j,c in enumerate(coverage):
        y=75+j*92;d.rectangle((22,y,1450,y+76),outline='#8ca4bb',width=2)
        text(d,(36,y+5),f'{j}. {c["stage"]}   数量={c["count"]}  最大IoU={c["max_gt_iou"]}',21)
        text(d,(36,y+39),'达标来源flat='+str(c['qualifying_flats']),19)
    canvas.save(out/'stage_flow.png')
    # Read only the failure image and saved XML/geometry; no failure model or inference.
    sys.path.insert(0,str(ROOT/'week08/day01'));sys.path.insert(0,str(ROOT/'week08/day02'))
    from llvip_dataset import _load_infrared,_parse_voc_annotation
    from paired_letterbox import PairedLetterbox
    path=ROOT/'data/LLVIP_1000/images/val/infrared/030129.jpg';xml=ROOT/'data/LLVIP_1000/annotations_voc/val/030129.xml'
    tensor,_=_load_infrared(path,3);target,_=_parse_voc_annotation(xml,0,'030129');_,tensor,target=PairedLetterbox(640)(tensor,tensor,target)
    failure_image=Image.fromarray(tensor.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy());failure_gt=target['boxes'].tolist()
    candidates=[]
    for row in failure_rows:
        if row['stage']=='model_output':require(row['xyxy']==next(f['xyxy'] for f in failure_final if f['output_index']==row['stage_index']),'Failure figure final table correspondence')
    canvas=Image.new('RGB',(1500,900),'#f1f5fa');d=ImageDraw.Draw(canvas)
    text(d,(20,15),'两例仅作描述性对照：粉色为最终最贴合G0的实际输出，黄色G0',26)
    text(d,(20,60),'030129使用Day04保存表和原图片，不重新推理；身份跨图片分别标注。',21)
    for col,c in enumerate(comparison):
        stem=c['stem'];img=image if stem==STEM else failure_image;GT=gts if stem==STEM else failure_gt
        if stem==STEM:src=summary['final_best_iou_source'];box=src['xyxy']
        else:src=max(failure_final,key=lambda r:float(r['gt_iou']));box=json.loads(src['xyxy'])
        boxes=GT+[box];bnd=(max(0,math.floor(min(b[0] for b in boxes))-20),max(0,math.floor(min(b[1] for b in boxes))-20),min(640,math.ceil(max(b[2] for b in boxes))+20),min(640,math.ceil(max(b[3] for b in boxes))+20))
        x=20+col*750;crop=img.crop(bnd);cd=ImageDraw.Draw(crop)
        for j,b in enumerate(GT):cd.rectangle((b[0]-bnd[0],b[1]-bnd[1],b[2]-bnd[0],b[3]-bnd[1]),outline='#ffd24a' if j==0 else '#a8b4c4',width=1)
        cd.rectangle((box[0]-bnd[0],box[1]-bnd[1],box[2]-bnd[0],box[3]-bnd[1]),outline='#e36f9a',width=2)
        scale=min(690/(bnd[2]-bnd[0]),450/(bnd[3]-bnd[1]));w,h=round((bnd[2]-bnd[0])*scale),round((bnd[3]-bnd[1])*scale)
        canvas.paste(crop.resize((w,h),Image.Resampling.BILINEAR),(x+(700-w)//2,140));text(d,(x,100),c['final_best_iou_identity'],24)
        for j,line in enumerate([f'最终最大IoU={c["final_best_iou"]:.9f}',f'该框分类分数={c["final_best_iou_score"]:.9f}',f'NMS前/后达标框数：{c["nms_before_good_count"]}/{c["nms_after_good_count"]}',f'最高分输入的GT IoU={c["nms_top_gt_iou"]:.9f}',f'原RPN好框person IoU={c["original_rpn_good_decoded_iou"]:.9f}']):text(d,(x,620+j*38),line,21)
        candidates.append(dict(stem=stem,zoom_xyxy=list(bnd)))
    text(d,(20,845),'不同图片不是控制变量实验；不估计总体比例、因果效应或AP75。',22);canvas.save(out/'two_case_key_boxes.png')
    save(out/'figure_geometry.json',dict(success_coordinates=data['coordinates'],success_common_zoom_xyxy=list(bounds),full_success_GT=gts,failure_GT_from_original_XML=failure_gt,two_case_zooms=candidates,crops_for_display_only=True))

def write_report(out,summary,coverage,outcomes,sources,traces,comparison,reproduction):
    target=summary['trace_9141'];targetfinal=next(t for t in target if t['stage']=='model_output');targetdecoded=next(t for t in target if t['stage']=='person_decoded');good=summary['final_good_sources'];orig=next(t for t in target if t['stage']=='rpn_proposals')
    final_description='；'.join(f'{s["source_candidate_id"]}/class1（RoI行{s["proposal_row"]}，输出索引{s["output_index"]}，IoU={s["gt_iou"]:.9f}，分类分数={s["classification_score"]:.9f}）' for s in good) or '本次没有最终达标框'
    absent=next((t['stage'] for t in target if not t['present']),None)
    lines=['# Week10 Day06：030078/IR成功案例的RoI身份与两例对照','',
           f'最终达标框的真实来源：{final_description}。来源由RPN flat、实际RoI输入行和各阶段索引直接串联，未按几何相似度猜测。',
           f'9141原RPN对G0 IoU={orig["gt_iou"]:.9f}、objectness={orig["rpn_objectness"]:.9f}；person解码后IoU={targetdecoded["gt_iou"]:.9f}、分类分数={targetdecoded["person_branch_score"]:.9f}。最终是否仍存在：{targetfinal["present"]}；最终IoU={targetfinal["gt_iou"]}；首次不再存在的阶段={absent}（null/None表示一直存在）。定位下降与筛选删除分别看表，二者不混同。','',
           f'Day03完整复现：{reproduction["exact_reproduced"]}。只有True才可把本次捕获作为Day03的内部过程说明；False时仅解释本次新CPU运行，所有差异原样保留，不追加前向。','',
           '## 先分清两个结论','',
           f'最终是否保住G0的0.75覆盖：{summary["final_covered"]}。NMS前最高IoU={summary["nms_input_max_iou"]}，对应来源flat={summary["nms_input_max_iou_flats"]}；NMS是否保住至少一个最高IoU分支：{summary["nms_keeps_any_pre_max_iou_branch"]}。达到0.75与保住最贴合的那个框是两件事。','',
           '## 全36个proposal的阶段覆盖','','|阶段|候选数|最大G0 IoU|达标flat|','|---|---:|---:|---|']
    for c in coverage:lines.append(f'|{c["stage"]}|{c["count"]}|{c["max_gt_iou"]}|{c["qualifying_flats"]}|')
    lines+=['','## 所有阶段曾达标的分支结局','','|完整来源|RPN IoU|person解码IoU|分类分数|原NMS保留|最终输出索引|最终达标|首次缺失阶段|','|---|---:|---:|---:|---|---:|---|---|']
    for s in outcomes:lines.append(f'|{s["branch_id"]}|{s["rpn_gt_iou"]:.9f}|{s["decoded_gt_iou"]:.9f}|{s["person_score"]:.9f}|{s["nms_kept"]}|{s["final_output_index"]}|{s["final_good"]}|{s["first_absent_stage"]}|')
    top=summary['nms_top_score_candidate']
    if top:lines+=['',f'NMS输入中最高分类分数候选为{top["branch_id"]}，分数={top["classification_score"]:.9f}、G0 IoU={top["gt_iou"]:.9f}；原NMS保留={top["nms_kept"]}，最终输出索引={top["final_output_index"]}。这条记录来自全部实际NMS输入，不是从最终输出倒推。']
    lines+=['','## 全部最终输出的真实来源','','|索引|完整来源|原RPN IoU|person解码IoU|分类分数|最终IoU|','|---:|---|---:|---:|---:|---:|']
    for s in sources:lines.append(f'|{s["output_index"]}|{s["person_branch_id"]}|{s["rpn_gt_iou"]:.9f}|{s["person_decoded_gt_iou"]:.9f}|{s["classification_score"]:.9f}|{s["gt_iou"]:.9f}|')
    lines+=['','## 两张图片的描述性对照','','|核查字段|030078/IR（本次）|030129/IR（原保存表）|','|---|---|---|']
    a,b=comparison
    for key,label in [('rpn_max_iou','RPN最大IoU'),('rpn_good_identities','RPN达标候选'),('decoded_max_iou','person解码最大IoU'),('decoded_good_count','解码后达标数'),('nms_before_good_count','NMS前达标数'),('nms_after_good_count','NMS后达标数'),('nms_top_score_identity','NMS最高分类分数来源'),('nms_top_classification_score','该候选分类分数'),('nms_top_gt_iou','该候选G0 IoU'),('nms_top_kept','该候选是否保留'),('final_best_iou_identity','最终最贴合来源'),('final_best_iou','最终最贴合IoU'),('final_best_iou_score','最终最贴合分类分数'),('nms_highest_iou_any_retained','是否保住NMS前最高IoU分支'),('original_rpn_good_identity','原RPN好框'),('original_rpn_good_decoded_iou','原RPN好框解码后IoU'),('original_rpn_good_person_score','原RPN好框person分数'),('original_rpn_good_final_index','原RPN好框最终索引')]:lines.append(f'|{label}|{a[key]}|{b[key]}|')
    lines+=['',
            f'实际处理差异：030078在NMS前有{a["nms_before_good_count"]}个达标分支，完整NMS后保留{a["nms_after_good_count"]}个；030129在NMS前有{b["nms_before_good_count"]}个，NMS后为{b["nms_after_good_count"]}。两图原RPN好框的回归结果和最终达标分支的真实来源也分别列出，不能因为端点IoU接近就把两个端点当作同一个候选。',
            '030129的8877回归后IoU降为0.625646、分类分数0.995233，原NMS删除；另外5个回归达标框也在最终类别NMS处丢失。其具体首次删除归因只引用Day05已有独立重放：8576、9156、12831、2576在step0由8862首次删除，8571在step2由2571首次删除；这是经过原完整输出匹配的重放解释，不是原内核事件记录。本轮未重新执行该重放。',
            '不能概括为成功案例所有候选的分类与定位都更一致。应分别看实际候选：哪一个最高分框对G0达标、哪一个最高IoU框仍被删除、最终达标来自哪个分支。两张图不是控制变量实验，不能据此估计总体比例、因果效应或AP75提升。','',
            '## 事实、重建与未知','',
            '实际捕获事实：RPN实际输出、RoI实际输入、predictor logits与regression、原decode/softmax/clip结果、小框和类别NMS完整输入与索引、原postprocess及完整模型返回。所有返回值原样交回原调用，GT只用于分析。RPN与类别NMS是本次原模型前向中的两次原调用，额外独立贪心NMS重放为0。',
            '经核对的离线重建：去背景/按类别展开、严格score>0.001的过滤索引与预算切片，分别与下一实际小框输入、实际NMS输入、实际postprocess返回逐项精确核对。没有新增argmax类别筛选。9141与所有最终达标分支的每一步完整状态见追踪CSV，全部proposal与背景/person分支未省略。',
            '原NMS保留索引直接确认删除阶段和保留集合；本轮没有独立重放或内核逐次事件，所以不命名030078中被删分支的实际抑制者。尚不能解释模型为何对某个分支形成这样的回归量和分类分数，也不能仅凭两例推断融合或模型根因。成功仅指指定G0最终存在IoU≥0.75框，不代表整图或整个模型成功。',
            '坐标：原图1280×1024，Letterbox缩放0.5为640×512，上下各64 padding；模型输入、内部transform及最终输出都是640×640画布，不重复去padding。完整GT保留，对G0不达标不等于背景。图中的局部裁剪仅展示，不重新推理。',
            '实际执行counts、模型持久状态/checkpoint/config/hook检查、来源SHA、环境、命令、日志和退出码分别在verification.json、model_checks.json、input_provenance.json、run_metadata.json及run.log中。030129仅读取已有表格和图像，前向0次。核验失败时保留证据并标记，不追加前向。','',
            '![关键分支](key_candidates.png)','![阶段覆盖](stage_flow.png)','![两例对照](two_case_key_boxes.png)','', '本限定步骤完成后停止，供审阅与教学。']
    (out/'case_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();require(out.is_relative_to(HERE) and out!=HERE,'Output outside day06');out.mkdir(parents=True,exist_ok=False)
    folders=[ROOT/'week08',ROOT/'week09']+[ROOT/f'week10/day{i:02}' for i in range(1,6)]
    extras=[ROOT/'AGENTS.md',ROOT/'data/LLVIP_1000/manifest.csv',Path(inspect.getfile(roi_module))]
    for stem in [STEM,'030129']:extras.extend([ROOT/f'data/LLVIP_1000/images/val/{mode}/{stem}.jpg' for mode in ['visible','infrared']]+[ROOT/f'data/LLVIP_1000/annotations_voc/val/{stem}.xml'])
    def snapshot():return {str(p):sha(p) for p in sorted(set(extras+[p for folder in folders for p in folder.rglob('*') if p.is_file()]))}
    before=snapshot();save(out/'history_before.json',before);provenance=[];result={};code=1;started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    class Tee:
        def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
        def flush(self):log.flush()
    with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
        try:
            data,old,check=capture(out,provenance)
            print('ONLY_FORWARD_CAPTURED',COUNTS,flush=True)
            summary,coverage,records,branches,traces,outcomes,sources,reproduction=analyze(out,data,old)
            comparison,fcov,ffinal,frows=make_comparison(out,summary,coverage,records)
            figures(out,data,summary,coverage,records,traces,comparison,frows,ffinal)
            write_report(out,summary,coverage,outcomes,sources,traces,comparison,reproduction)
            result=dict(status='passed' if reproduction['exact_reproduced'] else 'failed',scope='030078/IR/G0 new capture; 030129/IR/G0 saved-table comparison',day03_exact_reproduced=reproduction['exact_reproduced'],identity_chain_verified=True,all_proposals_saved=True,proposal_count=summary['proposal_count'],class_branch_rows=len(branches),final_count=summary['final_count'],final_good_sources=summary['final_good_sources'],final_covered=summary['final_covered'],nms_highest_iou_branch_retained=summary['nms_keeps_any_pre_max_iou_branch'],state_config_checkpoint_hooks_verified=True,comparison_from_tables=True,independent_nms_replays=0)
            code=0 if result['status']=='passed' else 1
        except BaseException:
            traceback.print_exc();save(out/'failure.json',dict(error=traceback.format_exc(),counts=COUNTS))
        finally:
            after=snapshot();unchanged=before==after;save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
            if not unchanged:code=1
            result.update(status='passed' if code==0 else 'failed',historical_files_unchanged=unchanged,protected_file_count=len(before),counts=COUNTS,exit_code=code);save(out/'verification.json',result)
            save(out/'run_metadata.json',dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment=environment(),source_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'README.md',HERE/'run_trace.ps1']},counts=COUNTS,exit_code=code))
            print(json.dumps(result,ensure_ascii=False),flush=True);print('EXIT',code,flush=True)
    save(out/'artifact_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()});return code

if __name__=='__main__':sys.exit(main())
