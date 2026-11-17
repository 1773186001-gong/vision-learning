"""Independent full-input greedy replay. Original keep indices used only AFTER replay."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,datetime,hashlib,json,math,platform,struct,subprocess,traceback
from pathlib import Path
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from unittest.mock import patch
import torch,torchvision
from torchvision.ops import boxes as ops
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
D1=ROOT/'week10/day01/outputs/candidate_review_run01'
D4=ROOT/'week09/day04/outputs/full_val_stage_run01'
SOURCE=ROOT/'week09/day03/nms_relationship_offline_run01/results/nms_source_readonly.txt'
CASES=(('030129',8877,8872,80),('030078',9141,8841,65))
KEYS=('0','1','pool')
COUNTS={'main_replay_calls':0,'full_output_verifications':0}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def csvout(p,rr,fields):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def require(ok,note):
    if not ok:raise ValueError(note)
def q(v):return struct.unpack('f',struct.pack('f',v))[0]
def iou(a,b,f32=True):
    f=q if f32 else float
    aa=f(f(a[2]-a[0])*f(a[3]-a[1]));bb=f(f(b[2]-b[0])*f(b[3]-b[1]))
    inter=f(max(0.,f(min(a[2],b[2])-max(a[0],b[0])))*max(0.,f(min(a[3],b[3])-max(a[1],b[1]))))
    return f(inter/f(f(aa+bb)-inter))

def greedy_replay(boxes,scores,levels,threshold):
    """No GT, IDs, saved keep indices, or outcome labels can enter this function."""
    require(boxes.dtype==scores.dtype==torch.float32,'float32 algorithm inputs required')
    bb=boxes.tolist();ss=scores.tolist();ll=levels.tolist()
    maximum=max(max(b) for b in bb);stride=q(maximum+q(1.))
    offsets=[q(q(l)*stride) for l in ll]
    shifted=[[q(v+offsets[i]) for v in b] for i,b in enumerate(bb)]
    # Fixed before inspecting any output. Same-score ambiguity is audited separately.
    order=sorted(range(len(ss)),key=lambda i:(-ss[i],i))
    active=list(order);keep=[];steps=[];comparisons=[];deleted_by={}
    while active:
        a=active[0];step=len(keep);keep.append(a);remaining=[];deleted=[]
        for b in active[1:]:
            x=iou(shifted[a],shifted[b]);decision=x>q(threshold)
            comparisons.append(dict(step=step,selected_position=a,other_position=b,path_iou=x,deleted=decision))
            if decision:deleted.append(b);deleted_by[b]=(step,a)
            else:remaining.append(b)
        steps.append(dict(step=step,selected_position=a,new_deleted_positions=deleted))
        active=remaining
    return dict(keep=keep,steps=steps,comparisons=comparisons,deleted_by=deleted_by,shifted=shifted,order=order,max_coordinate=maximum,offset_stride=stride,offsets=offsets)

def tensor_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)

def run(out):
    sys.path.insert(0,str(ROOT/'week08/day01'));sys.path.insert(0,str(ROOT/'week08/day02'))
    from llvip_dataset import LLVIPDataset
    from paired_letterbox import PairedLetterbox
    require(load(D1/'verification.json')['status']=='passed','Day01 not passed')
    d1prov=load(D1/'input_provenance.json');provenance=[]
    used={Path(x['path']):x['sha256'] for x in d1prov['inputs']}
    def record(p,expected=None,origin='current input'):
        h=sha(p)
        if expected is not None:require(h==expected,'SHA mismatch '+str(p))
        provenance.append(dict(path=str(p),sha256=h,expected_sha256=expected,origin=origin))
    for n in ['case_report_zh.md','case_summary.csv','input_provenance.json','verification.json']:
        record(D1/n,load(D1/'artifact_manifest.json')[n],'Day01 artifact manifest')
    record(ROOT/'week10/day01/README.md')
    ledger=read(D4/'forward_ledger.csv');record(D4/'forward_ledger.csv',used[D4/'forward_ledger.csv'],'Day01 recorded Day04 ledger')
    saved=SOURCE.read_text(encoding='utf-8-sig');record(SOURCE)
    require('boxes + offsets[:, None]' in saved and '100_000' in saved and 'decreasing order of scores' in saved,'Missing algorithm evidence')
    oldenv=load(D4/'run_metadata.json')['environment']
    sourcenv=load(SOURCE.parent/'run_metadata.json')['environment']
    record(D4/'run_metadata.json',used[D4/'run_metadata.json'],'Day01 recorded original CUDA metadata')
    record(SOURCE.parent/'run_metadata.json')
    record(D4/'source/agnostic_capture.py',used[D4/'source/agnostic_capture.py'],'Day04 saved capture source')
    evidence=dict(original_capture_environment=oldenv,saved_python_source_environment=sourcenv,current_environment=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,device='CPU'),saved_source_origin='Day03 local read-only source snapshot; not CUDA compiled kernel',path='small input coordinate trick: max + 1, float32 level offsets, addition then global descending-score greedy',sort='descending original score; predetermined input-index tie break',deletion='shifted float32 IoU > float32(0.7)',missing_evidence=['original GPU per-step suppression events','original compiled CUDA kernel source/build arithmetic details'],interpretation='full-output matching is necessary for replay attribution, not proof of original GPU event identity')
    save(out/'implementation_evidence.json',evidence)
    (out/'nms_source_evidence.txt').write_text(saved,encoding='utf-8')
    expected=load(D4/'preflight/expected_inputs.json')['sha256']
    record(D4/'preflight/expected_inputs.json',used[D4/'preflight/expected_inputs.json'],'Day01 source evidence')
    for n in ['week08/day01/llvip_dataset.py','week08/day02/paired_letterbox.py','data/LLVIP_1000/manifest.csv']:record(ROOT/n,expected[n],'Day04 original preflight')
    ds=LLVIPDataset(ROOT/'data/LLVIP_1000',split='val',modality='paired',transforms=PairedLetterbox(640))
    all_inputs=[];events=[];steps=[];checks=[];pairs=[];ties=[];sensitive=[];relations=[];survival=[];verifications=[];summaries=[];geometry=[]
    for stem,goodflat,targetflat,expected_n in CASES:
        entry=next(x for x in ledger if x['model']=='IR' and x['stem']==stem);p=D4/entry['relative_raw']
        record(p,entry['raw_sha256'],'Day04 full ledger');require(sha(p)==used[p],'Day01/Day04 raw mismatch')
        d=torch.load(p,weights_only=True,map_location='cpu');r=d['records'][0];s=d['stages'][0]['score_filtered'];cat=d['catalogues'][0]
        require(r['prefix']==f'full_val_stage_run01/IR/{stem}' and d['signature']==entry['signature'],'raw identity/signature')
        boxes,scores,levels=r['nms_boxes'],r['nms_scores'],r['nms_levels'];flat=s['ids'].tolist();N=len(flat)
        require(N==expected_n and tuple(r['layer_keys'])==KEYS and r['config']['nms_thresh']==.7 and r['config']['post_nms']==150,'original config/input size')
        require(tensor_equal(boxes,s['boxes']) and tensor_equal(scores,s['scores']),'complete score_filtered != original NMS input')
        fl=torch.repeat_interleave(torch.arange(3),torch.tensor(r['counts']))
        require(torch.equal(fl[s['ids']],levels),'actual layers from full index')
        ids=[cat[f]['candidate_id'] for f in flat]
        require(len(set(ids))==N and all(cat[f]['flat_index']==f and cat[f]['level_key']==KEYS[int(levels[i])] and ids[i]==r['prefix']+'/'+str(f) for i,f in enumerate(flat)),'candidate identities')
        # Independent replay is complete BEFORE consulting original nms_indices.
        COUNTS['main_replay_calls']+=1
        replay=greedy_replay(boxes,scores,levels,r['config']['nms_thresh'])
        generated=replay['keep']
        original=r['nms_indices'].tolist();COUNTS['full_output_verifications']+=1
        index_match=generated==original
        st=d['stages'][0];gen=torch.tensor(generated,dtype=torch.int64)
        fields_match=all(tensor_equal(st['layer_nms'][k],s[k][gen]) for k in ['ids','boxes','scores'])
        final_match=all(tensor_equal(st['post_nms_topn'][k],s[k][gen[:150]]) for k in ['ids','boxes','scores']) and tensor_equal(r['final_boxes'],boxes[gen[:150]]) and tensor_equal(r['final_scores'],scores[gen[:150]])
        matched=index_match and fields_match and final_match
        origpos={v:i for i,v in enumerate(original)};rank={v:i for i,v in enumerate(replay['order'])}
        for i in range(max(len(original),len(generated))):
            a=original[i] if i<len(original) else None;b=generated[i] if i<len(generated) else None
            checks.append(dict(stem=stem,output_position=i,original_position=a,replay_position=b,original_id=ids[a] if a is not None else '',replay_id=ids[b] if b is not None else '',original_box=boxes[a].tolist() if a is not None else [],replay_box=boxes[b].tolist() if b is not None else [],original_score=float(scores[a]) if a is not None else '',replay_score=float(scores[b]) if b is not None else '',identity_box_score_equal=a==b,within_final_150=i<150))
        verifications.append(dict(stem=stem,input_count=N,original_output_count=len(original),replay_output_count=len(generated),full_sequence_equal=index_match,layer_nms_fields_equal=fields_match,final150_equal=final_match,status='passed' if matched else 'failed'))
        index=next(i for i,x in enumerate(ds.records) if x.stem==stem);rec=ds.get_record(index)
        for pp in [rec.infrared_path,rec.visible_path,rec.annotation_path]:record(pp,expected[pp.relative_to(ROOT).as_posix()],'Day04 original image/XML')
        images,gt=ds[index];require(tensor_equal(gt['boxes'],d['observed']['transform_gt_all']) and tuple(r['image_size'])==(640,640),'Letterbox GT mismatch')
        bb=boxes.tolist();ss=scores.tolist();ll=levels.tolist();gtbox=gt['boxes'][0].tolist();gi=[iou(b,gtbox) for b in bb]
        def desc(i):return dict(candidate_id=ids[i],flat_index=flat[i],input_position=i,layer=KEYS[ll[i]],score=ss[i],xyxy=bb[i],gt_iou=gi[i],original_nms_kept=i in origpos,original_output_position=origpos.get(i,''),final150_kept=i in origpos and origpos[i]<150)
        for i in range(N):all_inputs.append(dict(stem=stem,**desc(i),sorted_position=rank[i],offset=replay['offsets'][i],shifted_xyxy=replay['shifted'][i]))
        for step in replay['steps']:
            a=step['selected_position'];deleted=step['new_deleted_positions']
            steps.append(dict(stem=stem,step=step['step'],selected_position=a,selected_id=ids[a],selected_flat=flat[a],selected_score=ss[a],new_deleted_positions=deleted,new_deleted_ids=[ids[b] for b in deleted]))
            events.append(dict(stem=stem,step=step['step'],action='keep',candidate_id=ids[a],input_position=a,flat_index=flat[a],selected_id=ids[a],selected_position=a,path_iou=''))
            for b in deleted:events.append(dict(stem=stem,step=step['step'],action='delete',candidate_id=ids[b],input_position=b,flat_index=flat[b],selected_id=ids[a],selected_position=a,path_iou=iou(replay['shifted'][a],replay['shifted'][b])))
        g=flat.index(goodflat);t=flat.index(targetflat)
        for c in replay['comparisons']:
            a,b=c['selected_position'],c['other_position']
            c.update(stem=stem,selected_id=ids[a],other_id=ids[b],same_layer=ll[a]==ll[b],raw_iou=iou(bb[a],bb[b]))
            if b==g:survival.append(dict(c))
        # Exhaustive all-input audit, not only pairs visited by greedy processing.
        local_sensitive=[];local_ties=[]
        for a in range(N):
            for b in range(a+1,N):
                raw32=iou(bb[a],bb[b]);raw64=iou(bb[a],bb[b],False);path32=iou(replay['shifted'][a],replay['shifted'][b]);path64=iou(replay['shifted'][a],replay['shifted'][b],False)
                near=any(abs(v-.7)<=1e-6 for v in [raw32,raw64,path32,path64]);diff=((ll[a]==ll[b] and raw32>q(.7))!=(path32>q(.7))) or (path32>q(.7))!=(path64>.7) or (ll[a]==ll[b] and (raw32>q(.7))!=(raw64>.7))
                z=dict(stem=stem,a_id=ids[a],b_id=ids[b],a_position=a,b_position=b,same_layer=ll[a]==ll[b],a_score=ss[a],b_score=ss[b],equal_score=ss[a]==ss[b],raw_iou_float32=raw32,raw_iou_float64=raw64,path_iou_float32=path32,path_iou_float64=path64,near_threshold=near,predicate_difference=diff)
                pairs.append(z)
                if near or diff:sensitive.append(z);local_sensitive.append(z)
                if ss[a]==ss[b]:ties.append(z);local_ties.append(z)
        # All ultimately retained higher-score candidates satisfying deletion geometry for t.
        possible=[a for a in original if ll[a]==ll[t] and ss[a]>ss[t] and (iou(bb[a],bb[t])>.7 or iou(replay['shifted'][a],replay['shifted'][t])>q(.7))]
        first=replay['deleted_by'].get(t);firsta=first[1] if first else None
        keypos=list(dict.fromkeys([g,t]+possible+([firsta] if firsta is not None else [])))
        for a in keypos:
            for b in keypos:
                if a>=b:continue
                relations.append(dict(stem=stem,a_id=ids[a],b_id=ids[b],a_flat=flat[a],b_flat=flat[b],same_layer=ll[a]==ll[b],a_score=ss[a],b_score=ss[b],a_gt_iou=gi[a],b_gt_iou=gi[b],a_original_kept=a in origpos,b_original_kept=b in origpos,raw_iou=iou(bb[a],bb[b]),path_iou=iou(replay['shifted'][a],replay['shifted'][b]),replay_first_deletion_direction=f'{ids[a]} -> {ids[b]}' if replay['deleted_by'].get(b,(None,None))[1]==a else f'{ids[b]} -> {ids[a]}' if replay['deleted_by'].get(a,(None,None))[1]==b else '',attribution_validated=matched))
        tie_ambiguity=any(z['same_layer'] and z['path_iou_float32']>q(.7) for z in local_ties)
        sensitive_decision=bool(local_sensitive)
        summary=dict(stem=stem,full_output_verified=matched,good=desc(g),deleted_competitor=desc(t),first_deletion_step=first[0] if first else None,replay_first_deleter=desc(firsta) if firsta is not None else None,possible_upstream=[dict(**desc(a),raw_iou_to_target=iou(bb[a],bb[t]),path_iou_to_target=iou(replay['shifted'][a],replay['shifted'][t]),raw_iou_to_good=iou(bb[a],bb[g]),path_iou_to_good=iou(replay['shifted'][a],replay['shifted'][g])) for a in possible],good_replay_kept=g in generated,good_deletion_comparisons=[c for c in replay['comparisons'] if c['other_position']==g],tie_ambiguity=tie_ambiguity,precision_sensitive_pairs=len(local_sensitive),attribution_unambiguous_within_replay=matched and not tie_ambiguity and not sensitive_decision)
        summaries.append(summary)
        zoom=make_figures(out,stem,images['infrared'],gtbox,[(desc(a),'存活好框' if a==g else '被删竞争者' if a==t else '上游保留框') for a in keypos],summary)
        geometry.append(dict(stem=stem,parameters=str(ds.transforms.parameters(1024,1280)),GT_exact=True,zoom=zoom,display_only=True))
        print('REPLAY_VERIFIED',stem,verifications[-1], 'FIRST_DELETER',summary['replay_first_deleter'],flush=True)
        # All actual comparisons exported; no second replay used for validation.
        csvout(out/f'{stem}_replay_pair_checks.csv',replay['comparisons'],list(replay['comparisons'][0]))
    for n,rr in [('all_inputs',all_inputs),('replay_steps',steps),('replay_events',events),('output_comparison',checks),('case_verification',verifications),('case_relations',relations),('good_survival_checks',survival),('all_pair_numeric_audit',pairs),('score_ties',ties),('threshold_precision_audit',sensitive)]:
        default=list(pairs[0]) if n in ['score_ties','threshold_precision_audit'] else ['note']
        csvout(out/(n+'.csv'),rr,list(rr[0]) if rr else default)
    save(out/'case_details.json',summaries);save(out/'figure_geometry.json',geometry);save(out/'input_provenance.json',provenance)
    report(out,summaries,verifications,local_env=evidence['current_environment'],oldenv=oldenv,ties=ties,sensitive=sensitive)
    return dict(status='passed' if all(v['status']=='passed' for v in verifications) else 'failed',cases=verifications,**COUNTS,input_candidates=len(all_inputs),full_input_pair_audit_rows=len(pairs),score_tie_pairs=len(ties),threshold_precision_pairs=len(sensitive),replay_event_rows=len(events),attribution_all_unambiguous=all(s['attribution_unambiguous_within_replay'] for s in summaries),new_model_forwards=0,checkpoint_loads=0,server_connections=0,training_runs=0,test_runs=0,fusion_replays=0,counterfactual_replays=0,existing_nms_operator_calls=0)

def make_figures(out,stem,tensor,gt,items,summary):
    fontpath='C:/Windows/Fonts/msyh.ttc'
    def text(d,xy,s,size=21,color='#26354a'):d.text(xy,s,font=ImageFont.truetype(fontpath,size),fill=color)
    palette={'存活好框':'#00b7a8','被删竞争者':'#ee6384','上游保留框':'#6874d9'}
    image=Image.fromarray(tensor.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy())
    bb=[gt]+[x['xyxy'] for x,_ in items];bounds=(max(0,math.floor(min(b[0] for b in bb))-22),max(0,math.floor(min(b[1] for b in bb))-22),min(640,math.ceil(max(b[2] for b in bb))+22),min(640,math.ceil(max(b[3] for b in bb))+22))
    cols=min(3,len(items));cw=510;ch=590;im=Image.new('RGB',(cols*cw,120+math.ceil(len(items)/cols)*ch),'#f2f5fa');draw=ImageDraw.Draw(im)
    text(draw,(20,15),f'{stem}/G0/IR 关键候选（每框单独展示，完整ID见CSV）',27)
    text(draw,(20,60),'黄色GT；青色好框；粉红被删竞争者；紫色上游保留框。',21)
    for j,(a,role) in enumerate(items):
        x=(j%cols)*cw;y=115+(j//cols)*ch
        text(draw,(x+15,y),f'{role} flat={a["flat_index"]} 层{a["layer"]}',23,palette[role])
        w,h=bounds[2]-bounds[0],bounds[3]-bounds[1];scale=min(470/w,350/h);rw,rh=round(w*scale),round(h*scale);ox=x+(cw-rw)//2;oy=y+45
        im.paste(image.crop(bounds).resize((rw,rh),Image.Resampling.BILINEAR),(ox,oy))
        def tr(b):return [ox+(b[0]-bounds[0])*rw/w,oy+(b[1]-bounds[1])*rh/h,ox+(b[2]-bounds[0])*rw/w,oy+(b[3]-bounds[1])*rh/h]
        draw.rectangle(tr(gt),outline='#ffd24a',width=3);draw.rectangle(tr(a['xyxy']),outline=palette[role],width=3)
        text(draw,(x+15,y+407),f'score={a["score"]:.9f}  GT IoU={a["gt_iou"]:.9f}',19)
        text(draw,(x+15,y+442),f'原输入#{a["input_position"]}；原NMS {"保留" if a["original_nms_kept"] else "删除"}',20)
        text(draw,(x+15,y+477),f'原输出位置 {a["original_output_position"]}；最终150 {a["final150_kept"]}',20)
    im.save(out/f'{stem}_key_candidates.png')
    upstream=summary['possible_upstream'];height=420+len(upstream)*145
    im=Image.new('RGB',(1500,height),'#f2f5fa');draw=ImageDraw.Draw(im)
    text(draw,(25,15),f'{stem}：与原完整输出{"一致" if summary["full_output_verified"] else "不一致"}的离线过程',30)
    text(draw,(25,65),'实线箭头＝该重放首次删除；虚线文字＝几何关系，不是GPU事件记录。',23)
    g=summary['good'];t=summary['deleted_competitor'];first=summary['replay_first_deleter']
    text(draw,(25,115),f'重放step={summary["first_deletion_step"]}：保留{first["flat_index"] if first else "未知"}，删除竞争者{t["flat_index"]}；之后好框{g["flat_index"]}保留。',24)
    for j,a in enumerate(upstream):
        y=180+j*145
        draw.rectangle((25,y,440,y+100),outline=palette['上游保留框'],width=3)
        text(draw,(40,y+8),f'上游flat={a["flat_index"]} s={a["score"]:.7f}',22)
        text(draw,(40,y+47),f'层{a["layer"]} GT IoU={a["gt_iou"]:.7f}',21)
        if first and a['candidate_id']==first['candidate_id']:
            draw.line((450,y+50,760,y+50),fill='#ee6384',width=4);draw.polygon([(760,y+50),(742,y+42),(742,y+58)],fill='#ee6384')
        else:
            for x in range(450,760,20):draw.line((x,y+50,x+10,y+50),fill='#8392a5',width=2)
        text(draw,(465,y+4),f'原IoU {a["raw_iou_to_target"]:.7f}',19)
        text(draw,(465,y+70),f'路径IoU {a["path_iou_to_target"]:.7f}',19)
        text(draw,(780,y+10),f'竞争者{t["flat_index"]} 原NMS删除',22)
        text(draw,(780,y+47),f'上游↔好框：原IoU={a["raw_iou_to_good"]:.7f} / 路径={a["path_iou_to_good"]:.7f}',20)
    y=200+len(upstream)*145
    text(draw,(25,y),f'好框{g["flat_index"]}：完整过程中，没有任何已保留框对它通过严格>0.7的路径判定。',23)
    text(draw,(25,y+50),'候选竞争者已被删除，不再获得保留资格；完整比较过程见good_survival_checks.csv。',22)
    text(draw,(25,y+100),'证据边界：原索引证明去留；箭头来自独立重放，不是原GPU逐次抑制事件。',22)
    im.save(out/f'{stem}_process.png');return bounds

def report(out,summaries,verifications,local_env,oldenv,ties,sensitive):
    lines=['# Week10 Day02：两个IR案例的上游关系与离线重放','',
    '可以把NMS理解为：先从剩余候选中挑最高分框保留，再删除与它高度重叠的候选；被删除的框以后不会再轮到它删别人。Day01发现的两个直接竞争者虽然分数高于好框，但它们本身是否先被处理掉，T并没有表达。本轮用完整原输入重现这个过程，再与原全部输出核对。','',
    '下面的“首次删除者”只属于本次独立离线重放。原捕获只保存了完整输入和输出，没有保存GPU逐次抑制事件。完整输出匹配后才能称为“与原完整输出一致的离线重放解释”。','',
    '## 完整输出核验','','|图片|全部输入|原输出|重放输出|完整序列|layer_nms字段|最终150|','|---|---:|---:|---:|---|---|---|']
    for v in verifications:lines.append(f'|{v["stem"]}|{v["input_count"]}|{v["original_output_count"]}|{v["replay_output_count"]}|{v["full_sequence_equal"]}|{v["layer_nms_fields_equal"]}|{v["final150_equal"]}|')
    for s in summaries:
        g=s['good'];t=s['deleted_competitor'];a=s['replay_first_deleter']
        lines+=['',f'## {s["stem"]}/G0/IR','',f'完整输出核验：{s["full_output_verified"]}。若为False，以下仅为未通过核验的重放观察，不可当作已核验解释。',
        f'在本次重放的第{s["first_deletion_step"]}步（步号从0开始），候选`{a["candidate_id"] if a else "未知"}`第一次删除`{t["candidate_id"]}`。好框`{g["candidate_id"]}`随后保留。',
        '', '|角色|完整候选ID|分数|实际层|GT IoU|原NMS去留|原输出位置|','|---|---|---:|---|---:|---|---:|']
        for role,x in [('存活好框',g),('被删竞争者',t)]+[('上游保留框',x) for x in s['possible_upstream']]:lines.append(f'|{role}|{x["candidate_id"]}|{x["score"]:.9f}|{x["layer"]}|{x["gt_iou"]:.9f}|{"保留" if x["original_nms_kept"] else "删除"}|{x["original_output_position"]}|')
        lines+=['','|上游flat|与竞争者原IoU|与竞争者路径IoU|与好框原IoU|与好框路径IoU|重放首次删除者|','|---|---:|---:|---:|---:|---|']
        for x in s['possible_upstream']:lines.append(f'|{x["flat_index"]}|{x["raw_iou_to_target"]:.9f}|{x["path_iou_to_target"]:.9f}|{x["raw_iou_to_good"]:.9f}|{x["path_iou_to_good"]:.9f}|{a is not None and x["candidate_id"]==a["candidate_id"]}|')
        cc=s['good_deletion_comparisons'];maximum=max([x['path_iou'] for x in cc],default=0)
        lines+=['',f'好框被选中前，重放实际让{len(cc)}个已保留候选与它比较，其中最大的路径IoU为{maximum:.9f}；没有任何一次通过严格>0.7的删除判定。完整比较含其他层，逐行见good_survival_checks.csv。被删竞争者不再留在活动队列中，不能在后续步骤删除好框。',
        f'全部最终保留且同层高分、满足原坐标或路径删除几何条件的上游候选共有{len(s["possible_upstream"])}个，已全部列出；不限制其GT IoU。列表描述可能几何关系，只有标为首次删除者的一项是该重放的首次归因。',
        f'本重放归因无歧义标记：{s["attribution_unambiguous_within_replay"]}；同分重叠歧义：{s["tie_ambiguity"]}；阈值/精度敏感候选对：{s["precision_sensitive_pairs"]}。',
        '',f'![关键候选]({s["stem"]}_key_candidates.png)',f'![简化过程]({s["stem"]}_process.png)']
    lines+=['','## 原实现依据、数值审计与限制','',
    'Day04保存的agnostic_capture.py记录原batched_nms调用的boxes、scores、levels和完整返回索引。Day03保存的nms_source_readonly.txt说明小输入走坐标偏移路径：max_coordinate+1乘层索引再加到每个坐标上，然后统一NMS。65/80个候选的boxes.numel为260/320，均低于保存的CPU/GPU分支阈值。',
    '该Python源码快照来自Day03本地CPU环境，而不是原GPU二进制源码。原捕获环境与当前环境分别记录如下；不能据当前安装版本自动推断原内核。没有原编译CUDA内核源码/编译选项，不能声称逐步算术必然相同。主重放忠实实现保存Python路径的float32偏移及标准贪心语义，再用完整原索引序列验收。',
    '原环境：'+json.dumps(oldenv,ensure_ascii=False),'当前环境：'+json.dumps(local_env,ensure_ascii=False),
    '排序在看原输出前已固定：原分数降序，同分按输入位置；未挑选能匹配的排序。主函数仅接收框、分数、层和阈值，不读取GT、标签或原保留索引。重放完成后才读取原索引做序列核验。不调用现成nms/batched_nms。',
    f'完整输入所有候选对都审计同分、原坐标float32/float64、偏移坐标float32/float64、阈值附近及判定分歧。同分对{len(ties)}项，阈值或精度敏感对{len(sensitive)}项。1e-6只标记，主删除判定仍严格>float32(0.7)。完整表保留全部审计值，不用容差强求匹配。',
    '原操作事实：捕获的候选身份、分数、坐标、层、原全部输出及最终150结果。重放归因：独立贪心过程中第一次删除者、删除步号、活动队列的变化。仍不确定：原GPU内部实际抑制事件/时序是否逐项等同这次重放，以及候选分数和坐标为何形成。即使全部输出相同，也不提升为直接捕获的GPU事件。',
    'T只统计对指定GT不达标的竞争者，本轮上游关系没有这个限制；好框之间的关系仍在完整重放中参与，未被排除。Fusion仅引用Day01两例失败的背景，无Fusion重放。没有删除某候选后的反事实实验，没有新增研究指标，也没有RoI或最终检测分析。',
    '图片复用原PairedLetterbox，完整XML GT精确匹配原640×640捕获；局部放大仅用于展示。未根据图像亮区或车灯判断原因。',
    '本次主重放次数='+str(COUNTS['main_replay_calls'])+'，完整输出核验次数='+str(COUNTS['full_output_verifications'])+'，不是Day01的重放0次。历史文件前后SHA及来源见history_hashes.json、input_provenance.json；最终状态、实际命令、日志、环境、进程退出码见verification.json、run_metadata.json、run.log和同级*_process.json。',
    '如果完整序列未匹配，case_verification.csv与output_comparison.csv保留所有差异；应补充原设备排序/内核数值证据，而不是改排序或放宽阈值。若核验通过，也只得到固定两例的离线过程解释，不能推广总体比例或融合根因。本步骤完成后停止。']
    (out/'case_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
    require(out.is_relative_to(HERE) and out!=HERE,'output outside day02');out.mkdir(parents=True,exist_ok=False)
    folders=[ROOT/'week08',ROOT/'week09',ROOT/'week10/day01']
    extra=[ROOT/'AGENTS.md',ROOT/'data/LLVIP_1000/manifest.csv']
    for stem,_,_,_ in CASES:extra += [ROOT/f'data/LLVIP_1000/images/val/{m}/{stem}.jpg' for m in ['visible','infrared']]+[ROOT/f'data/LLVIP_1000/annotations_voc/val/{stem}.xml']
    def snapshot():return {str(p):sha(p) for p in sorted(set(extra+[p for folder in folders for p in folder.rglob('*') if p.is_file()]))}
    before=snapshot();save(out/'history_before.json',before);code=1;result={};started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    class Tee:
        def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
        def flush(self):log.flush()
    with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
        try:
            torch.set_num_threads(2)
            with ExitStack() as guard:
                guard.enter_context(patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Forward forbidden')))
                for k in ['nms','batched_nms']:guard.enter_context(patch.object(ops,k,side_effect=RuntimeError('Existing NMS operator forbidden')))
                result=run(out)
            code=0 if result['status']=='passed' else 1
        except BaseException:
            traceback.print_exc();save(out/'failure.json',dict(traceback=traceback.format_exc(),counts=COUNTS))
        finally:
            after=snapshot();unchanged=before==after;save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
            if not unchanged:code=1
            result.update(status='passed' if code==0 else 'failed',**COUNTS,historical_files_unchanged=unchanged,protected_files=len(before),exit_code=code)
            save(out/'verification.json',result)
            save(out/'run_metadata.json',dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,source_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'README.md',HERE/'run_replay.ps1']},current_environment=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform()),counts=COUNTS))
            print(json.dumps(result,ensure_ascii=False),flush=True);print('EXIT',code,flush=True)
    save(out/'artifact_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return code
if __name__=='__main__':sys.exit(main())
