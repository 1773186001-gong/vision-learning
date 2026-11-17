import sys
sys.dont_write_bytecode=True
import argparse,csv,json,hashlib,datetime,subprocess,traceback,platform,math
from pathlib import Path
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from unittest.mock import patch
import torch,torchvision
from torchvision.ops import box_iou
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
D4=ROOT/'week09/day04/outputs/full_val_stage_run01'
STEMS=('030078','030129')
WEIGHTS={'IR':('week08/day03/outputs/ir_baseline_run01/best.pth','9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2'),'Fusion':('week08/day05/outputs/fusion_concat_run01/best.pth','9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a')}
COUNTS=dict(checkpoint_loads=0,strict_state_loads=0,structure_backbone_probes=0,validation_full_forwards=0)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,d):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def csvout(p,rr):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]) if rr else ['empty']);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def require(c,s):
 if not c:raise ValueError(s)
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cpu(v) for v in x]
 return x
def digest(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def config(d):
 return dict(rpn=dict(pre=d.rpn.pre_nms_top_n(),post=d.rpn.post_nms_top_n(),score=d.rpn.score_thresh,nms=d.rpn.nms_thresh,min_size=d.rpn.min_size,anchors=d.rpn.anchor_generator.sizes,ratios=d.rpn.anchor_generator.aspect_ratios),roi=dict(score=d.roi_heads.score_thresh,nms=d.roi_heads.nms_thresh,detections=d.roi_heads.detections_per_img,featmaps=d.roi_heads.box_roi_pool.featmap_names,bbox_weights=d.roi_heads.box_coder.weights),transform=dict(min=d.transform.min_size,max=d.transform.max_size,mean=d.transform.image_mean,std=d.transform.image_std,size_divisible=d.transform.size_divisible))
def best(boxes,g):
 if not len(boxes):return None,None
 v=box_iou(boxes,g[None])[:,0];i=int(v.argmax());return float(v[i]),i
def run(out):
 for n in ['week08/day03','week08/day05','week09/day04']:sys.path.insert(0,str(ROOT/n))
 import ir_common as ir
 import fusion_model as fusion
 from agnostic_capture import AgnosticCapture
 expected=load(D4/'preflight/expected_inputs.json')['sha256'];prov=[]
 for n,h in expected.items():
  if n.startswith('week') or n=='data/LLVIP_1000/manifest.csv' or any('/'+s+'.' in n for s in STEMS):
   p=ROOT/n;require(sha(p)==h,'Source/input SHA '+n);prov.append(dict(path=str(p),sha256=h))
 for day,folder in [('day01','candidate_review_run01'),('day02','upstream_replay_run01')]:
  base=ROOT/'week10'/day/'outputs'/folder
  for n in ['case_report_zh.md','verification.json','input_provenance.json']:
   content=(base/n).read_text(encoding='utf-8-sig');prov.append(dict(path=str(base/n),sha256=sha(base/n)))
  require(load(base/'verification.json')['status']=='passed','predecessor failed')
 ledger={(r['model'],r['stem']):r for r in read(D4/'forward_ledger.csv')}
 save(out/'input_provenance.json',prov)
 ds=ir.LLVIPDataset(ROOT/'data/LLVIP_1000',split='val',modality='paired',infrared_channels=3,transforms=ir.PairedLetterboxAndFlip('val'))
 samples={}
 for stem in STEMS:
  i=next(i for i,r in enumerate(ds.records) if r.stem==stem);samples[stem]=ds[i]
 summaries=[];rpns=[];finals=[];diffs=[];modelchecks=[];pictures={};probe_records=[]
 for name,(rel,h) in WEIGHTS.items():
  path=ROOT/rel;require(sha(path)==h,'checkpoint SHA');beforeload=COUNTS.copy()
  original_load=torch.load;original_state=torch.nn.Module.load_state_dict
  def ckload(p,*a,**kw):
   require(Path(p).resolve()==path.resolve(),'Unexpected load during restore');COUNTS['checkpoint_loads']+=1;return original_load(p,*a,**kw)
  def stateload(m,state,*a,**kw):
   require(kw.get('strict') is True,'strict restore required');COUNTS['strict_state_loads']+=1;return original_state(m,state,*a,**kw)
  def probe(m,inputs):
   if type(m).__name__=='BackboneWithFPN':
    x=inputs[0];require(x.shape==(1,3,640,640) and not bool(x.any()),'Unexpected restoration probe');COUNTS['structure_backbone_probes']+=1;probe_records.append(dict(model=name,type='restore zero-tensor BackboneWithFPN',shape=list(x.shape)))
  with patch.object(torch,'load',ckload),patch.object(torch.nn.Module,'load_state_dict',stateload),patch.object(torch.hub,'download_url_to_file',side_effect=RuntimeError('Download forbidden')):
   handle=torch.nn.modules.module.register_module_forward_pre_hook(probe)
   try:model,ck=(ir.checkpoint_model if name=='IR' else fusion.restore)(path,'cpu')
   finally:handle.remove()
  del ck;model.eval();det=model if name=='IR' else model.detector;cfg=config(det);initial=digest(model)
  require(cfg['roi']['score']==.001 and cfg['roi']['detections']==100 and cfg['rpn']['pre']==cfg['rpn']['post']==150,'Inference configuration')
  for stem in STEMS:
   images,target=samples[stem];require(bool((target['labels']==1).all()),'person label must be 1');g=target['boxes'][0]
   e=ledger[name,stem];oldpath=D4/e['relative_raw'];require(sha(oldpath)==e['raw_sha256'],'historical raw SHA')
   old=original_load(oldpath,map_location='cpu',weights_only=True);hist=old['records'][0]
   require(torch.equal(target['boxes'],old['observed']['transform_gt_all']),'Historical GT geometry')
   state0=digest(model);require(state0==initial,'state before forward')
   hooks0=[(m,dict(m._forward_hooks),dict(m._forward_pre_hooks)) for m in model.modules()]
   method0={k:(k in det.rpn.__dict__,det.rpn.__dict__.get(k)) for k in ['filter_proposals','_get_top_n_idx']}
   obs={};cap=None;identity=f'{out.name}/{name}/{stem}'
   with ExitStack() as stack:
    def pre(m,inputs):
     nonlocal cap
     im,features,targets=inputs;require(cap is None and tuple(features)==('0','1','pool'),'RPN call/features');require(im.image_sizes==[(640,640)],'internal transform size')
     obs['internal_image_sizes']=im.image_sizes;cap=stack.enter_context(AgnosticCapture(m,tuple(features),target['boxes'],identity))
    def transform(m,inputs,output):obs['transform_sizes']=output[0].image_sizes
    def rpnpost(m,inputs,output):obs['actual_rpn_return']=cpu(output[0][0])
    for m,kind,fn in [(det.rpn,'pre',pre),(det.transform,'post',transform),(det.rpn,'post',rpnpost)]:
     handle=m.register_forward_pre_hook(fn) if kind=='pre' else m.register_forward_hook(fn);stack.callback(handle.remove)
    COUNTS['validation_full_forwards']+=1;require(COUNTS['validation_full_forwards']<=4,'Forward budget exceeded')
    with torch.inference_mode():prediction=model([images['infrared'] if name=='IR' else images])[0]
   restored=all(dict(m._forward_hooks)==a and dict(m._forward_pre_hooks)==b for m,a,b in hooks0) and all((k in det.rpn.__dict__,det.rpn.__dict__.get(k))==v for k,v in method0.items())
   require(cap is not None and len(cap.records)==1 and restored,'capture/hook restore')
   r=cpu(cap.records[0]);st=cpu(cap.states[0]);pred=cpu(prediction);obs=cpu(obs)
   require(torch.equal(obs['actual_rpn_return'],r['final_boxes']),'actual RPN endpoint')
   raw=dict(run_identity=identity,model=name,stem=stem,gt_index=0,gt_all=target['boxes'],labels=target['labels'],records=[r],stages=[st],catalogues=cap.catalogues,observed=obs,model_output=pred,configuration=cfg,coordinates=dict(original_wh=[1280,1024],letterbox_wh=[640,640],scale=.5,padding_ltrb=[0,64,0,64],internal_wh=[640,640],final_output_wh=[640,640],final_output='model returns coordinates of supplied Letterbox input; no unpadding'))
   with (out/f'{name}_{stem}_raw_capture.pth').open('xb') as f:torch.save(raw,f)
   state1=digest(model);require(initial==state0==state1 and config(det)==cfg,'persistent state/config changed')
   ids=st['post_nms_topn']['ids'].tolist();oldids=old['stages'][0]['post_nms_topn']['ids'].tolist();rb=r['final_boxes'];rs=r['final_scores'];hv,_=best(hist['final_boxes'],g);rv,ri=best(rb,g)
   correct=torch.where(pred['labels']==1)[0];fv,fi0=best(pred['boxes'][correct],g);fi=int(correct[fi0]) if fi0 is not None else None;top=int(correct[pred['scores'][correct].argmax()]) if len(correct) else None
   for j,f in enumerate(ids):rpns.append(dict(model=name,stem=stem,run_identity=identity,output_index=j,flat_index=f,candidate_id=cap.catalogues[0][f]['candidate_id'],layer=cap.catalogues[0][f]['level_key'],xyxy=rb[j].tolist(),objectness=float(rs[j]),gt_iou=float(box_iou(rb[j:j+1],g[None])[0,0])))
   for j in range(len(pred['boxes'])):finals.append(dict(model=name,stem=stem,run_identity=identity,output_index=j,label=int(pred['labels'][j]),xyxy=pred['boxes'][j].tolist(),classification_score=float(pred['scores'][j]),gt_iou=float(box_iou(pred['boxes'][j:j+1],g[None])[0,0])))
   for j in range(max(len(ids),len(oldids))):
    exists=j<len(ids) and j<len(oldids)
    diffs.append(dict(model=name,stem=stem,index=j,historical_flat=oldids[j] if j<len(oldids) else None,new_flat=ids[j] if j<len(ids) else None,identity_equal=exists and ids[j]==oldids[j],coordinate_delta=(rb[j]-hist['final_boxes'][j]).tolist() if exists else None,score_delta=float(rs[j]-hist['final_scores'][j]) if exists else None))
   row=dict(model=name,stem=stem,run_identity=identity,historical_rpn_max_iou=hv,rpn_max_iou=rv,rpn_best_index=ri,rpn_best_id=cap.catalogues[0][ids[ri]]['candidate_id'] if ri is not None else None,rpn_best_objectness=float(rs[ri]) if ri is not None else None,final_correct_max_iou=fv,final_best_iou_index=fi,final_best_iou_score=float(pred['scores'][fi]) if fi is not None else None,final_top_score_index=top,final_top_score=float(pred['scores'][top]) if top is not None else None,final_top_score_gt_iou=float(box_iou(pred['boxes'][top:top+1],g[None])[0,0]) if top is not None else None,rpn_covered=rv is not None and rv>=.75,final_covered=fv is not None and fv>=.75,historical_identity_order_equal=ids==oldids,historical_coverage_equal=(hv>=.75)==(rv is not None and rv>=.75),rpn_count=len(ids),final_count=len(pred['boxes']))
   summaries.append(row);pictures[name,stem]=(raw,row,images['infrared']);modelchecks.append(dict(model=name,stem=stem,state_before=state0,state_after=state1,state_equal=True,config=cfg,hooks_restored=restored,checkpoint_sha_after=sha(path),historical_raw_sha=sha(oldpath)));print('CAPTURED',row,flush=True)
  del model
 csvout(out/'summary.csv',summaries);csvout(out/'rpn_candidates.csv',rpns);csvout(out/'final_detections.csv',finals);csvout(out/'historical_differences.csv',diffs);save(out/'model_checks.json',modelchecks);save(out/'structure_probes.json',probe_records)
 for stem in STEMS:
  im=Image.new('RGB',(1600,900),'#f2f5fa');d=ImageDraw.Draw(im)
  def text(x,y,t,size=20):d.text((x,y),t,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size),fill='#26354a')
  text(20,15,f'{stem}/G0 同次RPN与最终检测：共享IR底图；Fusion实际输入RGB+IR',25)
  items=[]
  for name in ['IR','Fusion']:
   raw,row,t=pictures[name,stem];items.extend([raw['gt_all'][0].tolist()]);
   for b,i in [(raw['records'][0]['final_boxes'],row['rpn_best_index']),(raw['model_output']['boxes'],row['final_best_iou_index'])]:
    if i is not None:items.append(b[i].tolist())
  bounds=(max(0,math.floor(min(b[0] for b in items))-25),max(0,math.floor(min(b[1] for b in items))-25),min(640,math.ceil(max(b[2] for b in items))+25),min(640,math.ceil(max(b[3] for b in items))+25))
  for col,name in enumerate(['IR','Fusion']):
   raw,row,t=pictures[name,stem];x=20+col*800;text(x,65,f'{name}：黄色GT / 青色RPN最大IoU / 粉红最终最大IoU',21)
   img=Image.fromarray(t.mul(255).round().byte().permute(1,2,0).numpy());w,h=bounds[2]-bounds[0],bounds[3]-bounds[1];scale=min(740/w,570/h);rw,rh=round(w*scale),round(h*scale);ox=x+(760-rw)//2;oy=110;im.paste(img.crop(bounds).resize((rw,rh)),(ox,oy))
   for b,color in [(raw['gt_all'][0].tolist(),'#ffd24a')]+([(raw['records'][0]['final_boxes'][row['rpn_best_index']].tolist(),'#00c4b0')] if row['rpn_best_index'] is not None else [])+([(raw['model_output']['boxes'][row['final_best_iou_index']].tolist(),'#ef6a92')] if row['final_best_iou_index'] is not None else []):
    d.rectangle((ox+(b[0]-bounds[0])*rw/w,oy+(b[1]-bounds[1])*rh/h,ox+(b[2]-bounds[0])*rw/w,oy+(b[3]-bounds[1])*rh/h),outline=color,width=3)
   text(x,710,f'RPN IoU={row["rpn_max_iou"]:.6f} objectness={row["rpn_best_objectness"]:.6f}')
   text(x,750,f'最终最大IoU={row["final_correct_max_iou"]:.6f} 分类分数={row["final_best_iou_score"]:.6f}')
   text(x,790,f'最高分类分数={row["final_top_score"]:.6f} 对G0 IoU={row["final_top_score_gt_iou"]:.6f}')
  text(20,850,'仅展示诊断选框；全部原始返回框见CSV。无内部身份映射，不把两个框连成回归链。');im.save(out/f'{stem}_comparison.png')
 lines=['# Week10 Day03：RPN与最终检测终点对照','','RPN提供候选，完整模型再返回带类别和分类分数的最终检测。这次只对同一次运行的两个终点检查：有没有对G0达到0.75的框。最大IoU只用于诊断选框，未参与模型筛选。','', '|图片|模型|历史RPN最大IoU|本次RPN最大IoU|最终正确类别最大IoU|该框分类分数|最高分类分数框的IoU|RPN/最终覆盖|','|---|---|---:|---:|---:|---:|---:|---|']
 for r in summaries:lines.append(f'|{r["stem"]}|{r["model"]}|{r["historical_rpn_max_iou"]}|{r["rpn_max_iou"]}|{r["final_correct_max_iou"]}|{r["final_best_iou_score"]}|{r["final_top_score_gt_iou"]}|{r["rpn_covered"]}/{r["final_covered"]}|')
 lines+=['','原始图片1280×1024，经Letterbox以0.5缩放，上下padding各64，输入640×640。内部transform实测640×640，输出坐标还原到传入的640×640 Letterbox输入，不是原始图片；无需再次缩放或去padding。完整GT与历史精确核对。person标签为1，模型输出类别正确集合按labels==1定义。','历史GPU与本次CPU分别记录；逐位置flat/顺序、坐标及分数差见historical_differences.csv。不设数值容差来强求复现。身份/顺序或覆盖不同的项目只能视为新CPU结果，不可拼接成历史NMS链的后续。','RPN objectness与最终分类分数含义不同。所有返回检测均保存，没有提高0.001阈值，也未因绘图隐藏而删除CSV中的低分结果。完整模型的RoI配置及RPN配置见model_checks.json。','端点最大IoU可来自不同候选，没有RoI内部身份追踪证据，不宣称某个RPN框回归成某个最终框，不能把现象归因为回归、分类阈值或最终NMS。这不是AP75，也不是正式一对一匹配TP。','下一步优先审阅出现RPN未覆盖而最终覆盖的项目；若没有这种组合，则优先检查最终最大IoU框与最高分框差异最大的固定项目。只作下一步建议，本轮不追加前向或阶段归因。','']
 for stem in STEMS:lines.append(f'![{stem}]({stem}_comparison.png)')
 (out/'case_report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
 return dict(status='passed',**COUNTS,items=4,all_outputs_saved=True,coordinate_verified=True,model_state_config_hooks_verified=True,server_connections=0,test_inference=0,training=0,historical_identity_order_equal=all(r['historical_identity_order_equal'] for r in summaries),historical_coverage_equal=all(r['historical_coverage_equal'] for r in summaries))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();require(out.is_relative_to(HERE),'output scope');out.mkdir(parents=True,exist_ok=False)
 dirs=[ROOT/'week08',ROOT/'week09',ROOT/'week10/day01',ROOT/'week10/day02']
 def snapshot():return {str(p):sha(p) for base in dirs for p in base.rglob('*') if p.is_file()}
 before=snapshot();save(out/'history_before.json',before);code=1;result={}
 class Tee:
  def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
  def flush(self):log.flush()
 with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
  try:torch.set_num_threads(2);torch.manual_seed(42);result=run(out);code=0
  except BaseException:traceback.print_exc();save(out/'failure.json',dict(error=traceback.format_exc(),counts=COUNTS))
  finally:
   after=snapshot();unchanged=before==after;save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
   if not unchanged:code=1
   result.update(status='passed' if code==0 else 'failed',counts=COUNTS,historical_files_unchanged=unchanged,exit_code=code);save(out/'verification.json',result)
   save(out/'run_metadata.json',dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),exit_code=code,environment=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform()),source_sha256=sha(Path(__file__)),counts=COUNTS));print(result,flush=True);print('EXIT',code,flush=True)
 save(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});return code
if __name__=='__main__':sys.exit(main())
