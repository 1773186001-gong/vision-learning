import sys
sys.dont_write_bytecode=True
import argparse,json,csv,hashlib,subprocess,traceback,inspect,platform,datetime
from pathlib import Path
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from unittest.mock import patch
import torch,torchvision
from torchvision.ops import box_iou
from torchvision.models.detection import roi_heads as roi_module
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
D3=ROOT/'week10/day03/outputs/rpn_final_comparison_run01'
COUNT=dict(checkpoint_loads=0,strict_loads=0,full_model_forwards=0,structure_probes=0,independent_nms_replays=0)
for n in ['week10/day03','week08/day03','week09/day04']:sys.path.insert(0,str(ROOT/n))
import compare_endpoints as helper
import ir_common as ir
from agnostic_capture import AgnosticCapture
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def csvout(p,rr):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]) if rr else ['empty']);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
def require(c,s):
 if not c:raise ValueError(s)
def run(out):
 cp=helper.cpu;source=Path(inspect.getfile(roi_module));src=source.read_text(encoding='utf-8');(out/'roi_heads_source.py.txt').write_text(src,encoding='utf-8')
 require('scores > self.score_thresh' in src and 'min_size=1e-2' in src,'Unexpected actual implementation')
 provenance=[dict(path=str(source),sha256=sha(source),role='actual installed RoI implementation')]
 for x in load(D3/'input_provenance.json'):
  require(sha(x['path'])==x['sha256'],'input/source SHA');provenance.append(x)
 manifest=load(D3/'artifact_manifest.json')
 for n in ['IR_030129_raw_capture.pth','model_checks.json','summary.csv','verification.json','case_report_zh.md']:
  require(sha(D3/n)==manifest[n],'Day03 modified '+n);provenance.append(dict(path=str(D3/n),sha256=sha(D3/n)))
 old=torch.load(D3/'IR_030129_raw_capture.pth',weights_only=True,map_location='cpu')
 checkpoint=ROOT/'week08/day03/outputs/ir_baseline_run01/best.pth';expected='9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2';require(sha(checkpoint)==expected,'checkpoint SHA')
 save(out/'provenance.json',provenance)
 torch.set_num_threads(2);torch.manual_seed(42)
 original_load=torch.load;original_state=torch.nn.Module.load_state_dict
 def restoreload(p,*a,**kw):
  require(Path(p).resolve()==checkpoint.resolve(),'unexpected restore load');COUNT['checkpoint_loads']+=1;return original_load(p,*a,**kw)
 def strict(m,s,*a,**kw):
  require(kw.get('strict') is True,'strict restore required');COUNT['strict_loads']+=1;return original_state(m,s,*a,**kw)
 with patch.object(torch,'load',restoreload),patch.object(torch.nn.Module,'load_state_dict',strict),patch.object(torch.hub,'download_url_to_file',side_effect=RuntimeError('download forbidden')):
  model,ck=ir.checkpoint_model(checkpoint,'cpu')
 del ck;model.eval();cfg=helper.config(model);require(json.loads(json.dumps(cfg))==json.loads(json.dumps(old['configuration'])),'Day03 configuration')
 initial=helper.digest(model);require(initial==next(x['state_before'] for x in load(D3/'model_checks.json') if x['model']=='IR' and x['stem']=='030129'),'Day03 persistent state')
 ds=ir.LLVIPDataset(ROOT/'data/LLVIP_1000',split='val',modality='infrared',infrared_channels=3,transforms=ir.PairedLetterboxAndFlip('val'));idx=next(i for i,r in enumerate(ds.records) if r.stem=='030129');image,target=ds[idx]
 require(eq(target['boxes'],old['gt_all']),'GT order/geometry')
 hooks=[(m,dict(m._forward_hooks),dict(m._forward_pre_hooks)) for m in model.modules()]
 methodkeys=[(model.rpn,k) for k in ['filter_proposals','_get_top_n_idx']]+[(model.roi_heads,'postprocess_detections')]
 methods=[(m,k,k in m.__dict__,m.__dict__.get(k)) for m,k in methodkeys]
 data={};cap=None;roi=model.roi_heads;original_post=roi.postprocess_detections
 def post(logits,regression,proposals,shapes):
  data.update(class_logits=cp(logits),box_regression=cp(regression),post_proposals=cp(proposals),image_shapes=shapes)
  def wrap(obj,name,key):
   fn=getattr(obj,name)
   def call(*a,**kw):
    result=fn(*a,**kw);data[key]=dict(args=cp(a),kwargs=cp(kw),result=cp(result));return result
   return call
  with ExitStack() as narrow:
   for obj,name,key in [(roi.box_coder,'decode','decode'),(roi_module.F,'softmax','softmax'),(roi_module.box_ops,'clip_boxes_to_image','clip'),(roi_module.box_ops,'remove_small_boxes','small'),(roi_module.box_ops,'batched_nms','nms')]:narrow.enter_context(patch.object(obj,name,wrap(obj,name,key)))
   result=original_post(logits,regression,proposals,shapes)
  data['actual_post_output']=cp(result);return result
 with ExitStack() as stack:
  def rpnpre(m,args):
   nonlocal cap
   images,features,targets=args;require(cap is None and images.image_sizes==[(640,640)],'RPN geometry/calls');cap=stack.enter_context(AgnosticCapture(m,tuple(features),target['boxes'],f'{out.name}/IR/030129'))
  def rpnpost(m,args,output):data['actual_rpn_return']=cp(output[0][0])
  def roipre(m,args):data['actual_roi_input']=cp(args[1]);data['actual_roi_shapes']=args[2]
  def predictor(m,args,output):data['predictor_output']=cp(output)
  def trans(m,args,output):data['transform_sizes']=output[0].image_sizes
  for m,kind,fn in [(model.rpn,'pre',rpnpre),(model.rpn,'post',rpnpost),(roi,'pre',roipre),(roi.box_predictor,'post',predictor),(model.transform,'post',trans)]:
   handle=m.register_forward_pre_hook(fn) if kind=='pre' else m.register_forward_hook(fn);stack.callback(handle.remove)
  stack.enter_context(patch.object(roi,'postprocess_detections',post))
  COUNT['full_model_forwards']+=1
  with torch.inference_mode():output=model([image])[0]
 data.update(model_output=cp(output),gt_all=cp(target['boxes']),rpn_records=cp(cap.records),rpn_stages=cp(cap.states),catalogues=cap.catalogues,configuration=cfg,run_identity=f'{out.name}/IR/030129')
 with (out/'raw_capture.pth').open('xb') as f:torch.save(data,f)
 restored=all(dict(m._forward_hooks)==a and dict(m._forward_pre_hooks)==b for m,a,b in hooks) and all((k in m.__dict__,m.__dict__.get(k))==(present,value) for m,k,present,value in methods)
 require(restored and helper.digest(model)==initial and helper.config(model)==cfg and sha(checkpoint)==expected,'state/config/checkpoint/hooks')
 save(out/'model_checks.json',dict(state_before=initial,state_after=helper.digest(model),configuration=cfg,hooks_methods_restored=restored,checkpoint_sha=sha(checkpoint),roi_source_sha=sha(source),counts=COUNT))
 return analyze(out,data,old,image)
def analyze(out,d,old,image):
 proposal=d['actual_roi_input'][0];r=d['rpn_records'][0];st=d['rpn_stages'][0];flats=st['post_nms_topn']['ids'].tolist();N=len(flats)
 require(N==51 and eq(proposal,r['final_boxes']) and eq(proposal,d['actual_rpn_return']) and eq(proposal,d['post_proposals'][0]),'true proposal row mapping')
 logits=d['class_logits'];reg=d['box_regression'];decoded=d['decode']['result'];prob=d['softmax']['result'];clipped=d['clip']['result'];g=d['gt_all'][0]
 require(logits.shape==(N,2) and reg.shape==(N,8) and decoded.shape==clipped.shape==(N,2,4) and prob.shape==(N,2),'class count/branch shape')
 require(eq(logits,d['predictor_output'][0]) and eq(reg,d['predictor_output'][1]),'predictor row identity')
 expanded=clipped[:,1:].reshape(-1,4);scores=prob[:,1:].reshape(-1);labels=torch.ones(N,dtype=torch.int64);ids=torch.arange(N)
 scoreidx=torch.where(scores>float(d['configuration']['roi']['score']))[0];smallargs=d['small']['args'][0]
 require(eq(expanded[scoreidx],smallargs),'reconstructed score indices vs actual small-box input')
 smallidx=d['small']['result'];nids=ids[scoreidx][smallidx];nb,ns,nl=d['nms']['args'][:3]
 require(eq(expanded[nids],nb) and eq(scores[nids],ns) and eq(labels[nids],nl),'actual NMS input mapped')
 nmsidx=d['nms']['result'];kept=nids[nmsidx];finalids=kept[:d['configuration']['roi']['detections']]
 pred=d['model_output'];actual=d['actual_post_output']
 require(eq(expanded[finalids],actual[0][0]) and eq(scores[finalids],actual[1][0]) and eq(labels[finalids],actual[2][0]),'actual post output')
 require(eq(pred['boxes'],actual[0][0]) and eq(pred['scores'],actual[1][0]) and eq(pred['labels'],actual[2][0]),'model output coordinate/order identity')
 branches=[]
 for i,f in enumerate(flats):
  for c in range(2):branches.append(dict(proposal_row=i,flat=f,candidate_id=d['catalogues'][0][f]['candidate_id'],class_label=c,rpn_box=proposal[i].tolist(),objectness=float(r['final_scores'][i]),rpn_gt_iou=float(box_iou(proposal[i:i+1],g[None])[0,0]),class_logits=logits[i].tolist(),box_regression=reg[i].tolist(),probability=float(prob[i,c]),decoded_box=decoded[i,c].tolist(),clipped_box=clipped[i,c].tolist(),decoded_gt_iou=float(box_iou(decoded[i,c:c+1],g[None])[0,0])))
 stage_defs=[('rpn_proposals',proposal,ids,'actual RPN/RoI capture'),('person_decoded',decoded[:,1],ids,'actual decode capture, person slice'),('person_clipped',clipped[:,1],ids,'actual clip capture, person slice'),('background_removed_expanded',expanded,ids,'offline original reshape; verified at actual small input'),('score_filtered',expanded[scoreidx],ids[scoreidx],'offline strict score indices; actual small input exact'),('small_filtered',nb,nids,'actual small indices and NMS input'),('nms_all_output',nb[nmsidx],kept,'actual NMS returned full indices'),('detections_budget',pred['boxes'],finalids,'slice verified with actual post output'),('model_output',pred['boxes'],finalids,'actual full model output')]
 records=[];coverage=[];trace=[];ti=flats.index(8877)
 for stage,boxes,rows,evidence in stage_defs:
  overlaps=box_iou(boxes,g[None])[:,0] if len(boxes) else torch.empty(0);good=[flats[int(rows[j])] for j in range(len(rows)) if float(overlaps[j])>=.75];maximum=float(overlaps.max()) if len(overlaps) else None
  coverage.append(dict(stage=stage,count=len(rows),max_gt_iou=maximum,covered=bool(good),qualifying_flats=good,evidence=evidence))
  for j,row in enumerate(rows.tolist()):records.append(dict(stage=stage,stage_index=j,proposal_row=row,flat=flats[row],class_label=1,xyxy=boxes[j].tolist(),gt_iou=float(overlaps[j]),classification_score=float(prob[row,1]) if stage!='rpn_proposals' else None,objectness=float(r['final_scores'][row]),evidence=evidence))
  hit=next((x for x in records if x['stage']==stage and x['flat']==8877),None);trace.append(dict(stage=stage,present=hit is not None,stage_index=hit['stage_index'] if hit else None,xyxy=hit['xyxy'] if hit else None,gt_iou=hit['gt_iou'] if hit else None,classification_score=float(prob[ti,1]),objectness=float(r['final_scores'][ti]),evidence=evidence))
 sources=[dict(output_index=j,proposal_row=int(row),flat=flats[int(row)],candidate_id=d['catalogues'][0][flats[int(row)]]['candidate_id'],label=int(pred['labels'][j]),score=float(pred['scores'][j]),xyxy=pred['boxes'][j].tolist(),gt_iou=float(box_iou(pred['boxes'][j:j+1],g[None])[0,0])) for j,row in enumerate(finalids)]
 oldids=old['stages'][0]['post_nms_topn']['ids'].tolist();oldmap={f:i for i,f in enumerate(oldids)};diff=[]
 for i,f in enumerate(flats):
  oi=oldmap.get(f);diff.append(dict(flat=f,new_row=i,day03_row=oi,box_equal=oi is not None and eq(proposal[i],old['records'][0]['final_boxes'][oi]),score_equal=oi is not None and eq(r['final_scores'][i],old['records'][0]['final_scores'][oi]),box_delta=(proposal[i]-old['records'][0]['final_boxes'][oi]).tolist() if oi is not None else None,score_delta=float(r['final_scores'][i]-old['records'][0]['final_scores'][oi]) if oi is not None else None))
 finaldiff=[]
 for i in range(max(len(pred['boxes']),len(old['model_output']['boxes']))):
  both=i<len(pred['boxes']) and i<len(old['model_output']['boxes']);finaldiff.append(dict(index=i,boxes_equal=both and eq(pred['boxes'][i],old['model_output']['boxes'][i]),scores_equal=both and eq(pred['scores'][i],old['model_output']['scores'][i]),labels_equal=both and eq(pred['labels'][i],old['model_output']['labels'][i]),box_delta=(pred['boxes'][i]-old['model_output']['boxes'][i]).tolist() if both else None,score_delta=float(pred['scores'][i]-old['model_output']['scores'][i]) if both else None))
 reproduced=flats==oldids and all(x['box_equal'] and x['score_equal'] for x in diff) and len(oldids)==N and all(x['boxes_equal'] and x['scores_equal'] and x['labels_equal'] for x in finaldiff) and eq(d['gt_all'],old['gt_all'])
 for name,rr in [('proposal_branches',branches),('stage_candidates',records),('stage_coverage',coverage),('trace_8877',trace),('final_sources',sources),('day03_flat_comparison',diff),('day03_final_comparison',finaldiff)]:csvout(out/(name+'.csv'),rr)
 save(out/'stage_indices.json',dict(flat_sequence=flats,score_filter=scoreidx.tolist(),small_filter=smallidx.tolist(),nms_all=nmsidx.tolist(),budget_proposal_rows=finalids.tolist(),score_indices_evidence='offline verified against next actual input'))
 save(out/'day03_reproduction.json',dict(exact_reproduced=reproduced,flat_sequence_equal=flats==oldids,gt_equal=eq(d['gt_all'],old['gt_all']),new_count=N,day03_count=len(oldids),final_count=len(pred['boxes']),scope='no tolerances; final order compared directly; proposals aligned by flat'))
 result=dict(status='passed',identity_chain_verified=True,day03_exact_reproduced=reproduced,counts=COUNT,proposal_count=N,classes=2,final_count=len(sources),target_final_index=next((s['output_index'] for s in sources if s['flat']==8877),None),target_person_score=float(prob[ti,1]),target_person_box=decoded[ti,1].tolist(),target_person_gt_iou=float(box_iou(decoded[ti,1:2],g[None])[0,0]),final_best_iou_source=max(sources,key=lambda s:s['gt_iou']),final_highest_score_source=max(sources,key=lambda s:s['score']))
 render(out,image,d['gt_all'],proposal[ti],decoded[ti,1],sources,coverage,result)
 lines=['# Week10 Day04：8877的真实RoI身份与阶段追踪','',f'8877的person分支解码坐标为{result["target_person_box"]}，对G0 IoU={result["target_person_gt_iou"]:.9f}，分类分数={result["target_person_score"]:.9f}。其最终输出索引为{result["target_final_index"]}（null表示未进入）。必须区分定位变化和筛选删除，下面完整列出所有阶段。','',f'最终最贴合G0的框来自flat={result["final_best_iou_source"]["flat"]}，输出索引{result["final_best_iou_source"]["output_index"]}；最高分框来自flat={result["final_highest_score_source"]["flat"]}，输出索引{result["final_highest_score_source"]["output_index"]}。身份来自真实行号和索引链，没有使用最近框匹配。','',f'Day03逐位复现：{reproduced}。若False，只解释本次新运行，不能声称这是Day03的内部过程；差异原样保留，无容差放宽、无重复前向。','', '|阶段|数量|最大GT IoU|达标flat|8877存在|8877 IoU|','|---|---:|---:|---|---|---:|']
 for c,t in zip(coverage,trace):lines.append(f'|{c["stage"]}|{c["count"]}|{c["max_gt_iou"]}|{c["qualifying_flats"]}|{t["present"]}|{t["gt_iou"]}|')
 lines+=['','## 证据性质','', '实际捕获：原RPN返回、实际RoI输入、predictor原始logits与regression、原decode/softmax/clip结果、原小框过滤与类别NMS索引、实际postprocess返回与完整模型输出。原函数照常调用，返回对象不改变。','经核对的离线重建：去背景、按类别展开、严格scores>0.001的分数索引，以及NMS后前100切片。分数过滤结果与原remove_small_boxes输入逐位一致；小框索引接到原NMS输入，完整NMS索引接到实际postprocess与模型输出，均核对顺序、框、分数与label。背景分支按原实现去掉，不加argmax类别筛选。','RPN阈值0.05/0.7、预算150，与RoI阈值0.001/0.5、预算100、小框min_size=0.01分开。原始图像1280×1024，Letterbox后640×640，上下padding64，内部及最终均640×640。完整GT顺序保留，对G0不达标不等于背景。','本例仅在固定CPU运行中描述阶段变化；没有反事实干预，不证明模型根因，不推广为AP75或全验证集结论。若某框被NMS删除，原返回索引只能确认去留，不能单独命名实际抑制者。本轮没有独立NMS贪心重放。','checkpoint加载一次，strict加载一次，完整模型前向一次，其他探测0次，无Fusion、训练、test、服务器或依赖下载。状态/config/hook/checkpoint核验见model_checks.json；所有历史文件哈希见history_hashes.json。','', '![身份流程](identity_flow.png)','![关键框](key_boxes.png)']
 (out/'case_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
def render(out,image,gts,before,after,sources,coverage,result):
 def text(d,x,y,s,size=22):d.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size),fill='#26354a')
 im=Image.new('RGB',(1550,820),'#f2f5fa');dr=ImageDraw.Draw(im);text(dr,20,15,'030129/IR：G0黄色，其他GT灰色；8877 RPN青色，person回归框粉红色',25)
 img=Image.fromarray(image.mul(255).round().byte().permute(1,2,0).numpy());bounds=(260,180,390,380)
 for col,(box,title,color) in enumerate([(before,'8877 RPN输入','#00c4b0'),(after,'8877 person解码','#ef6a92')]):
  ox=120+col*760;oy=100;rw,rh=325,500;im.paste(img.crop(bounds).resize((rw,rh)),(ox,oy));text(dr,ox,60,title)
  for j,b in enumerate(gts.tolist()+[box.tolist()]):
   c=color if j==len(gts) else '#ffd24a' if j==0 else '#aab3be';xy=[ox+(b[0]-260)*rw/130,oy+(b[1]-180)*rh/200,ox+(b[2]-260)*rw/130,oy+(b[3]-180)*rh/200];dr.rectangle(xy,outline=c,width=3)
 text(dr,20,650,f'回归后GT IoU={result["target_person_gt_iou"]:.9f}；person分数={result["target_person_score"]:.9f}；最终索引={result["target_final_index"]}')
 text(dr,20,700,'同次身份由proposal行与实际阶段索引确定；局部裁剪仅用于展示，未再推理。');im.save(out/'key_boxes.png')
 im=Image.new('RGB',(1500,160+len(coverage)*77),'#f2f5fa');dr=ImageDraw.Draw(im);text(dr,25,15,'真实身份链：RPN flat → RoI行 → person分支 → 阶段索引 → 最终输出',28)
 for i,c in enumerate(coverage):
  y=80+i*77;dr.rectangle((25,y,1460,y+60),outline='#7895b2',width=2);text(dr,40,y+10,f'{i}. {c["stage"]}  数量={c["count"]}  最大IoU={c["max_gt_iou"]}  达标flat={c["qualifying_flats"]}',21)
 im.save(out/'identity_flow.png')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();require(out.is_relative_to(HERE) and out!=HERE,'output scope');out.mkdir(parents=True,exist_ok=False)
 dirs=[ROOT/'week08',ROOT/'week09']+[ROOT/f'week10/day{i:02}' for i in range(1,4)]
 def snapshot():return {str(p):sha(p) for base in dirs for p in base.rglob('*') if p.is_file()}
 before=snapshot();save(out/'history_before.json',before);code=1;result={}
 class Tee:
  def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
  def flush(self):log.flush()
 with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
  try:result=run(out);code=0
  except BaseException:traceback.print_exc();save(out/'failure.json',dict(error=traceback.format_exc(),counts=COUNT))
  finally:
   after=snapshot();unchanged=before==after;save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
   if not unchanged:code=1
   result.update(status='passed' if code==0 else 'failed',historical_files_unchanged=unchanged,exit_code=code,counts=COUNT);save(out/'verification.json',result)
   save(out/'run_metadata.json',dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),environment=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform()),source_sha256=sha(Path(__file__)),counts=COUNT,exit_code=code,finished_at=datetime.datetime.now().astimezone().isoformat()));print(json.dumps(result,ensure_ascii=False));print('EXIT',code)
 save(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});return code
if __name__=='__main__':sys.exit(main())
