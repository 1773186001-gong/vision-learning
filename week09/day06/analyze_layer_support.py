"""Day06: one fixed L hypothesis, existing captures only."""
import argparse
from collections import Counter,defaultdict
from contextlib import redirect_stdout,redirect_stderr
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
import torchvision
from torchvision.ops import boxes as ops
from layer_support import KEYS,SCENES,measure,make_pairs,direction_summary,background_summary

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
D4=ROOT/'week09/day04/outputs/full_val_stage_run01'
D5=ROOT/'week09/day05/outputs/full_val_nms_relations_run01'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
    import csv
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def write(p,rows):
    import csv
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def flag(v):
    if v not in ('True','False'):raise ValueError('Unexpected source boolean')
    return v=='True'
def exact(a,b):
    if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise ValueError('Raw tensor mismatch')
def rate(v):return '—' if v is None else f'{100*v:.2f}%'

def inspect_raw(data):
    if len(data['records'])!=1 or len(data['stages'])!=1:raise ValueError('Single image capture expected')
    r=data['records'][0];states=data['stages'][0];pre=states['score_filtered'];post=states['layer_nms']
    gt=data['observed']['transform_gt_all'];cat=data['catalogues'][0]
    if tuple(r['layer_keys'])!=KEYS or r['image_size']!=(640,640):raise ValueError('Actual keys/geometry changed')
    if (r['config']['pre_nms'],r['config']['post_nms'],r['config']['score_thresh'],r['config']['nms_thresh'])!=(150,150,.05,.7):raise ValueError('Fixed config differs')
    exact(pre['boxes'],r['nms_boxes']);exact(pre['scores'],r['nms_scores'])
    keep=r['nms_indices']
    if keep.dtype!=torch.int64 or len(keep.unique())!=len(keep) or bool(((keep<0)|(keep>=len(pre['ids']))).any()):raise ValueError('Original NMS indices invalid')
    for field in ('ids','boxes','scores'):exact(post[field],pre[field][keep])
    ids=pre['ids'];levels=r['nms_levels']
    full_levels=torch.repeat_interleave(torch.arange(len(KEYS)),torch.tensor(r['counts']))
    if ids.dtype!=torch.int64 or len(ids.unique())!=len(ids):raise ValueError('Invalid original flat IDs')
    if bool(((ids<0)|(ids>=len(full_levels))).any()):raise ValueError('Flat ID outside real counts')
    exact(levels,full_levels[ids])
    for pos,flat in enumerate(ids.tolist()):
        e=cat[flat]
        if e['flat_index']!=flat or e['candidate_id']!=f"{r['prefix']}/{flat}" or e['level_key']!=KEYS[int(levels[pos])]:
            raise ValueError('Original layer/catalogue mapping differs')
    if 'gt_all' in r:exact(gt,r['gt_all'])
    return r,pre,post,gt,cat

def analyze(out):
    accepted=load(ROOT/'week09/day05/checks/independent_run01/verification.json')
    if accepted['status']!='passed' or accepted['exit_code']!=0:raise ValueError('Day05 not accepted')
    cohort=read(D5/'common_risk_pairing.csv')
    if len(cohort)!=297 or len({(c['stem'],int(c['gt_index'])) for c in cohort})!=297:raise ValueError('Common-risk keys/count differ')
    expected_categories=Counter({'both_lost':10,'IR_only_lost':11,'Fusion_only_lost':20,'neither_lost':256})
    if Counter(c['loss_pair_category'] for c in cohort)!=expected_categories:raise ValueError('Day05 category counts differ')
    source=defaultdict(dict)
    for s in read(D4/'per_gt_stage.csv'):
        key=s['model'],s['stem'],int(s['gt_index'])
        if s['stage'] in source[key]:raise ValueError('Duplicate source GT/stage')
        source[key][s['stage']]=s
    allkeys={k[1:] for k in source if k[0]=='IR'}
    actual_common={k for k in allkeys if all(flag(source[m,*k]['score_filtered']['covered_075_float32_cpu']) for m in ('IR','Fusion'))}
    if actual_common!={(c['stem'],int(c['gt_index'])) for c in cohort}:raise ValueError('Mother set differs from actual Day04 precoverage')
    ledger=read(D4/'forward_ledger.csv');lookup={(e['model'],e['stem']):e for e in ledger}
    if len(lookup)!=len(ledger):raise ValueError('Duplicate source capture ledger')
    bystem=defaultdict(list)
    for c in cohort:
        if not all(flag(c[k]) for k in ('IR_precovered','Fusion_precovered','both_precovered')):raise ValueError('Mother set includes zero risk')
        bystem[c['stem']].append(c)
    support=[];layers=[];trace=[];provenance=[]
    for stem,cs in sorted(bystem.items()):
      for model in ('IR','Fusion'):
        e=lookup[model,stem];path=D4/e['relative_raw']
        if sha(path)!=e['raw_sha256']:raise ValueError('Capture SHA differs')
        d=torch.load(path,map_location='cpu',weights_only=True)
        r,pre,post,gt,cat=inspect_raw(d)
        for c in cs:
            gi=int(c['gt_index']);old=source[model,stem,gi]['score_filtered']
            if c['scene_id']!=old['scene_id'] or json.loads(old['gt_xyxy'])!=gt[gi].tolist():raise ValueError('GT index/coordinates/scene differ')
            L,counts,identities,values=measure(gt[gi],pre['boxes'],r['nms_levels'],pre['ids'],cat,r['layer_keys'])
            if sum(counts.values())!=int(old['count_iou_ge_075_float32_cpu']):raise ValueError('Good candidates differ from Day04')
            postcovered=bool((values[r['nms_indices']]>=.75).any())
            lost=not postcovered
            if lost!=flag(c[model+'_nms_lost']) or postcovered!=flag(c[model+'_after_nms_covered']):raise ValueError('Day05 loss state differs')
            if postcovered!=flag(source[model,stem,gi]['layer_nms']['covered_075_float32_cpu']):raise ValueError('Day04 NMS state differs')
            row=dict(model=model,stem=stem,gt_index=gi,scene_id=c['scene_id'],gt_xyxy=json.dumps(gt[gi].tolist()),
                L=L,nms_lost=lost,nms_survived=postcovered,supported_layer_keys=json.dumps([key for key in KEYS if counts[key]>0]),
                **{'count_'+key:counts[key] for key in KEYS})
            support.append(row)
            for level,key in enumerate(KEYS):
                layers.append(dict(model=model,stem=stem,gt_index=gi,scene_id=c['scene_id'],layer_key=key,
                    good_candidate_count=counts[key],candidate_ids=json.dumps(identities[key]),L=L))
                for pos in torch.where((values>=.75)&(r['nms_levels']==level))[0].tolist():
                    flat=int(pre['ids'][pos])
                    trace.append(dict(model=model,stem=stem,gt_index=gi,scene_id=c['scene_id'],layer_key=key,
                        candidate_id=cat[flat]['candidate_id'],flat_index=flat,nms_input_position_0based=pos,
                        candidate_xyxy=json.dumps(pre['boxes'][pos].tolist()),gt_xyxy=json.dumps(gt[gi].tolist()),
                        iou_float32_cpu=float(values[pos])))
        provenance.append(dict(model=model,stem=stem,relative_raw=e['relative_raw'],raw_sha256=e['raw_sha256'],
            selected_gt_indices=json.dumps([int(c['gt_index']) for c in cs]),actual_layer_keys=list(r['layer_keys']),
            actual_layer_counts=list(r['counts']),raw_deserializations=1))
        print('CAPTURE_VERIFIED',model,stem,'common_GT',len(cs),flush=True)
    pairs=make_pairs(support,cohort)
    if Counter(p['loss_pair_category'] for p in pairs)!=expected_categories:raise ValueError('Raw-recomputed categories differ')
    asym=[p for p in pairs if p['failed_model'] is not None]
    if len(asym)!=31 or Counter(p['failed_model'] for p in asym)!=Counter({'IR':11,'Fusion':20}):raise ValueError('Asymmetric cohort differs')
    directions=direction_summary(pairs);background=background_summary(support)
    if len(support)!=594 or len(layers)!=1782:raise ValueError('Missing GT/model/layer rows')
    tables={'per_gt_support':support,'per_layer_support':layers,'good_candidate_trace':trace,
        'all_common_gt_pairs':pairs,'asymmetric_pairs':asym,'direction_summary':directions,'model_scene_L_summary':background}
    for name,rows in tables.items():write(out/(name+'.csv'),rows)
    save(out/'capture_provenance.json',provenance)
    summary={'status':'passed','hypothesis':'failed model may have fewer pre-NMS supporting feature levels',
        'primary_metric':'L only; CPU float32 IoU >= .75; no threshold tolerance','common_GT':len(cohort),
        'images':len(bystem),'model_image_captures_loaded_once':len(provenance),'model_GT_rows':len(support),
        'layer_rows':len(layers),'good_candidate_GT_trace_rows':len(trace),'asymmetric_GT':len(asym),
        'category_counts':dict(Counter(p['loss_pair_category'] for p in pairs)),
        'direction_counts':dict(Counter(p['direction'] for p in asym)),
        'tables_rows':{k:len(v) for k,v in tables.items()}}
    save(out/'analysis_summary.json',summary);report(out,summary,asym,directions,background)
    return summary

def report(out,summary,asym,directions,background):
    lines=['# Week09 Day06：NMS前跨层达标支持与覆盖存活','',
        '本轮唯一假设：同一GT上NMS后失去覆盖的模型，其NMS前包含达标候选的实际特征层数L可能较少。允许被数据否定。只读既有捕获，没有服务器、checkpoint、新模型前向、NMS运行、训练、调参或test。','',
        '## 固定母集与口径','',
        f"母集为Day05已核验297个共同风险GT，涉及{summary['images']}张图；两模型各297个GT项。实际分类为{summary['category_counts']}。逐项用原坐标核对风险和丢失状态，不用旧报告文字填数。",
        'CPU float32 IoU≥0.75，不加阈值容差。实际层键是0、1、pool，按原NMS输入层标识、展平索引和候选目录相互核对，不按框大小猜层。每层多个好框仍只贡献一个支持层；母集L=0即报错，不丢弃。','',
        '## 31个不对称GT的方向','',
        '|范围|场景|GT分母|失败L更少|L相同|失败L更多|',
        '|---|---|---:|---:|---:|---:|']
    for r in directions:
        lines.append(f"|{r['failed_model']}|{r['scene_id']}|{r['asymmetric_GT']}|{r['fewer_GT']}|{r['equal_GT']}|{r['more_GT']}|")
    lines+=['','ΔL=失败模型L−保住覆盖模型L。负值符合假设方向，0说明L在该GT上不能区分成败，正值为相反方向。分母为0的比例在CSV留空；不把相同或反向案例过滤掉。','',
        '## 全部31项名单','',
        '|stem/GT（0起始）|场景|失败模型|失败L|保住覆盖模型|保住L|ΔL|方向|',
        '|---|---|---|---:|---|---:|---:|---|']
    for p in asym:
        lines.append(f"|{p['stem']}/G{p['gt_index']}|{p['scene_id']}|{p['failed_model']}|{p['failed_L']}|{p['surviving_model']}|{p['surviving_L']}|{p['delta_L']}|{p['direction']}|")
    lines+=['','## 完整母集按L分组的背景','',
        '|模型|场景|L|NMS前风险GT数|NMS覆盖丢失GT|条件率|','|---|---|---:|---:|---:|---:|']
    for r in background:
        lines.append(f"|{r['model']}|{r['scene_id']}|{r['L']}|{r['risk_GT']}|{r['nms_lost_GT']}|{rate(r['conditional_loss_rate'])}|")
    lines+=['',
        'L分组是各模型在共同母集中的描述性背景；各L组未必包含相同GT，不等于额外做了同GT对照。小分母必须连同计数读，零分母留空。多层支持不保证存活，不假定各层独立，不能由这些条件率推断增加一层就能改善检测。','',
        '## 对假设的回答方式与边界','',
        f"完整31项：失败L更少{summary['direction_counts'].get('fewer',0)}项，相同{summary['direction_counts'].get('equal',0)}项，更多{summary['direction_counts'].get('more',0)}项。分别按失败模型与场景的完整方向见上表。",
        '这里只检验预先给定的L描述假设；不临时添加目标大小、分数间隔、候选排名等解释指标。符合方向的实例是描述性支持，反向实例限制通用解释，相同实例说明单凭L不足。不能把关联写成因果，也不能据此直接改网络。',
        '仍未解释：真实GPU实际抑制事件、为何形成特定候选与分数结构、融合特征或回归的根因、RoI阶段及最终AP75差异。没有把可能关系写成实际抑制配对。','',
        '## 可追溯性与验收','',
        'per_gt_support.csv含594项模型×GT的L及逐层数量；per_layer_support.csv含1782项逐层候选数与原ID；good_candidate_trace.csv保存达标候选原坐标、原索引、实际层及CPU float32 IoU。asymmetric_pairs.csv是31项完整名单。原NMS全部输出核对丢失，不提前截至最终150。',
        '输入SHA、来源、设备／dtype、实际命令、退出码及历史文件前后保护见run_metadata.json、capture_provenance.json、run.log、execution_status.json；人工测试和独立原始坐标核验另存checks。未覆盖历史运行。',
        '本轮完成这一项离线假设检验后停止，等待反馈。']
    with (out/'day06_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s);stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:stream.flush()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(HERE):raise ValueError('Day06 output only')
    out.mkdir(exist_ok=False,parents=True)
    protected=[q for day in ('day01','day02','day03','day04','day05') for q in (ROOT/'week09'/day).rglob('*') if q.is_file() and '__pycache__' not in q.parts]
    before={str(q):sha(q) for q in protected}
    meta={'status':'running','scope':'Day05_297_common_risk_GT_L_only','command':subprocess.list2cmdline(sys.orig_argv),
        'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'environment':{'python':sys.version,
        'executable':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'device':'cpu','platform':platform.platform()},
        'source_sha256':{str(p):sha(p) for p in HERE.glob('*.py')},'historical_sha256_before':before,
        'checkpoint_loaded':False,'model_forward_performed':False,'NMS_executed':False,'server_connected':False}
    code=1
    with (out/'run.log').open('x',encoding='utf-8') as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
       try:
        print('COMMAND',meta['command'],flush=True);torch.set_num_threads(2)
        manifest=load(D4/'artifact_manifest.json')
        if not all(sha(D4/n)==h for n,h in manifest.items()):raise ValueError('Day04 input artifact SHA differs')
        inputs=[D4/'artifact_manifest.json',D4/'forward_ledger.csv',D4/'per_gt_stage.csv',
            D5/'common_risk_pairing.csv',D5/'paired_summary.csv',
            ROOT/'week09/day05/checks/independent_run01/verification.json']
        meta['input_sha256']={str(p):sha(p) for p in inputs};meta['Day04_artifact_files_SHA_verified']=len(manifest)
        with patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forward forbidden')), \
             patch.object(ops,'nms',side_effect=RuntimeError('NMS forbidden')),patch.object(ops,'batched_nms',side_effect=RuntimeError('NMS forbidden')):
            info=analyze(out)
        meta['status']='completed';meta['analysis_summary']=info;code=0
        print('L_SUPPORT_ANALYSIS_COMPLETED',json.dumps(info),flush=True)
       except BaseException:
        meta['status']='failed';meta['error']=traceback.format_exc();traceback.print_exc()
        save(out/'failure.json',{'status':'failed','error':meta['error'],'automatic_rerun':False})
       finally:
        meta['historical_sha256_after']={str(q):sha(q) for q in protected}
        meta['historical_files_unchanged']=before==meta['historical_sha256_after']
        if not meta['historical_files_unchanged']:meta['status']='failed';code=1
        meta['exit_code']=code;meta['finished_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        save(out/'run_metadata.json',meta);save(out/'execution_status.json',{'status':meta['status'],'exit_code':code})
        print('OFFLINE_EXIT',code,'historical_files_unchanged',meta['historical_files_unchanged'],flush=True)
    return code
if __name__=='__main__':sys.exit(main())
