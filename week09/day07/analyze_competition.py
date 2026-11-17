"""Fixed Day07 T hypothesis; saved CPU captures only; no NMS execution."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,json,hashlib,struct,traceback,subprocess,platform
from pathlib import Path
from collections import Counter,defaultdict
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
import torch,torchvision
from torchvision.ops import box_iou
from torchvision.ops import boxes as ops

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
D4=ROOT/'week09/day04/outputs/full_val_stage_run01'
D6=ROOT/'week09/day06/outputs/layer_support_run01'
KEYS=('0','1','pool')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def write(p,rows,fields=None):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def scalar(a,b,round32=True):
    q=f32 if round32 else float
    aa=q(q(a[2]-a[0])*q(a[3]-a[1]));bb=q(q(b[2]-b[0])*q(b[3]-b[1]))
    inter=q(max(0.,q(min(a[2],b[2])-max(a[0],b[0])))*max(0.,q(min(a[3],b[3])-max(a[1],b[1]))))
    return q(inter/q(q(aa+bb)-inter))

def metric(boxes,scores,levels,ids,gt):
    assert len(ids)==len(set(ids)), 'Duplicate actual candidate identity'
    assert boxes.dtype==gt.dtype==scores.dtype==torch.float32 and boxes.device.type=='cpu'
    v=box_iou(gt[None],boxes)[0];good=torch.where(v>=.75)[0]
    if not len(good):raise ValueError('Empty good set in common risk cohort')
    overlaps=box_iou(boxes[good],boxes)
    same=levels[good,None]==levels[None,:]
    geometric=same & (v[None,:]<.75) & (overlaps>.7)
    strict=geometric & (scores[None,:]>scores[good,None])
    ties=geometric & (scores[None,:]==scores[good,None])
    counts=strict.sum(1)
    return v,good,overlaps,strict,ties,counts,int(counts.min())

def synthetic():
    g=torch.tensor([0.,0.,10.,10.]);a=[0.,0.,10.,10.];b=[0.,0.,7.4,10.]
    def run(bb,ss,ll,ids=None):return metric(torch.tensor(bb),torch.tensor(ss),torch.tensor(ll),ids or list(range(len(bb))),g)
    assert run([a,b],[.5,.9],[0,0])[-1]==1
    assert run([a,b],[.5,.9],[0,1])[-1]==0
    t=run([a,b],[.5,.5],[0,0]);assert t[-1]==0 and int(t[4].sum())==1
    assert run([a,b],[.9,.5],[0,0])[-1]==0
    assert run([a,[0.,0.,7.,10.]],[.5,.9],[0,0])[-1]==0 # exact .7 excluded
    assert run([a,[0.,0.,7.5,10.]],[.5,.9],[0,0])[-1]==0 # exact .75 is good
    assert run([a,b,a],[.5,.9,.95],[0,0,0])[-1]==0 # minimum over good candidates
    assert run([a,b,b],[.5,.9,.9],[0,0,0])[-1]==2 # identical coordinates distinct IDs
    try:run([a,b],[.5,.9],[0,0],[1,1]);raise AssertionError('duplicate accepted')
    except AssertionError as e:assert str(e)=='Duplicate actual candidate identity'
    try:run([b],[.5],[0]);raise AssertionError('empty accepted')
    except ValueError:pass
    assert direction(2,1)=='positive' and direction(1,1)=='equal' and direction(0,1)=='negative'
    assert Counter(direction(a,b) for a,b in [(2,1),(1,1),(0,1)])=={'positive':1,'equal':1,'negative':1}
    assert scalar(a,b)==float(box_iou(torch.tensor([a]),torch.tensor([b]))[0,0])
    return {'status':'passed','checks':13,'scope':'synthetic unit checks, not LLVIP results'}
def direction(f,s):return 'positive' if f>s else 'negative' if f<s else 'equal'

def analyze(out):
    for p in [ROOT/'week09/day05/checks/independent_run01/verification.json',ROOT/'week09/day06/checks/independent_run01/verification.json']:
        assert load(p)['status']=='passed'
    pairs=read(D6/'all_common_gt_pairs.csv');asym=read(D6/'asymmetric_pairs.csv')
    assert len(pairs)==len({(r['stem'],r['gt_index']) for r in pairs})==297
    selected=[r for r in pairs if r['failed_model'] and int(r['IR_L'])==int(r['Fusion_L'])]
    assert len(selected)==23 and selected==[r for r in asym if int(r['delta_L'])==0]
    ledger={(r['model'],r['stem']):r for r in read(D4/'forward_ledger.csv')}
    groups=defaultdict(list)
    for r in pairs:groups[r['stem']].append(r)
    support=[];goods=[];relations=[];ties=[];audits=[];provenance=[]
    for stem,cohort in sorted(groups.items()):
      for model in ('IR','Fusion'):
        e=ledger[model,stem];p=D4/e['relative_raw'];assert sha(p)==e['raw_sha256']
        d=torch.load(p,map_location='cpu',weights_only=True);r=d['records'][0];stage=d['stages'][0]['score_filtered']
        boxes=stage['boxes'];scores=stage['scores'];levels=r['nms_levels'];flat=stage['ids'];cat=d['catalogues'][0]
        assert tuple(r['layer_keys'])==KEYS and r['config']['nms_thresh']==.7
        assert torch.equal(boxes,r['nms_boxes']) and torch.equal(scores,r['nms_scores'])
        full=torch.repeat_interleave(torch.arange(3),torch.tensor(r['counts']))
        assert torch.equal(full[flat],levels)
        ids=[cat[int(i)]['candidate_id'] for i in flat]
        assert len(set(ids))==len(ids)
        for pos,i in enumerate(flat.tolist()):assert cat[i]['level_key']==KEYS[int(levels[pos])] and cat[i]['flat_index']==i
        bb=boxes.tolist();ss=scores.tolist();ll=levels.tolist()
        for c in cohort:
            gi=int(c['gt_index']);gt=d['observed']['transform_gt_all'][gi];g=gt.tolist()
            v,good,ov,strict,tie,counts,T=metric(boxes,scores,levels,ids,gt)
            context={'model':model,'stem':stem,'gt_index':gi,'scene_id':c['scene_id']}
            # Independent scalar area formula, binary32 after EACH arithmetic operation.
            vv=[scalar(g,b) for b in bb];assert vv==v.tolist()
            gg=[i for i,x in enumerate(vv) if x>=.75];assert gg==good.tolist()
            independent_counts=[]
            for pos,x in enumerate(vv):
                diag=scalar(g,bb[pos],False)
                if abs(x-.75)<=1e-6 or (x>=.75)!=(diag>=.75):
                    audits.append(dict(**context,kind='GT_threshold',a_id=ids[pos],b_id='',primary=x,float64=diag,decision_diff=(x>=.75)!=(diag>=.75)))
            for ai,a in enumerate(gg):
                expected=[];expected_ties=[]
                for b in range(len(bb)):
                    if ll[a]!=ll[b] or vv[b]>=.75 or ss[b]<ss[a]:continue
                    x=scalar(bb[a],bb[b]);assert x==float(ov[ai,b])
                    diag=scalar(bb[a],bb[b],False)
                    if abs(x-.7)<=1e-6 or (x>.7)!=(diag>.7):
                        audits.append(dict(**context,kind='pair_threshold',a_id=ids[a],b_id=ids[b],primary=x,float64=diag,decision_diff=(x>.7)!=(diag>.7)))
                    if x>.7:(expected if ss[b]>ss[a] else expected_ties).append(b)
                assert expected==torch.where(strict[ai])[0].tolist() and expected_ties==torch.where(tie[ai])[0].tolist()
                independent_counts.append(len(expected))
                goods.append(dict(**context,a_id=ids[a],layer=KEYS[ll[a]],input_position=a,flat_index=int(flat[a]),xyxy=bb[a],gt_xyxy=g,score=ss[a],gt_iou=vv[a],C=len(expected),tie_count=len(expected_ties),is_T_minimizer=len(expected)==T))
                for b in expected+expected_ties:
                    row=dict(**context,a_id=ids[a],b_id=ids[b],layer=KEYS[ll[a]],a_position=a,b_position=b,a_flat=int(flat[a]),b_flat=int(flat[b]),a_xyxy=bb[a],b_xyxy=bb[b],a_score=ss[a],b_score=ss[b],a_gt_iou=vv[a],b_gt_iou=vv[b],pair_iou=scalar(bb[a],bb[b]),pair_iou_float64=scalar(bb[a],bb[b],False))
                    (relations if b in expected else ties).append(row)
            assert independent_counts==counts.tolist() and min(independent_counts)==T
            L=len({ll[i] for i in gg});assert L==int(c[model+'_L'])
            # Outcome consulted ONLY after T is fixed, never inside competition metric.
            lost=not any(vv[int(i)]>=.75 for i in r['nms_indices'])
            assert lost==(c[model+'_nms_lost']=='True')
            support.append(dict(**context,T=T,L=L,nms_lost=lost,good_count=len(gg),minimizer_ids=[ids[a] for a,n in zip(gg,independent_counts) if n==T],tied_overlap_exists=bool(tie.any())))
        provenance.append(dict(model=model,stem=stem,path=str(p),sha256=e['raw_sha256']))
        print('RAW_COMPETITION_VERIFIED',model,stem,flush=True)
    lookup={(r['model'],r['stem'],r['gt_index']):r for r in support};paired=[]
    for c in selected:
        gi=int(c['gt_index']);failed=c['failed_model'];survivor=c['surviving_model'];a=lookup[failed,c['stem'],gi];b=lookup[survivor,c['stem'],gi]
        paired.append(dict(stem=c['stem'],gt_index=gi,scene_id=c['scene_id'],L=a['L'],failed_model=failed,surviving_model=survivor,failed_T=a['T'],surviving_T=b['T'],delta_T=a['T']-b['T'],direction=direction(a['T'],b['T'])))
    summaries=[]
    for scene in ('ALL','03','08','13','17'):
      for model in ('ALL','IR','Fusion'):
        rr=[p for p in paired if (scene=='ALL' or p['scene_id']==scene) and (model=='ALL' or p['failed_model']==model)]
        n=len(rr);cc=Counter(p['direction'] for p in rr)
        summaries.append(dict(scene_id=scene,failed_model=model,GT=n,**{k:cc[k] for k in ('positive','equal','negative')},**{k+'_rate':cc[k]/n if n else None for k in ('positive','equal','negative')}))
    background=[]
    for scene in ('ALL','03','08','13','17'):
      for model in ('IR','Fusion'):
        rr=[r for r in support if r['model']==model and (scene=='ALL' or r['scene_id']==scene)]
        for T in sorted({r['T'] for r in rr}):
            tt=[r for r in rr if r['T']==T];n=len(tt);lost=sum(r['nms_lost'] for r in tt)
            background.append(dict(scene_id=scene,model=model,T=T,risk_GT=n,nms_lost_GT=lost,loss_rate=lost/n))
    for model in ('IR','Fusion'):
        rr=[x for x in background if x['scene_id']=='ALL' and x['model']==model];assert sum(x['risk_GT'] for x in rr)==297
    for name,rows in [('per_gt_metrics',support),('good_candidates',goods),('competition_relations',relations),('equal_score_relations',ties),('precision_audit',audits),('paired_23',paired),('paired_summary',summaries),('model_scene_T_summary',background)]:
        write(out/(name+'.csv'),rows,fields=list(rows[0]) if rows else ['model','stem','gt_index','note'])
    save(out/'capture_provenance.json',provenance)
    result=dict(status='passed',GT=297,model_GT=594,main_pairs=23,directions=dict(Counter(p['direction'] for p in paired)),good_candidate_GT_rows=len(goods),strict_relation_rows=len(relations),equal_score_relation_rows=len(ties),audit_rows=len(audits),audit_disagreements=sum(a['decision_diff'] for a in audits),raw_captures=len(provenance),independent_scalar_verification='exact membership, C, T, raw layers and outcomes; binary32 area formula; all candidate GT IoUs exact',new_forwards=0,NMS_runs=0)
    save(out/'verification.json',result)
    lines=['# Day07：NMS前高分重叠竞争离线检验','',str(result),'','ΔT=失败T−保住T；正值符合假设，0无法区分，负值为反例。母集固定297GT，主比较从真实Day06配对表筛得23项L相同而结局不同的GT。','', '|场景|失败模型|分母|正向|相同|反向|','|---|---|---:|---:|---:|---:|']
    lines += [f"|{s['scene_id']}|{s['failed_model']}|{s['GT']}|{s['positive']}|{s['equal']}|{s['negative']}|" for s in summaries]
    lines += ['', '## 全部23项名单','','|GT|场景|失败模型|失败T|保住T|ΔT|方向|','|---|---|---|---:|---:|---:|---|']
    lines += [f"|{p['stem']}/G{p['gt_index']}|{p['scene_id']}|{p['failed_model']}|{p['failed_T']}|{p['surviving_T']}|{p['delta_T']}|{p['direction']}|" for p in paired]
    lines += ['', '## 全母集背景（实际T值，无事后分箱）','','|场景|模型|T|风险GT|丢失GT|条件率|','|---|---|---:|---:|---:|---:|']
    lines += [f"|{p['scene_id']}|{p['model']}|{p['T']}|{p['risk_GT']}|{p['nms_lost_GT']}|{p['loss_rate']:.4%}|" for p in background]
    lines += ['', '## 解释与限制','', 'T数的是：在至少一个好框的周围，最少有几个同层、对本GT不达标、分数严格更高、重叠严格超过0.7的候选。竞争集合全部来自score_filtered，不使用保留集合。相同坐标不同ID仍分别计数。T不是NMS模拟，也没有确定GPU实际抑制者。','不达标只针对指定GT，不等于背景或错误语义类别。T=0不保证存活，尤其不能排除同分重叠；T>0也不保证丢失。所有同分关系单列，不进入主指标。','主口径CPU float32，0.75与0.7均无容差。1e-6仅标记阈值附近；float64为诊断，未扩大主集合。GT阈值分歧审计遍历全部输入候选，两框阈值审计遍历主好框面对的同层主不达标且分数不低候选；诊断不用于替换主分类。','独立验证使用另一套标量面积公式并逐步舍入binary32，从原始坐标核对所有GT IoU、竞争关系、C及T；另按原层索引核对。汇总按独立核验后的逐GT结果重新计数。','这些均为描述性比较，没有显著性或因果检验；不能直接证明Fusion特征或回归根因。不能用本指标解释真实逐次抑制过程。本轮到此停止。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tests-only',action='store_true');args=ap.parse_args()
    out=args.output.resolve();assert out.is_relative_to(HERE);out.mkdir(parents=True,exist_ok=False)
    protected=[p for day in range(1,7) for p in (ROOT/f'week09/day{day:02}').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    before={str(p):sha(p) for p in protected};code=1
    meta=dict(command=subprocess.list2cmdline(sys.orig_argv),environment=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform(),device='CPU'),source_sha256=sha(Path(__file__)),historical_before=before)
    class Tee:
        def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
        def flush(self):log.flush()
    with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
      try:
        torch.set_num_threads(2)
        with patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Forward forbidden')),patch.object(ops,'nms',side_effect=RuntimeError('NMS forbidden')),patch.object(ops,'batched_nms',side_effect=RuntimeError('NMS forbidden')):
            tests=synthetic();save(out/'synthetic_tests.json',tests);print(tests,flush=True)
            if not args.tests_only:
                meta['input_sha256']={str(p):sha(p) for p in [D6/'all_common_gt_pairs.csv',D6/'asymmetric_pairs.csv',D4/'forward_ledger.csv',D4/'artifact_manifest.json']}
                result=analyze(out);print(json.dumps(result),flush=True)
        code=0
      except BaseException:
        meta['failure']=traceback.format_exc();traceback.print_exc();save(out/'failure.json',dict(error=meta['failure']))
      finally:
        meta['historical_after']={str(p):sha(p) for p in protected};meta['historical_unchanged']=before==meta['historical_after']
        if not meta['historical_unchanged']:code=1
        meta['exit_code']=code;save(out/'run_metadata.json',meta);print('EXIT',code,'HISTORY_UNCHANGED',meta['historical_unchanged'],flush=True)
    return code
if __name__=='__main__':sys.exit(main())
