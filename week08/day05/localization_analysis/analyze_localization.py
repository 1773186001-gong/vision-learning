"""Offline localization diagnostics; trusted prediction caches only, no model forward."""
import sys,json,csv,time,platform
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'week08/day05'))
from error_trace import trace
from fusion_model import ir
from torchvision.transforms.functional import to_pil_image
BASE=Path(__file__).parent
PATHS={'IR':ROOT/'week08/day03/outputs/ir_baseline_run01/test_predictions.pth',
       'RGB':ROOT/'week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/test_predictions.pth',
       'Fusion':ROOT/'week08/day05/outputs/fusion_concat_run01/test_predictions.pth'}


def save(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(x):
    a=np.asarray(x,dtype=float)
    if not len(a): return dict(n=0,mean=None,median=None,std_population=None,min=None,max=None,q25=None,q50=None,q75=None)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),std_population=float(a.std(ddof=0)),min=float(a.min()),max=float(a.max()),
                **dict(zip(['q25','q50','q75'],np.quantile(a,[.25,.5,.75]).tolist())))


def ap_at(cache,threshold):
    # Exact same match and 101-point expression as ir.metrics; no score>=.5 filtering for AP.
    events,total=ir.match(cache['predictions'],cache['truths'],threshold)
    tp=0; points=[]
    for n,(_,correct) in enumerate(events,1):
        tp+=correct; points.append((tp/max(total,1),tp/n))
    return sum(max((p for r,p in points if r>=k/100),default=0) for k in range(101))/101 if total else 0.


def main():
    torch.set_num_threads(2); out=BASE/'results'
    if out.exists(): raise FileExistsError('Existing results must not be overwritten')
    expected={}
    expected['IR']=json.loads((ROOT/'week08/day04/protected_inputs.json').read_text())['files']['week08/day03/outputs/ir_baseline_run01/test_predictions.pth']
    expected['RGB']=json.loads((PATHS['RGB'].parent/'artifact_sha256.json').read_text())['test_predictions.pth']['sha256']
    expected['Fusion']=json.loads((ROOT/'week08/day05/checks/final_review_1793608106351600200/artifact_sha256.json').read_text())['test_predictions.pth']['sha256']
    hashes={m:ir.sha(p) for m,p in PATHS.items()}; assert hashes==expected
    caches={m:torch.load(p,map_location='cpu',weights_only=False) for m,p in PATHS.items()}
    ref=caches['IR']; stems=ref['stems']; assert len(stems)==len(set(stems))==150
    metrics={}; traces={}; tp_rows=[]; ap_rows=[]; match_rows=[]; tp_stats=[]; high=[]; high_details=[]; owners={}
    for mode,c in caches.items():
        assert c['stems']==stems and len(c['predictions'])==len(c['truths'])==150 and sum(map(len,c['truths']))==348
        assert all(torch.equal(a,b) for a,b in zip(c['truths'],ref['truths']))
        metrics[mode]=ir.metrics(c['predictions'],c['truths'])
        assert metrics[mode]==json.loads((PATHS[mode].parent/'test_metrics.json').read_text()),'Historical metrics mismatch; stop'
        aps=[ap_at(c,.5+.05*j) for j in range(10)]
        assert aps[0]==metrics[mode]['AP50'] and sum(aps)/10==metrics[mode]['mAP50_95']
        metrics[mode]['AP75']=aps[5]
        traces[mode]=[]; owners[mode]={}
        for i,(p,g) in enumerate(zip(c['predictions'],c['truths'])):
            assert len(p['scores'])<=100 and (p['labels']==1).all() and torch.isfinite(p['boxes']).all()
            rows=trace(p,g); traces[mode].append(rows)
            for r in rows:
                if r['status']=='TP':
                    j=r['matched_gt']; iou=r['ious'][j]
                    item=dict(model=mode,image_id=stems[i],gt_index=j,prediction_id=r['prediction_id'],score=r['score'],matched_iou=iou,box=r['box'])
                    tp_rows.append(item);owners[mode][(stems[i],j)]=item
                if r['selected'] and r['score']>=.9: high_details.append(dict(model=mode,image_id=stems[i],**r))
        values=[r['matched_iou'] for r in tp_rows if r['model']==mode]
        tp_stats.append(dict(model=mode,**stats(values),**{f'count_ge_{v:.2f}':sum(x>=v for x in values) for v in [.6,.7,.75,.8,.9]}))
        hs=[r for r in high_details if r['model']==mode]; ht=[r for r in hs if r['status']=='TP']; hf=[r for r in hs if r['status']=='FP']
        types=Counter(r['fp_type'] for r in hf); hti=[r['ious'][r['matched_gt']] for r in ht]
        high.append(dict(model=mode,all_high=len(hs),high_TP=len(ht),high_FP=len(hf),FP_fraction_of_all_high=len(hf)/len(hs) if hs else None,
            high_TP_IoU_ge75=sum(v>=.75 for v in hti),high_TP_IoU_ge75_fraction=sum(v>=.75 for v in hti)/len(hti) if hti else None,
            **{'high_TP_'+k:v for k,v in stats(hti).items()},**{'high_FP_maxIoU_'+k:v for k,v in stats([r['max_iou'] for r in hf if r['max_iou'] is not None]).items()},
            **{'high_FP_'+t:types[t] for t in ['duplicate','localization','low_overlap','other']}))
        for j,ap in enumerate(aps):
            threshold=.5+.05*j; events,n=ir.match(c['predictions'],c['truths'],threshold,.5); tp=sum(v for _,v in events)
            ap_rows.append(dict(model=mode,IoU=threshold,AP=ap));match_rows.append(dict(model=mode,IoU=threshold,score_threshold=.5,TP=tp,FP=len(events)-tp,FN=n-tp))
    print('Historical metrics gate PASS',json.dumps(metrics),flush=True)
    common=sorted(owners['IR'].keys()&owners['Fusion'].keys()); only_ir=sorted(owners['IR'].keys()-owners['Fusion'].keys()); only_fusion=sorted(owners['Fusion'].keys()-owners['IR'].keys())
    groups=dict(common=len(common),only_IR=len(only_ir),only_Fusion=len(only_fusion),neither=348-len(common)-len(only_ir)-len(only_fusion))
    assert groups==dict(common=295,only_IR=12,only_Fusion=12,neither=29)
    paired=[]
    for key in common:
        a,b=owners['IR'][key],owners['Fusion'][key]
        paired.append(dict(image_id=key[0],gt_index=key[1],IR_prediction=a['prediction_id'],Fusion_prediction=b['prediction_id'],IR_box=json.dumps(a['box']),Fusion_box=json.dumps(b['box']),
            IR_score=a['score'],Fusion_score=b['score'],IR_IoU=a['matched_iou'],Fusion_IoU=b['matched_iou'],delta=b['matched_iou']-a['matched_iou']))
    deltas=[r['delta'] for r in paired]; eps=1e-6
    paired_summary=dict(**groups,approx_equal_tolerance=eps,Fusion_better=sum(v>eps for v in deltas),IR_better=sum(v < -eps for v in deltas),approximately_equal=sum(abs(v)<=eps for v in deltas),delta_statistics=stats(deltas),
        top10_absolute_difference=sorted(paired,key=lambda r:-abs(r['delta']))[:10])
    out.mkdir(); (out/'case_examples').mkdir()
    save(out/'metrics_validation.json',metrics); save(out/'paired_summary.json',paired_summary);save(out/'high_confidence_prediction_details.json',high_details)
    save(out/'all_prediction_matching_details.json',traces)
    table(out/'tp_iou_statistics.csv',tp_stats);table(out/'tp_matches.csv',tp_rows);table(out/'high_confidence_analysis.csv',high)
    table(out/'ap_by_iou.csv',ap_rows);table(out/'matching_by_iou.csv',match_rows);table(out/'paired_gt_iou_comparison.csv',paired)
    save(out/'exclusive_GT_groups.json',dict(only_IR=only_ir,only_Fusion=only_fusion))
    correlations={}
    for mode in caches:
        vals=[r for r in tp_rows if r['model']==mode]
        correlations[mode]=dict(n=len(vals),definition='Pearson over original fixed-match TP only; not calibration or causation',pearson=float(np.corrcoef([r['score'] for r in vals],[r['matched_iou'] for r in vals])[0,1]))
    save(out/'score_iou_correlation.json',correlations)
    def plot(name,draw):
        fig,ax=plt.subplots(figsize=(8,5));draw(ax);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    def hist(ax,high_only=False):
        for m in caches:
            vals=[r['matched_iou'] for r in tp_rows if r['model']==m and (not high_only or r['score']>=.9)]
            ax.hist(vals,bins=np.linspace(.5,1,21),histtype='step',linewidth=2,label=f'{m} n={len(vals)}')
        ax.set(xlabel='Matched TP IoU',ylabel='TP count');ax.legend()
    plot('tp_iou_distribution',hist);plot('high_confidence_tp_iou',lambda ax:hist(ax,True))
    for name,rows,key,ylabel in [('tp_vs_iou_threshold',match_rows,'TP','TP at score >= .5'),('ap_vs_iou_threshold',ap_rows,'AP','AP, all cached scores')]:
        def draw(ax,rows=rows,key=key,ylabel=ylabel):
            for m in caches:
                vals=[r for r in rows if r['model']==m];ax.plot([r['IoU'] for r in vals],[r[key] for r in vals],marker='o',label=m)
            ax.set(xlabel='Matching IoU threshold',ylabel=ylabel);ax.legend()
        plot(name,draw)
    def scatter(ax):
        for m in caches:
            vals=[r for r in tp_rows if r['model']==m];ax.scatter([r['score'] for r in vals],[r['matched_iou'] for r in vals],s=10,alpha=.5,label=m)
        ax.set(xlabel='Score (fixed-match TP)',ylabel='Matched IoU');ax.legend()
    plot('confidence_vs_iou_scatter',scatter)
    def fpplot(ax):
        xs=np.arange(3);bottom=np.zeros(3)
        for t in ['duplicate','localization','low_overlap','other']:
            vals=[r['high_FP_'+t] for r in high];ax.bar(xs,vals,bottom=bottom,label=t);bottom+=vals
        ax.set_xticks(xs,[r['model'] for r in high]);ax.set_ylabel('FP count among score >= .9 predictions');ax.legend()
    plot('high_confidence_fp_comparison',fpplot)
    def delta_plot(ax):
        ax.hist(deltas,bins=30);ax.axvline(0,color='black');ax.set(xlabel='Fusion - IR matched IoU on common GT',ylabel='GT count')
    plot('paired_gt_iou_difference',delta_plot)
    cases=[('IR_better',min(paired,key=lambda r:r['delta'])['image_id'],min(paired,key=lambda r:r['delta'])['gt_index']),
           ('Fusion_better',max(paired,key=lambda r:r['delta'])['image_id'],max(paired,key=lambda r:r['delta'])['gt_index'])]
    for label,threshold,high_only in [('high_score_low_iou',.5,True),('boundary_050',.5,False),('boundary_075',.75,False)]:
        candidates=[r for r in tp_rows if r['model'] in ('IR','Fusion') and (not high_only or r['score']>=.9)]
        chosen=min(candidates,key=lambda r:abs(r['matched_iou']-threshold));cases.append((label,chosen['image_id'],chosen['gt_index']))
    cases += [('only_IR',*only_ir[0]),('only_Fusion',*only_fusion[0]),('preferred_190474','190474',0)]
    ds=ir.LLVIPDataset(ROOT/'data/LLVIP_1000','test',modality='paired',infrared_channels=3,transforms=ir.PairedLetterboxAndFlip('test'))
    assert [r.stem for r in ds.records]==stems
    case_details=[]
    for label,stem,j in cases:
        i=stems.index(stem);images,target=ds[i];g=ref['truths'][i]
        assert torch.equal(target['boxes'],g) and images['visible'].shape==images['infrared'].shape==(3,640,640)
        box=g[j].tolist(); zoom=[max(0,box[0]-35),min(640,box[2]+35),min(640,box[3]+35),max(0,box[1]-35)]
        fig,axs=plt.subplots(2,2,figsize=(13,12)); texts=[]
        for col,mode in enumerate(['IR','Fusion']):
            rs=traces[mode][i];used={r['matched_gt'] for r in rs if r['status']=='TP'}
            relevant=[r for r in rs if r['selected'] and (r['matched_gt']==j or (r['ious'] and max(r['ious'])>0 and r['ious'].index(max(r['ious']))==j))]
            case_details.append(dict(case=label,model=mode,image_id=stem,gt_index=j,GT=box,predictions=relevant))
            texts.append(mode+': '+ ('; '.join(f"P{r['prediction_id']} {r['status']} s={r['score']:.6f} IoU(G{j})={r['ious'][j]:.6f}" for r in relevant) or 'no selected prediction associated with this GT'))
            for row in (0,1):
                ax=axs[row,col]; ax.imshow(to_pil_image(images['infrared' if mode=='IR' else 'visible']))
                for k,b in enumerate(g.tolist()):ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='green' if k in used else 'cyan',linewidth=2.5 if k==j else 1))
                for r in rs:
                    if not r['selected']:continue
                    b=r['box'];color='blue' if r['status']=='TP' else 'red'
                    ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1))
                    ax.text(b[0],b[1],f"P{r['prediction_id']}",color=color,fontsize=7,clip_on=True,bbox=dict(facecolor='white',alpha=.7,pad=0))
                ax.set(xlim=(0,640),ylim=(640,0),title=f'{mode} {stem} focus G{j}'+(' zoom' if row else ''))
                if row: ax.set_xlim(zoom[:2]);ax.set_ylim(zoom[2:])
        fig.legend(handles=[Patch(color=c,label=l) for l,c in [('GT matched','green'),('GT missed FN','cyan'),('TP','blue'),('FP','red')]],loc='lower center',ncol=4)
        fig.text(.02,.045,'\n'.join(texts),fontsize=8,wrap=True);fig.suptitle(label);fig.tight_layout(rect=(0,.12,1,.97));fig.savefig(out/'case_examples'/f'{label}_{stem}_G{j}.png',dpi=160);plt.close(fig)
    save(out/'case_examples/details.json',case_details)
    assert hashes=={m:ir.sha(p) for m,p in PATHS.items()}
    metadata=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=ir.torchvision.__version__,matplotlib=matplotlib.__version__,numpy=np.__version__,device='CPU',
        command=f'{sys.executable} -B {Path(__file__).resolve()}',cache_paths={m:str(p) for m,p in PATHS.items()},cache_SHA256=hashes,cache_unchanged=True,
        matching_source_SHA256=ir.sha(ROOT/'week08/day03/ir_common.py'),trace_source_SHA256=ir.sha(ROOT/'week08/day05/error_trace.py'),no_model_forward=True,no_network=True)
    save(out/'run_metadata.json',metadata)
    lines=['# Day05 Step8：IR与融合离线定位质量分析交接','本次已实际运行CPU离线分析，无模型构造/forward、训练、CUDA或网络。原始缓存前后SHA一致。历史指标复算门槛通过；只分析已有预测，不改变模型、best或部署阈值。',
        '## 环境、命令、可信缓存','```json\n'+json.dumps(metadata,ensure_ascii=False,indent=2)+'\n```',
        '加载前分别对照Day04保护清单、RGB artifact清单、Day05最终核验清单SHA；仅固定可信项目文件使用torch.load(map_location=cpu,weights_only=False)。predictions为逐图boxes/scores/labels Tensor，truths为GT xyxy Tensor，stems为编号；150图348GT，三缓存顺序及GT完全相等。',
        '## 匹配与AP复算','稳定降序分数排序，已占用GT IoU置-1，再选剩余最大IoU；每GT最多匹配一次。直接使用ir_common.match/metrics及已核对逐事件一致的error_trace.trace。AP按所有缓存预测分数，不筛到score≥.5。不同IoU重新匹配，101点插值表达式与metrics相同，并断言AP50及十阈值均值逐值相等。该自定义协议不是完整官方COCO API。',
        '```json\n'+json.dumps(metrics,indent=2)+'\n```','## TP IoU统计（原始score≥.5、IoU≥.5 TP）','标准差采用总体ddof=0。阈值计数只筛原匹配TP，不等于提高阈值后重新匹配，也不等于AP75。','```json\n'+json.dumps(tp_stats,indent=2)+'\n```',
        '## 不同IoU重新匹配及AP','完整30行TP/FP/FN见matching_by_iou.csv，AP见ap_by_iou.csv。',
        '|IoU|IR TP|Fusion TP|IR AP|Fusion AP|差值|\n|---|---:|---:|---:|---:|---:|']
    for j in range(10):
        t=.5+.05*j; a=next(r for r in ap_rows if r['model']=='IR' and r['IoU']==t);b=next(r for r in ap_rows if r['model']=='Fusion' and r['IoU']==t)
        counts=[next(r['TP'] for r in match_rows if r['model']==m and r['IoU']==t) for m in ['IR','Fusion']]
        lines[-1]+=f"\n|{t:.2f}|{counts[0]}|{counts[1]}|{a['AP']:.6f}|{b['AP']:.6f}|{b['AP']-a['AP']:+.6f}|"
    lines+=['## 高置信度分析','score≥.9仅在原score≥.5匹配标签完成后筛选，没有重新匹配。FP比例分母为所有高分预测；高分TP IoU≥.75比例分母为高分TP；FP maxIoU不等于有效匹配IoU。','```json\n'+json.dumps(high,indent=2)+'\n```',
        'Pearson样本仅是各模态原固定匹配TP，受TP条件筛选及场景相关性影响，不代表概率校准误差，也不能推断因果。','```json\n'+json.dumps(correlations,indent=2)+'\n```',
        '## 相同GT配对比较','delta=Fusion−IR，abs(delta)≤1e-6视为近似相等。完整295行含框坐标和分数见paired_gt_iou_comparison.csv。差异最大Top10按绝对差排序。','```json\n'+json.dumps(paired_summary,indent=2)+'\n```',
        '## 真实案例与选择规则','IR优势/融合优势选共同GT中差值两端；高分低IoU选score≥.9的TP中最接近.5者；边界选TP中最接近.5/.75者（不改阈值）；独有检出按编号和GT索引最先者；另含190474。所有选择保留全局统计，不掩盖整体不足。边界案例仅展示TP侧最近值，不能声称覆盖边界两侧。不同类别可能同图，独有检出另一模型可能存在FP或低分框，不意味着完全没有任何预测。',
        '上排640全图，下排同坐标局部放大；融合结果画在RGB底图，IR结果画在IR底图。颜色图例区分TP/FP/FN，图下精确分数及针对焦点GT的IoU。标记P编号以减少遮挡，完整逐框信息见details.json。既有Dataset验证原RGB/IR/XML尺寸和变换GT，未强行对齐或独立验证像素配准。']
    for label,stem,j in cases:lines.append(f'![{label}](results/case_examples/{label}_{stem}_G{j}.png)')
    lines+=['## 四个问题的证据化回答',
        '1. AP50稍高、AP75下降：全阈值表直接显示严格匹配表现下降；结合TP分布和同GT差值进一步区分位置质量。这些是缓存层面的观察，AP同时受分数排序/重复/漏检影响，不能仅据TP均值解释全部AP变化，不能确定是配准、融合权重或优化过程造成。',
        '2. 高分不保证高IoU：请比较高分TP均值、中位数及≥.75比例，并看高分FP。不能仅凭融合高分FP少就说高分框定位更准；高分分数并非IoU预测，分母已在表中明确。',
        '3. 共同GT谁更高：以paired_summary中IR_better/Fusion_better及delta均值为准；这是295个共同TP子集，不能涵盖24个独有检出或29个共同漏检。',
        '4. 真实定位不足：IR_better图直接展示相同GT两框的差异，high_score_low_iou和阈值边界图说明高分或刚过0.5并不意味着精定位。图中几何不匹配不能直接证明背景误识别或漏标。',
        '## 输出与后续学习','7张统计图、8张案例图、各CSV/JSON及终端run.log均独立保存。诊断阈值.9和近似相等1e-6不是部署调参。单seed与连续场景样本不独立，未进行显著性检验。下一步先学习IoU匹配、AP与固定阈值统计、相同GT配对的优势及子集偏差；可能的配准误差、回归优化、融合竞争均仅是假设，需另行授权设计验证。本次停止，不追加训练。']
    for name in ['tp_iou_distribution','tp_vs_iou_threshold','ap_vs_iou_threshold','confidence_vs_iou_scatter','high_confidence_tp_iou','high_confidence_fp_comparison','paired_gt_iou_difference']:lines.append(f'![{name}](results/{name}.png)')
    with (BASE/'DAY05_LOCALIZATION_ANALYSIS_GPT_HANDOFF.md').open('x',encoding='utf-8') as f:f.write('\n\n'.join(lines))
    print(json.dumps(dict(tp_stats=tp_stats,high=high,paired=paired_summary),ensure_ascii=False,indent=2));print('PASS: offline statistics complete; input hashes unchanged.')


if __name__=='__main__':main()
