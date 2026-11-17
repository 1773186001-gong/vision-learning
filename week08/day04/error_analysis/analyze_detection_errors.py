"""Offline cache analysis only. No model construction, inference, CUDA or network."""
import csv
import json
import sys
import platform
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'week08/day03'))
import ir_common as ir
from torchvision.transforms.functional import to_pil_image

BASE = Path(__file__).parent
PATHS = {
    'IR': ROOT/'week08/day03/outputs/ir_baseline_run01/test_predictions.pth',
    'RGB': ROOT/'week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/test_predictions.pth',
}
TYPES = ['duplicate', 'localization', 'low_overlap', 'other']
COLORS = {'GT':'green','TP':'blue','duplicate':'orange','localization':'purple','low_overlap':'red','other':'gray'}


def dump(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def csv_write(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def trace(p, gt):
    """Instrument ir.match verbatim decisions; verify complete event sequence per image."""
    used=set(); owner={}; rows=[]
    for idx in torch.argsort(p['scores'], descending=True, stable=True).tolist():
        score=float(p['scores'][idx])
        overlaps=ir.box_iou(p['boxes'][idx:idx+1],gt)[0]
        selected=score>=0.5; hit=None
        if selected and len(gt):
            available=overlaps.clone()
            for j in used: available[j]=-1
            value,j=available.max(0)
            if float(value)>=0.5:
                hit=int(j); used.add(hit); owner[hit]=idx
        rows.append(dict(prediction_id=idx,score=score,box=p['boxes'][idx].tolist(),
                         selected=selected,status=('TP' if hit is not None else 'FP') if selected else 'below_score',
                         matched_gt=hit,ious=overlaps.tolist(),
                         max_iou=float(overlaps.max()) if len(gt) else None))
    events,total=ir.match([p],[gt],0.5,0.5)
    assert events==[(r['score'],int(r['status']=='TP')) for r in rows if r['selected']]
    for r in rows:
        r.update(fp_type=None,duplicate_links=[],reason=None,no_gt_fp=False)
        if r['status']!='FP': continue
        links=[dict(gt_id=j,tp_prediction_id=owner[j],fp_gt_iou=r['ious'][j],
                    tp_gt_iou=float(ir.box_iou(p['boxes'][owner[j]:owner[j]+1],gt)[0,j]))
               for j in sorted(owner) if r['ious'][j]>=0.5]
        if links: category='duplicate'
        elif not len(gt): category='low_overlap'; r['no_gt_fp']=True
        elif .1<=r['max_iou']<.5: category='localization'
        elif r['max_iou']<.1: category='low_overlap'
        else: category='other'; r['reason']='maxIoU >= .5 but no finally matched GT overlap; inspect ordering'
        r['fp_type']=category; r['duplicate_links']=links
    return rows


def distribution(values):
    return dict(n=len(values),min=min(values) if values else None,max=max(values) if values else None,
                mean=float(np.mean(values)) if values else None,
                quantiles=dict(zip(['q0','q25','q50','q75','q100'],np.quantile(values,[0,.25,.5,.75,1]).tolist())) if values else {})


def main():
    torch.set_num_threads(2)
    result=BASE/'results'
    if result.exists(): raise FileExistsError('Existing analysis results; do not overwrite')
    # Only the two trusted project caches, with historical SHA records checked BEFORE loading.
    hashes={k:ir.sha(v) for k,v in PATHS.items()}
    protected=json.loads((ROOT/'week08/day04/protected_inputs.json').read_text())['files']
    assert hashes['IR']==protected[str(PATHS['IR'].relative_to(ROOT)).replace('\\','/')]
    rgb_manifest=json.loads((PATHS['RGB'].parent/'artifact_sha256.json').read_text())
    assert hashes['RGB']==rgb_manifest['test_predictions.pth']['sha256']
    caches={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in PATHS.items()}
    assert caches['RGB']['modality']=='visible'
    assert caches['IR']['stems']==caches['RGB']['stems']
    stems=caches['IR']['stems']; assert len(stems)==len(set(stems))==150
    for cache in caches.values():
        assert len(cache['predictions'])==len(cache['truths'])==150
        assert sum(map(len,cache['truths']))==348
        for p,g in zip(cache['predictions'],cache['truths']):
            assert p['boxes'].dtype==g.dtype==torch.float32 and g.shape[1:]==(4,)
            assert (p['labels']==1).all()
            for boxes in (p['boxes'],g):
                assert torch.isfinite(boxes).all() and (boxes>=0).all() and (boxes<=640).all()
                assert (boxes[:,2:]>boxes[:,:2]).all()
            assert torch.isfinite(p['scores']).all()
    for a,b in zip(caches['IR']['truths'],caches['RGB']['truths']): assert torch.equal(a,b)
    # Mandatory gate: use historical matching itself, not rounded historical metrics.
    metrics={}
    for mode,cache in caches.items():
        events,n=ir.match(cache['predictions'],cache['truths'],.5,.5)
        tp=sum(v for _,v in events)
        metrics[mode]=dict(TP=tp,FP=len(events)-tp,FN=n-tp,GT=n)
        assert tuple(metrics[mode][k] for k in ('TP','FP','FN'))=={'IR':(307,54,41),'RGB':(299,309,49)}[mode], 'Historical count mismatch; stop attribution'
    print('Historical ir.match reproduction:',json.dumps(metrics),flush=True)
    # Validate original paired images/XML/640 coordinates for ALL 150 samples before comparisons.
    ds=ir.LLVIPDataset(ROOT/'data/LLVIP_1000','test',modality='paired',infrared_channels=3,
                       transforms=ir.PairedLetterboxAndFlip('test'))
    assert [r.stem for r in ds.records]==stems
    example_images={}
    for i,stem in enumerate(stems):
        images,target=ds[i]
        assert images['visible'].shape==images['infrared'].shape==(3,640,640)
        assert torch.equal(target['boxes'],caches['IR']['truths'][i]),stem
        if stem in ('190001','190040','190474'): example_images[stem]=images
    result.mkdir(); (result/'comparison_examples').mkdir()
    per_image=[]; per_pred=[]; details={}; summaries={}
    for mode,cache in caches.items():
        images=[]
        for i,(stem,p,g) in enumerate(zip(stems,cache['predictions'],cache['truths'])):
            rows=trace(p,g); counts=Counter(r['status'] for r in rows)
            types=Counter(r['fp_type'] for r in rows if r['status']=='FP')
            item=dict(modality=mode,image_id=stem,dataset_image_id=i,GT=len(g),predictions_above_threshold=counts['TP']+counts['FP'],
                      TP=counts['TP'],FP=counts['FP'],FN=len(g)-counts['TP'],**{t:types[t] for t in TYPES})
            per_image.append(item); images.append(dict(**item,gt_boxes=g.tolist(),predictions=rows))
            for r in rows:
                per_pred.append(dict(modality=mode,image_id=stem,**{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()}))
        details[mode]=images
        fps=[p for im in images for p in im['predictions'] if p['status']=='FP']
        types=Counter(p['fp_type'] for p in fps); assert sum(types.values())==metrics[mode]['FP']
        assert {k:sum(im[k] for im in images) for k in ('TP','FP','FN')}=={k:metrics[mode][k] for k in ('TP','FP','FN')}
        summaries[mode]=dict(**metrics[mode],categories={t:dict(count=types[t],fraction=types[t]/len(fps)) for t in TYPES},
            no_gt_images=sum(im['GT']==0 for im in images),no_gt_fp=sum(p['no_gt_fp'] for p in fps),
            per_image_fp_distribution=dict(sorted(Counter(im['FP'] for im in images).items())),
            score_distribution=distribution([p['score'] for p in fps]),
            max_iou_distribution=distribution([p['max_iou'] for p in fps if p['max_iou'] is not None]),
            high_score_definition='score >= 0.9 (diagnostic only; no threshold tuning)',
            high_score_fp=sum(p['score']>=.9 for p in fps),
            high_score_fp_fraction=sum(p['score']>=.9 for p in fps)/len(fps),
            top10=sorted([r for r in per_image if r['modality']==mode],key=lambda r:(-r['FP'],r['image_id']))[:10])
        dump(result/f'{mode.lower()}_error_summary.json',summaries[mode])
        dump(result/f'{mode.lower()}_matching_details.json',images)
    csv_write(result/'per_image_errors.csv',per_image); csv_write(result/'per_prediction_errors.csv',per_pred)
    comparisons=[dict(category=t,IR=summaries['IR']['categories'][t]['count'],RGB=summaries['RGB']['categories'][t]['count'],
                      IR_fraction=summaries['IR']['categories'][t]['fraction'],RGB_fraction=summaries['RGB']['categories'][t]['fraction'],
                      delta=summaries['RGB']['categories'][t]['count']-summaries['IR']['categories'][t]['count']) for t in TYPES]
    csv_write(result/'fp_comparison.csv',comparisons)
    paired=[dict(image_id=s,IR_FP=details['IR'][i]['FP'],RGB_FP=details['RGB'][i]['FP'],delta=details['RGB'][i]['FP']-details['IR'][i]['FP']) for i,s in enumerate(stems)]
    paired.sort(key=lambda x:(-x['delta'],x['image_id'])); csv_write(result/'paired_fp_difference.csv',paired)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for j,mode in enumerate(caches):
        axs[0].bar(np.arange(4)+j*.35,[summaries[mode]['categories'][t]['count'] for t in TYPES],width=.35,label=mode)
        axs[1].bar(np.arange(4)+j*.35,[100*summaries[mode]['categories'][t]['fraction'] for t in TYPES],width=.35,label=mode)
    for ax in axs: ax.set_xticks(np.arange(4)+.175,TYPES); ax.legend()
    axs[0].set_ylabel('FP count'); axs[1].set_ylabel('% of modality FP'); fig.tight_layout(); fig.savefig(result/'fp_type_comparison.png',dpi=180); plt.close(fig)
    for key,name,bins in [('score','fp_score_distribution',np.linspace(.5,1,11)),('max_iou','fp_iou_distribution',[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])]:
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for mode in caches:
            vals=[p[key] for im in details[mode] for p in im['predictions'] if p['status']=='FP' and p[key] is not None]
            axs[0].hist(vals,bins=bins,histtype='step',linewidth=2,label=mode)
            axs[1].hist(vals,bins=bins,weights=np.ones(len(vals))/max(len(vals),1),histtype='step',linewidth=2,label=mode)
        for ax in axs: ax.set_xlabel(key); ax.legend()
        axs[0].set_ylabel('FP count'); axs[1].set_ylabel('Fraction of valid values'); fig.tight_layout(); fig.savefig(result/(name+'.png'),dpi=180); plt.close(fig)
    examples={}
    for stem,images in example_images.items():
        i=stems.index(stem); examples[stem]={m:details[m][i] for m in caches}
        fig,axs=plt.subplots(1,2,figsize=(16,9))
        for mode,ax,channel in zip(['RGB','IR'],axs,['visible','infrared']):
            info=details[mode][i]; ax.imshow(to_pil_image(images[channel]))
            for j,b in enumerate(info['gt_boxes']):
                ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='green',linewidth=2)); ax.text(b[0],b[3],f'G{j}',color='green',bbox=dict(facecolor='white',alpha=.7,pad=0),fontsize=8)
            for p in info['predictions']:
                if not p['selected']: continue
                b=p['box']; color=COLORS['TP' if p['status']=='TP' else p['fp_type']]
                ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.6))
                label=f"P{p['prediction_id']} {p['score']:.2f} iou={p['max_iou']:.2f}"
                ax.text(min(b[0],480),max(8,b[1]-4),label,color=color,fontsize=7,bbox=dict(facecolor='white',alpha=.75,pad=0))
            ax.set(xlim=(0,640),ylim=(640,0),title=f"{mode} {stem} TP={info['TP']} FP={info['FP']} FN={info['FN']}")
        fig.legend(handles=[Patch(color=v,label=k) for k,v in COLORS.items()],loc='lower center',ncol=6)
        fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(result/'comparison_examples'/f'{stem}_rgb_ir.png',dpi=180); plt.close(fig)
    dump(result/'comparison_examples/matching_details.json',examples)
    assert hashes=={k:ir.sha(v) for k,v in PATHS.items()}
    env=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=ir.torchvision.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),device='cpu',command=' '.join([sys.executable,'-B',str(Path(__file__).resolve())]),
             caches={k:dict(path=str(v),sha256=hashes[k],keys=list(caches[k]),prediction_fields=list(caches[k]['predictions'][0])) for k,v in PATHS.items()},
             validation='all 150 original paired samples have identical transformed GT to both caches; equal image dimensions; no independent image registration audit',
             matching_source_sha256=ir.sha(ROOT/'week08/day03/ir_common.py'),cache_sha_unchanged=True)
    dump(result/'run_metadata.json',env)
    lines=['# Day04 Step7：IR/RGB 离线检测错误分析交接',
           '本次仅加载两份已核验SHA256的项目预测缓存，不加载模型/权重，不做前向推理、训练、CUDA或远程操作。全部结果为本次重新计算。',
           '## 环境、命令和缓存结构','```json\n'+json.dumps(env,ensure_ascii=False,indent=2)+'\n```',
           '缓存predictions为每图boxes(float32 N×4)、scores(float32 N)、labels(int64 N)；truths为每图GT xyxy Tensor，stems为同顺序编号。IR无modality键，RGB含visible。CPU加载map_location=cpu、weights_only=False，与历史项目代码一致；仅允许固定路径、加载前验证历史哈希的可信缓存。',
           '## 匹配复核（先执行的停止门槛）',json.dumps(metrics,indent=2),
           '直接调用ir_common.match，score≥0.5、IoU≥0.5。每图torch.argsort(descending=True,stable=True)按分数排序；对已匹配GT将IoU置-1，在剩余GT中取最大IoU（并列取最先索引），达到阈值则占用。不是先找全局最佳GT再判断其是否被占用。追踪版本逐图逐事件与原函数完全相等；数据集事件最终稳定按分数排序，不改变计数。所有编号为0起始：prediction_id保持原缓存索引，dataset_image_id为测试集索引，image_id为文件stem。',
           '## 分类规则',
           '只对score≥0.5的FP分类。优先duplicate：与最终已由其他TP匹配的任一GT IoU≥0.5，保存所有关联GT、TP编号及各自IoU；再localization：0.1≤maxIoU<0.5；再low_overlap：maxIoU<0.1，无GT图计入本类且额外no_gt_fp=true、maxIoU=null；其余other并记录原因。每个FP恰好一个主类别。0.1与高分0.9是探索性诊断阈值，不改变历史评估，不调参。低重合不等于已证实背景误识别，定位候选不等于已经人工确认定位错误。',
           '## 分类实际数量与占比',
           '|类型|IR数量|IR占FP%|RGB数量|RGB占FP%|RGB−IR|\n|---|---:|---:|---:|---:|---:|']
    lines[-1]+='\n'+'\n'.join(f"|{r['category']}|{r['IR']}|{100*r['IR_fraction']:.2f}|{r['RGB']}|{100*r['RGB_fraction']:.2f}|{r['delta']}|" for r in comparisons)
    for mode in caches: lines+=['## '+mode+' 完整统计（分布、空GT、高分、Top10）','```json\n'+json.dumps(summaries[mode],ensure_ascii=False,indent=2)+'\n```']
    largest=max(comparisons,key=lambda r:r['delta'])
    lines+=['## 两模态对比',f"绝对增量最大为{largest['category']}，增加{largest['delta']}个。总FP增加255个。占比差异请同时读表，数量差不能替代占比差。",
            '高分(score≥0.9)FP：'+ '; '.join(f"{m}={summaries[m]['high_score_fp']} ({summaries[m]['high_score_fp_fraction']:.2%} of FP)" for m in caches),
            'RGB−IR FP差值Top10：```json\n'+json.dumps(paired[:10],indent=2)+'\n```',
            '## 三张同编号实际框、分数、IoU和匹配',
            '所有图保持640×640坐标，绿GT/蓝TP/橙duplicate/紫localization/红low_overlap/灰other，图底有图例。原始RGB/IR尺寸及XML由既有Dataset检查，所有150张变换GT逐元素等于两缓存。该检查不能独立证明传感器的每像素物理配准，也未将图像强行重新对齐。']
    for stem,info in examples.items():
        lines+=['### '+stem, f'![RGB IR {stem}](results/comparison_examples/{stem}_rgb_ir.png)']
        for mode,im in info.items():
            lines += [f"{mode}: GT={im['GT']}, TP={im['TP']}, FP={im['FP']}, FN={im['FN']}; GT框="+json.dumps(im['gt_boxes']),
                      '|预测编号|xyxy|score|状态/类型|maxIoU|匹配GT|对所有GT的IoU|重复关联|\n|---|---|---|---|---|---|---|---|']
            lines[-1]+='\n'+'\n'.join(f"|{p['prediction_id']}|{p['box']}|{p['score']:.6f}|{p['status']}/{p['fp_type']}|{p['max_iou']}|{p['matched_gt']}|{p['ious']}|{p['duplicate_links']}|" for p in im['predictions'] if p['selected'])
    lines+=['## 输出、可支持结论与限制',
            'results中两份error_summary、两份matching_details(JSON包含每图全部缓存预测，含低于阈值框)、per_image_errors.csv、per_prediction_errors.csv（列表字段为JSON）、fp_comparison.csv、paired_fp_difference.csv、3张统计图、3张双模态对照图及详细样例JSON。所有FP分类和每图计数均守恒，缓存运行前后SHA一致。',
            '![类型](results/fp_type_comparison.png)\n\n![分数](results/fp_score_distribution.png)\n\n![IoU](results/fp_iou_distribution.png)',
            '可支持：固定测试集、阈值与匹配下的几何错误组成、分数分布和同图差异；额外FP属于哪种候选可量化。不能支持：自动证明背景根因、判断所有低重合框都是误识别（可能漏标）、证明模态融合有效、跨数据集或多seed普遍优劣。图像中的暗区、遮挡和车辆纹理可作为进一步人工检查假设，不能用几何分类代替逐框审核。未根据测试结果调整任何模型阈值或参数。',
            '给ChatGPT：先解释原始一对一贪心匹配和GT占用，再沿190001逐框核对为什么是FP，区分IoU不足与重复候选；结合数量和比例讨论差异，再看分数≥0.9的错误。只提出可验证假设，不把启发式类别说成已证实根因。不主动进入融合训练或Day05。']
    with (BASE/'DAY04_ERROR_ANALYSIS_GPT_HANDOFF.md').open('x',encoding='utf-8') as f: f.write('\n\n'.join(lines)+'\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
    print('PASS: matching event parity, category conservation, 150 paired GT checks, unchanged cache SHA; no inference.',flush=True)


if __name__=='__main__': main()
