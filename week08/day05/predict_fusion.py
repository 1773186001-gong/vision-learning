"""Offline analysis and visualization of saved test predictions only."""
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from torchvision.transforms.functional import to_pil_image
from fusion_common import *
from error_trace import trace


def render_and_report(root,out,history,best_epoch,scores):
    verify_package()
    saved=torch.load(out/'test_predictions.pth',map_location='cpu',weights_only=False)
    assert ir.metrics(saved['predictions'],saved['truths'])==scores
    refs={}
    for mode in ('IR','RGB'):
        folder=Path(__file__).parent/'references'/mode
        # Reference files are trusted package copies checked by package_manifest.json.
        refs[mode]=torch.load(folder/'test_predictions.pth',map_location='cpu',weights_only=False)
        assert refs[mode]['stems']==saved['stems']
        assert all(torch.equal(a,b) for a,b in zip(refs[mode]['truths'],saved['truths']))
    caches={**refs,'Fusion':saved}; summaries={}; traces={}; counts={}
    for mode,cache in caches.items():
        traces[mode]=[trace(p,g) for p,g in zip(cache['predictions'],cache['truths'])]
        metric=ir.metrics(cache['predictions'],cache['truths'])
        types=Counter(r['fp_type'] for rows in traces[mode] for r in rows if r['status']=='FP')
        assert sum(types.values())==metric['FP']
        summaries[mode]=dict(metrics=metric,fp_categories={t:dict(count=types[t],fraction=types[t]/max(metric['FP'],1)) for t in ('duplicate','localization','low_overlap','other')})
        if mode in refs:
            reference=Path(__file__).parent/'references'/mode
            summaries[mode]['experiment']=json.loads((reference/'completed.json').read_text(encoding='utf-8'))
            summaries[mode]['model_size_bytes']={n:json.loads((reference/(n+'.metadata.json')).read_text())['bytes'] for n in ('best.pth','last.pth')}
            summaries[mode]['parameter_counts']=json.loads((reference/'parameter_counts.json').read_text())
        counts[mode]=[dict(TP=sum(r['status']=='TP' for r in rows),FP=sum(r['status']=='FP' for r in rows),FN=len(cache['truths'][i])-sum(r['status']=='TP' for r in rows)) for i,rows in enumerate(traces[mode])]
    save(out/'three_model_comparison.json',summaries)
    save(out/'fusion_error_details.json',dict(stems=saved['stems'],truths=[g.tolist() for g in saved['truths']],predictions=traces['Fusion']))
    selected={}
    for cat in ('success','miss','false_positive'):
        indices=[i for i,c in enumerate(counts['Fusion']) if (c['TP']>0 and c['FP']==c['FN']==0 if cat=='success' else c['FN']>0 if cat=='miss' else c['FP']>0)]
        selected[cat]=indices[:3]
    ds=loader(root,'test').dataset
    folder=out/'prediction_examples'; folder.mkdir()
    colors={'TP':'blue','duplicate':'orange','localization':'purple','low_overlap':'red','other':'gray'}
    def draw(ax,image,mode,i):
        ax.imshow(to_pil_image(image))
        cache=caches[mode]; matched={r['matched_gt'] for r in traces[mode][i] if r['status']=='TP'}
        for j,b in enumerate(cache['truths'][i].tolist()):
            color='green' if j in matched else 'cyan'
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=2))
        for p in traces[mode][i]:
            if not p['selected']: continue
            b=p['box']; color=colors['TP' if p['status']=='TP' else p['fp_type']]
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.4))
            ax.text(min(b[0],470),max(b[1]-3,8),f"person P{p['prediction_id']} {p['score']:.2f}",color=color,fontsize=6,bbox=dict(facecolor='white',alpha=.7,pad=0))
        ax.set_title(mode+' '+saved['stems'][i]+' '+str(counts[mode][i]),fontsize=9); ax.set_xlim(0,640); ax.set_ylim(640,0)
    legend=[Patch(color=c,label=l) for l,c in {'GT matched':'green','GT missed (FN)':'cyan',**colors}.items()]
    for cat,indices in selected.items():
        for i in indices:
            image,t=ds[i]; assert torch.equal(t['boxes'],saved['truths'][i])
            fig,ax=plt.subplots(figsize=(8,9)); draw(ax,image['visible'],'Fusion',i); fig.legend(handles=legend,loc='lower center',ncol=4,fontsize=7)
            fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(folder/f'{cat}_{saved["stems"][i]}.png',dpi=160); plt.close(fig)
    for stem in ('190001','190040','190474','190435','190593'):
        i=saved['stems'].index(stem); image,t=ds[i]; assert torch.equal(t['boxes'],saved['truths'][i])
        fig,axes=plt.subplots(1,3,figsize=(21,8))
        for ax,mode in zip(axes,('IR','RGB','Fusion')): draw(ax,image['infrared' if mode=='IR' else 'visible'],mode,i)
        fig.legend(handles=legend,loc='lower center',ncol=7); fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(folder/f'comparison_{stem}.png',dpi=160); plt.close(fig)
    save(folder/'selection.json',dict(selected={k:[saved['stems'][i] for i in v] for k,v in selected.items()},shortfall={k:3-len(v) for k,v in selected.items()}))
    meta=json.loads((out/'model_metadata.json').read_text(encoding='utf-8'))
    summaries['Fusion'].update(best_epoch=best_epoch,total_parameters=meta['total_parameters'],trainable_parameters=meta['trainable_parameters'],
        epoch_seconds_sum=sum(r['epoch_seconds'] for r in history),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in history),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in history),checkpoint_bytes={n:(out/n).stat().st_size for n in ('best.pth','last.pth')})
    # Update the already generated comparison before reporting, within this new run only.
    ir.save_json(out/'three_model_comparison.json',summaries)
    report=['# Day05 双分支特征融合真实训练交接','仅在10轮、选模、一次test及缓存分析成功后生成；进程最终退出码另见exit_status_*.json。',
            '架构：RGB MobileNetV3+FPN 与独立IR MobileNetV3+FPN → 对应实际尺度Concat+1×1Conv → 共享RPN/RoI。两个分支来自相同官方COCO权重，检测头兼容部分复用，最终person预测头新初始化。融合卷积初始化为[0.5I,0.5I]、bias=0；不读取旧IR/RGB微调权重。',
            '输入为明确的paired字典；仅在共享几何transform时内部临时打包，进入双分支前拆分；不做六通道卷积。两份ImageNet mean/std，各路IR复制三通道，640 Letterbox，train同步flip=.5，val/test=0。',
            '```json\n'+(out/'config.json').read_text(encoding='utf-8')+'\n```',
            '实际结构、参数来源与冻结状态详见model_metadata.json；参数量与可训练参数量：'+str((meta['total_parameters'],meta['trainable_parameters'])),
            '```json\n'+json.dumps(meta['architecture'],indent=2)+'\n```',
            'GPU冒烟（正/空样本四项Loss、更新、评估forward、完整state恢复）：\n```json\n'+(out/'smoke_test.json').read_text(encoding='utf-8')+'\n```',
            '|Epoch|total|classifier|box|objectness|rpn_box|lr|val P|val R|val AP50|val mAP|秒|peak allocated|peak reserved|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    keys=['total_loss',*ir.LOSS_KEYS,'lr','precision','recall','AP50','mAP50_95','epoch_seconds','peak_allocated_bytes','peak_reserved_bytes']
    report[-1]+='\n'+'\n'.join('|'+str(r['epoch'])+'|'+'|'.join(str(r[k]) for k in keys)+'|' for r in history)
    report += [f'最佳Epoch={best_epoch}，依据val mAP50:95，不用test选模；best/last大小(bytes)='+str({n:(out/n).stat().st_size for n in ('best.pth','last.pth')}),
               '三模型实测对照与四类FP（历史结果重新由可信缓存复核）：\n```json\n'+json.dumps(summaries,ensure_ascii=False,indent=2)+'\n```',
               '单person 101点自定义AP，IoU=.50:.05:.95；P/R为score≥.5、IoU≥.5，稳定排序、先排除已用GT再最大IoU；不是完整官方COCO API。错误分类复用Day04 trace；low_overlap不证明背景根因。',
               '训练+验证秒数='+str(sum(r['epoch_seconds'] for r in history))+'；最大allocated='+str(max(r['peak_allocated_bytes'] for r in history))+'；最大reserved='+str(max(r['peak_reserved_bytes'] for r in history)),
               '对照历史配置/最佳轮数/耗时见references/IR、RGB的config和completed；模型大小见各参考best/last.metadata.json。历史参数量未存时不编造，应通过独立结构计数补充。',
               '![loss](loss_curve.png)\n\n![mAP](mAP_curve.png)',
               '案例不足3张时记录shortfall，不制造案例；对照图为5组，见prediction_examples/comparison_*.png。',
               '同数据、训练超参数与评估协议的探索性对照；容量与计算量不同，不能把差异全部归因于融合。不根据test改参。单seed，测试已用于研究启发，不是全新盲测。续训保存optimizer/RNG/history，但未经中断恢复实测不能声称已验证。',
               '当前服务器结果目录：'+str(out)+'。本地取回与完整性核验尚待完成，原始结果不覆盖。请ChatGPT依次解释双分支/尺度检查、平均初始化、四项Loss、最佳轮次和三模型FP组成，再讨论可验证假设；不自动追加其他融合或注意力实验。']
    with (out/'DAY05_FUSION_TRAINING_GPT_HANDOFF.md').open('x',encoding='utf-8') as f: f.write('\n\n'.join(report))
