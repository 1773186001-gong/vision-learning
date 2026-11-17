"""Shared offline validation/merge for GT-independent and accepted legacy captures."""
from pathlib import Path
import json
import torch
from torchvision.ops import boxes as ops
from coverage_statistics import STAGES,measure_image,summarize
from offline_full_gt import ROOT,D1,sha,read_csv,load_json,save_json,write_csv,exact

def validate_capture(data):
    if len(data['records'])!=1 or len(data['stages'])!=1:raise ValueError('Single capture required')
    r,s=data['records'][0],data['stages'][0];gt=data['observed']['transform_gt_all']
    if tuple(s)!=STAGES or r['layer_keys']!=('0','1','pool') or r['image_size']!=(640,640):raise ValueError('Stage/keys/geometry mismatch')
    c=r['config']
    if (c['pre_nms'],c['post_nms'],c['score_thresh'],c['nms_thresh'],c['min_size'])!=(150,150,.05,.7,.001):raise ValueError('Fixed config differs')
    if 'gt_all' in r:exact(r['gt_all'],gt)
    ids=torch.arange(len(r['boxes']));levels=torch.repeat_interleave(torch.arange(3),torch.tensor(r['counts']))
    exact(s['decoded']['ids'],ids);exact(s['decoded']['boxes'],r['boxes'])
    cat=data['catalogues'][0]
    if len(cat)!=len(ids) or any(e['flat_index']!=i or e['candidate_id']!=f"{r['prefix']}/{i}" or e['level_key']!=r['layer_keys'][int(levels[i])] for i,e in enumerate(cat)):raise ValueError('ID catalogue differs')
    top=r['top_indices']
    if top.dtype!=torch.int64 or len(top.unique())!=len(top) or bool(((top<0)|(top>=len(ids))).any()):raise ValueError('Invalid actual Top-N')
    exact(s['pre_nms_topn']['ids'],ids[top]);exact(s['pre_nms_topn']['boxes'],r['boxes'][top])
    if any(int((levels[top]==i).sum())!=min(150,n) for i,n in enumerate(r['counts'])):raise ValueError('Pre budget not per layer')
    exact(s['clipped']['ids'],ids[top]);exact(s['clipped']['boxes'],ops.clip_boxes_to_image(r['boxes'][top],r['image_size']))
    exact(s['clipped']['scores'],s['pre_nms_topn']['scores'])
    def select(old,new,index):
        if index.dtype!=torch.int64 or len(index.unique())!=len(index) or bool(((index<0)|(index>=len(old['ids']))).any()):raise ValueError('Invalid original index')
        for field in ('ids','boxes','scores'):exact(new[field],old[field][index])
    select(s['clipped'],s['small_box_filtered'],ops.remove_small_boxes(s['clipped']['boxes'],c['min_size']))
    prev=s['small_box_filtered'];select(prev,s['score_filtered'],torch.where(prev['scores']>=c['score_thresh'])[0])
    for f in ('boxes','scores'):exact(s['score_filtered'][f],r['nms_'+f])
    exact(levels[s['score_filtered']['ids']],r['nms_levels'])
    select(s['score_filtered'],s['layer_nms'],r['nms_indices'])
    select(s['layer_nms'],s['post_nms_topn'],torch.arange(min(150,len(s['layer_nms']['ids']))))
    exact(s['post_nms_topn']['boxes'],r['final_boxes']);exact(s['post_nms_topn']['scores'],r['final_scores'])
    exact(data['observed']['rpn_forward_boxes'],r['final_boxes'])
    if any(data['observed'][key]!=1 for key in ('model_forward_calls','transform_calls','rpn_calls')):raise ValueError('Original call count differs')
    return r,s,cat,gt

def compare_item(data,model,mr,index,historical):
    r,s,cat,gt=validate_capture(data)
    if len(gt)!=int(mr['person_boxes']):raise ValueError('GT count differs from manifest')
    rows,image=measure_image(model,mr['stem'],mr['scene_id'],index,gt,s,cat,r['image_size'])
    comparisons=[]
    for gi,g in enumerate(gt):
        old=historical[model,mr['stem'],gi]
        if int(old['image_index'])!=index or old['scene_id']!=mr['scene_id']:raise ValueError('Day01 GT identity differs')
        exact(g,torch.tensor([float(old[k]) for k in ('xmin','ymin','xmax','ymax')]))
        final=next(v for v in rows if v['gt_index']==gi and v['stage']=='post_nms_topn')
        delta=final['max_iou_float32_cpu']-float(old['max_iou_top300'])
        coverage_equal=final['covered_075_float32_cpu']==bool(int(old['covered_top300_iou0.75']))
        comparisons.append({'model':model,'stem':mr['stem'],'gt_index':gi,'day01_max_iou':float(old['max_iou_top300']),
            'pilot_max_iou':final['max_iou_float32_cpu'],'actual_difference':delta,'atol':1e-6,'rtol':0.,'coverage_equal':coverage_equal})
        # Persist caller's original capture before this check is reached.
        if abs(delta)>1e-6 or not coverage_equal or final['proposal_count']!=int(old['proposal_count']):
            raise ValueError('Day01 disagreement '+json.dumps(comparisons[-1])+'; no rerun/tolerance relaxation')
    return rows,image,comparisons

def merge_full(out,entries):
    out=Path(out);rows=[];images=[];comparisons=[];captures=[]
    val=[r for r in read_csv(ROOT/'data/LLVIP_1000/manifest.csv') if r['split']=='val']
    historic_rows=read_csv(D1/'rpn_per_gt.csv');history={(r['model'],r['stem'],int(r['gt_index'])):r for r in historic_rows}
    if len(history)!=len(historic_rows):raise ValueError('Duplicate Day01 keys')
    if len(entries)!=len(val)*2 or len({(e['model'],e['stem']) for e in entries})!=len(entries):raise ValueError('Full item count/uniqueness')
    for e in entries:
        path=out/e['relative_raw'];data=torch.load(path,map_location='cpu',weights_only=True)
        if sha(path)!=e['raw_sha256']:raise ValueError('Raw checksum mismatch')
        rr,im,cc=compare_item(data,e['model'],val[e['image_index']],e['image_index'],history)
        rows.extend(rr);images.append(im);comparisons.extend(cc)
        captures.append(dict(e,raw_path=str(path),all_gt_count=im['gt_count']))
    if {(r['model'],r['stem'],r['gt_index']) for r in rows}!=set(history):raise ValueError('Day01 GT key set differs')
    stats=summarize(rows,images);reference=load_json(D1/'rpn_metrics.json')
    actual={}
    for model in ('IR','Fusion'):
        ims=[r for r in images if r['model']==model]
        final=next(r for r in stats['stage_summary'] if (r['scope'],r['model'],r['stage'],r['iou_convention'])==('overall',model,'post_nms_topn','float32_cpu'))
        expected=reference['models'][model]
        actual[model]={'images':len(ims),'valid_gt':sum(r['gt_count'] for r in ims),'covered_075':final['covered_gt']}
        if (len(ims),actual[model]['valid_gt'],final['covered_gt'])!=(expected['processed_images'],expected['total_valid_gt'],expected['top_k']['300']['hits']['0.75']):raise ValueError('Full reference totals differ')
    write_csv(out/'per_gt_stage.csv',rows);write_csv(out/'per_image.csv',images)
    for key,ss in stats.items():write_csv(out/(key+'.csv'),ss)
    ts=stats['gt_coverage_sequences'];write_csv(out/'never_covered_gt.csv',[r for r in ts if r['never_covered']],list(ts[0]))
    write_csv(out/'day01_per_gt_comparison.csv',comparisons)
    save_json(out/'reuse_evidence.json',{'status':'passed_full_validation','captures':captures,'full_val_source':'manifest+Day01 GT keys'})
    verification={'status':'passed','scope':'full_validation','actual':actual,'day01_comparisons':comparisons,
        'gt_model_items':len(comparisons),'per_gt_stage_rows':len(rows),'atol':1e-6,'rtol':0.,
        'max_abs_day01_difference':max(abs(r['actual_difference']) for r in comparisons),'new_forward_calls':sum(e['new_forward_calls'] for e in entries)}
    save_json(out/'verification.json',verification)
    report_full(out,stats,verification)
    return verification

def report_full(out,stats,v):
    primary=lambda rows:[r for r in rows if r.get('iou_convention')=='float32_cpu']
    net=primary(stats['net_difference']);overall=[r for r in net if r['scope']=='overall']
    lines=['# Week09 Day04：完整validation RPN逐阶段覆盖统计','',
        '固定原val、原best、原预处理及pre/post150、score0.05、NMS0.7。六图12份原捕获复用，剩余144图IR/Fusion各一次，共288次新增原eval前向；没有训练、test、调参或NMS抑制者配对。','',
        '主口径为CPU float32 IoU≥0.75；float64诊断另列CSV。覆盖以GT为单位，整体不平均逐图或逐场景比例。','',
        '## 实际范围与终点','',str(v['actual']),
        f"逐GT对照Day01共{v['gt_model_items']}项，最大绝对IoU差={v['max_abs_day01_difference']:.16g}，atol=1e-6、rtol=0；覆盖判定不加容差。",'',
        '## 全部阶段','', '|阶段|IR覆盖|Fusion覆盖|IR−Fusion净差|净差变化|','|---|---:|---:|---:|---:|']
    for r in overall:lines.append(f"|{r['stage']}|{r['IR_covered_gt']}/{r['valid_gt']}|{r['Fusion_covered_gt']}/{r['valid_gt']}|{r['IR_minus_Fusion']}|{r['net_change_from_previous']}|")
    lines += ['',f"解码时IR−Fusion覆盖GT净差为{overall[0]['IR_minus_Fusion']}；最终净差为{overall[-1]['IR_minus_Fusion']}。这描述达标覆盖差距，不直接归因为特征或回归。",'',
        '净差变化=(Fusion丢失−IR丢失)+(IR新增覆盖−Fusion新增覆盖)，只是算术分解，不是因果贡献。净差相同也可能有双方丢失相互抵消。','',
        '## 丢失、出现与部分好框流失','', '|模型|阶段|丢失GT|前阶已覆盖GT|条件丢失率|新增覆盖GT|部分好框减少仍覆盖GT|','|---|---|---:|---:|---:|---:|---:|']
    for r in primary(stats['stage_summary']):
        if r['scope']=='overall' and r['stage']!='decoded':lines.append(f"|{r['model']}|{r['stage']}|{r['coverage_lost_gt']}|{r['previous_covered_gt']}|{r['lost_rate_previous_covered']}|{r['coverage_gained_gt']}|{r['partial_high_loss_coverage_kept_gt']}|")
    lines += ['','## 始终未覆盖与配对','']
    for r in primary(stats['never_covered_summary']):
        if r['scope']=='overall':lines.append(f"- {r['model']}七阶段始终未覆盖：{r['never_covered_gt']}/{r['valid_gt']}。")
    for r in primary(stats['paired_summary']):
        if r['scope']=='overall':lines.append(f"- {r['stage']}：两者覆盖{r['both']}、仅IR{r['IR_only']}、仅Fusion{r['Fusion_only']}、均无{r['neither']}；GT分母{r['valid_gt']}。")
    lines += ['','## 场景如何组成总体净差','', '|场景|GT分母|解码净差|最终净差|解码后净差变化|','|---|---:|---:|---:|---:|']
    scenes=sorted({r['scene_id'] for r in net if r['scope']=='scene'})
    for scene in scenes:
        a=next(r for r in net if r['scene_id']==scene and r['stage']=='decoded');b=next(r for r in net if r['scene_id']==scene and r['stage']=='post_nms_topn')
        lines.append(f"|{scene}|{a['valid_gt']}|{a['IR_minus_Fusion']}|{b['IR_minus_Fusion']}|{b['IR_minus_Fusion']-a['IR_minus_Fusion']}|")
    lines += ['','各场景净差相加组成总体净差，是GT计数的描述性分解。场景分母不同，不直接平均比例，也不能把场景净差称为某个模块的因果贡献。场景逐阶段变化见net_difference.csv。','',
        '## 证据与限制','',
        '原Top-N/NMS索引传播稳定候选ID；原方法调用并原样返回。新捕获在原设备精确重放NMS输入、最终坐标／顺序／分数，并与实际RPN输出一致；模型state、配置、权重和受保护文件前后校验及hooks恢复见逐项原捕获和run_metadata.json。',
        '原NMS全部输出先保存，再计算最终150截断。裁剪后首次达标及恢复保留完整序列，不能只看decoded。空GT图单独记录，不伪造GT。',
        '原操作事实与离线覆盖统计不等于根因；当前未做抑制者配对、独立贪心归因、融合模块或RoI因果分析。RPN覆盖不是最终检测Recall或AP75。',
        '完整前向账本见forward_ledger.csv，六图来源见reuse_evidence.json，原始捕获逐项落盘。下载SHA和本地独立重算为单独检查记录。完成本轮后停止，等待反馈。']
    with (out/'full_val_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
