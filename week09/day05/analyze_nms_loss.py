"""Full validation NMS coverage-loss relationships, saved tensors only."""
import argparse
from collections import Counter,defaultdict
from contextlib import redirect_stdout,redirect_stderr
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import subprocess
import traceback
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'week09/day04'))
import torch
import torchvision
from torchvision.ops import boxes as ops
from full_val_analysis import validate_capture
from offline_full_gt import load_json,read_csv,save_json,write_csv,sha,Tee
from nms_relation_core import build_queue,relationship,model_summary,pair_summary,SCENES,THRESHOLD32,NEAR

SOURCE=ROOT/'week09/day04/outputs/full_val_stage_run01'

def box_fields(pos,prefix,r,cat,pre,keep_positions,gi,g32,g64):
    flat=int(pre['ids'][pos]);output_position=keep_positions.get(pos)
    return {prefix+'candidate_id':cat[flat]['candidate_id'],prefix+'flat_index':flat,
        prefix+'level_key':r['layer_keys'][int(r['nms_levels'][pos])],
        prefix+'xyxy':json.dumps(r['nms_boxes'][pos].tolist()),prefix+'score':float(r['nms_scores'][pos]),
        prefix+'gt_iou_float32_cpu':float(g32[gi,pos]),prefix+'gt_iou_float64_diagnostic':float(g64[gi,pos]),
        prefix+'nms_input_position_0based':pos,prefix+'nms_output_position_0based':output_position,
        prefix+'passes_final_budget':output_position is not None and output_position<len(r['final_boxes'])}

def analyze_capture(entry,events,source_gt):
    path=SOURCE/entry['relative_raw'];data=torch.load(path,map_location='cpu',weights_only=True)
    r,states,cat,gt=validate_capture(data)
    pre=states['score_filtered'];post=states['layer_nms'];b,s,l,keep=(r[k] for k in ('nms_boxes','nms_scores','nms_levels','nms_indices'))
    assert b.dtype==s.dtype==gt.dtype==torch.float32 and b.device.type=='cpu'
    assert bool((b[:,2:]>b[:,:2]).all()) and bool(torch.isfinite(b).all())
    g32=ops.box_iou(gt,b);g64=ops.box_iou(gt.double(),b.double())
    for gi in range(len(gt)):
        expected=source_gt[entry['model'],entry['stem'],gi]
        assert json.loads(expected['score_filtered']['gt_xyxy'])==gt[gi].tolist()
        for stage,values in (('score_filtered',g32[gi]),('layer_nms',g32[gi,keep])):
            old=expected[stage]
            assert (float(values.max()) if len(values) else 0.)==float(old['max_iou_float32_cpu'])
            assert int((values>=.75).sum())==int(old['count_iou_ge_075_float32_cpu'])
    assert len(gt)==len([k for k in source_gt if k[:2]==(entry['model'],entry['stem'])])
    shifted=b+(l.to(b)*(b.max()+1))[:,None]
    keep_positions={int(pos):k for k,pos in enumerate(keep.tolist())}
    retained=[]
    for k,j in enumerate(keep.tolist()):
        flat=int(pre['ids'][j])
        retained.append({'model':entry['model'],'stem':entry['stem'],'scene_id':events[0]['scene_id'],
            'candidate_id':cat[flat]['candidate_id'],'flat_index':flat,'level_key':r['layer_keys'][int(l[j])],
            'xyxy':json.dumps(b[j].tolist()),'score':float(s[j]),'nms_input_position_0based':j,
            'nms_output_position_0based':k,'passes_final_budget':k<len(r['final_boxes'])})
    audit=[];deleted=[];relations=[];diagnostic=[];overlaps=[];completed=[]
    for event in events:
        event=dict(event);gi=event['gt_index'];good=torch.where(g32[gi]>=.75)[0]
        assert len(good)>0 and len(good)==event['day04_pre_high_count'] and int((g32[gi,keep]>=.75).sum())==0
        assert not any(int(i) in keep_positions for i in good)
        mutual32=ops.box_iou(b[good],b[keep]);mutual64=ops.box_iou(b[good].double(),b[keep].double())
        shifted32=ops.box_iou(shifted[good],shifted[keep])
        event_deleted=[];event_related=[];event_audit=[];event_diagnostic=[];related_positions=set()
        for q,i in enumerate(good.tolist()):
            df=box_fields(i,'deleted_',r,cat,pre,keep_positions,gi,g32,g64)
            candidate_rows=[]
            for k,j in enumerate(keep.tolist()):
                rf=box_fields(j,'retained_',r,cat,pre,keep_positions,gi,g32,g64)
                flags=relationship(bool(l[i]==l[j]),float(s[i]),float(s[j]),float(mutual32[q,k]),float(mutual64[q,k]),float(shifted32[q,k]))
                other=[t for t in range(len(gt)) if t!=gi]
                maxother32=max((float(g32[t,j]) for t in other),default=None)
                maxother64=max((float(g64[t,j]) for t in other),default=None)
                row={key:event[key] for key in ('event_id','model','stem','scene_id','gt_index')}
                row.update(df);row.update(rf);row.update(flags)
                row.update(mutual_iou_float32_cpu=float(mutual32[q,k]),mutual_iou_float64_diagnostic=float(mutual64[q,k]),
                    mutual_iou_shifted_float32_diagnostic=float(shifted32[q,k]),nms_threshold=.7,threshold_float32=THRESHOLD32,
                    raw_float32_margin=float(mutual32[q,k])-THRESHOLD32,raw_float64_margin=float(mutual64[q,k])-.7,
                    shifted_float32_margin=float(shifted32[q,k])-THRESHOLD32,
                    retained_other_gt_max_iou_float32=maxother32,retained_other_gt_max_iou_float64=maxother64,
                    retained_max_other_gt_indices=json.dumps([t for t in other if float(g32[t,j])==maxother32]),
                    retained_other_gt_overlap_larger=maxother32 is not None and maxother32>float(g32[gi,j]),
                    retained_specified_gt_iou_worse=float(g32[gi,j])<float(g32[gi,i]),
                    evidence_label='possible_related_retained_not_actual_GPU_suppression')
                candidate_rows.append(row);event_audit.append(row)
                if flags['primary_possible_relation']:
                    event_related.append(row);related_positions.add(j)
                if flags['diagnostic_only_relation']:
                    event_diagnostic.append(row);related_positions.add(j)
            primary=[a for a in candidate_rows if a['primary_possible_relation']]
            diag=[a for a in candidate_rows if a['diagnostic_only_relation']]
            dr={key:event[key] for key in ('event_id','model','stem','scene_id','gt_index')}
            dr.update(df)
            dr.update(candidate_gt_key=df['deleted_candidate_id']+f"|G{gi}",primary_relation_count=len(primary),
                diagnostic_only_relation_count=len(diag),unexplained_primary=not bool(primary),
                GT_threshold_classification_disagreement=bool((g32[gi,i]>=.75)!=(g64[gi,i]>=.75)),
                primary_tie_relation_count=sum(a['score_tie'] for a in primary))
            event_deleted.append(dr)
        event.update(deleted_good_candidate_count=len(event_deleted),primary_relation_count=len(event_related),
            diagnostic_only_relation_count=len(event_diagnostic),has_primary_relation=bool(event_related),
            all_deleted_good_have_primary_relation=all(r['primary_relation_count']>0 for r in event_deleted),
            unexplained_deleted_good_count=sum(r['unexplained_primary'] for r in event_deleted),
            has_primary_score_tie=any(r['score_tie'] for r in event_related),
            has_eligible_score_tie=any(r['score_tie'] and r['eligible_same_layer_score'] for r in event_audit),
            near_nms_threshold=any(r['near_threshold_diagnostic_only'] for r in event_audit),
            relation_predicate_disagreement=any(r['relation_predicate_disagreement'] for r in event_audit),
            has_diagnostic_only_relation=bool(event_diagnostic),
            GT_float32_float64_coverage_disagreement=bool((g64[gi].max()>=.75)!=True or (g64[gi,keep].max()>=.75)!=False),
            any_related_retained_other_gt_overlap_larger=any(r['retained_other_gt_overlap_larger'] for r in event_related),
            nms_input_count=len(b),original_nms_output_count=len(keep),final_proposal_count=len(r['final_boxes']))
        event['numerically_sensitive']=(event['near_nms_threshold'] or event['relation_predicate_disagreement']
            or event['GT_float32_float64_coverage_disagreement'] or any(r['GT_threshold_classification_disagreement'] for r in event_deleted))
        for j in sorted(related_positions):
            flat=int(pre['ids'][j])
            for t in range(len(gt)):
                overlaps.append({'event_id':event['event_id'],'model':event['model'],'stem':event['stem'],'specified_gt_index':gi,
                    'retained_candidate_id':cat[flat]['candidate_id'],'retained_flat_index':flat,'compared_gt_index':t,
                    'candidate_xyxy':json.dumps(b[j].tolist()),'gt_xyxy':json.dumps(gt[t].tolist()),
                    'is_specified_gt':t==gi,'iou_float32_cpu':float(g32[t,j]),'iou_float64_diagnostic':float(g64[t,j]),
                    'semantic_assignment_proven':False})
        completed.append(event);deleted.extend(event_deleted);audit.extend(event_audit)
        relations.extend(event_related);diagnostic.extend(event_diagnostic)
    provenance={key:entry[key] for key in ('model','stem','relative_raw','raw_sha256')}
    provenance.update(events=len(events),raw_load_count=1,nms_input_count=len(b),original_nms_output_count=len(keep),
        final_count=len(r['final_boxes']),real_indices_verified=True,gt_count=len(gt))
    return completed,deleted,audit,relations,diagnostic,retained,overlaps,provenance

def execute(out):
    stage=read_csv(SOURCE/'per_gt_stage.csv');events,paired=build_queue(stage)
    if Counter(e['model'] for e in events)!=Counter({'IR':24,'Fusion':34}):raise ValueError('NMS loss event count differs')
    ledger=read_csv(SOURCE/'forward_ledger.csv');lookup={(e['model'],e['stem']):e for e in ledger}
    if len(lookup)!=len(ledger):raise ValueError('Duplicate capture ledger')
    source_gt=defaultdict(dict)
    for r in stage:source_gt[r['model'],r['stem'],int(r['gt_index'])][r['stage']]=r
    groups=defaultdict(list)
    for e in events:groups[e['model'],e['stem']].append(e)
    allparts=[[] for _ in range(8)]
    for key,queue in sorted(groups.items()):
        result=analyze_capture(lookup[key],queue,source_gt)
        for index,part in enumerate(result):
            if index==7:allparts[index].append(part)
            else:allparts[index].extend(part)
        print('CAPTURE_ANALYZED',key[0],key[1],'events',len(queue),'deleted_good_rows',len(result[1]),'primary_relations',len(result[3]),flush=True)
    events,deleted,audit,relations,diagnostic,inventory,overlaps,provenance=allparts
    assert len({(r['event_id'],r['deleted_candidate_id']) for r in deleted})==len(deleted)
    assert len({(r['event_id'],r['deleted_candidate_id'],r['retained_candidate_id']) for r in audit})==len(audit)
    assert len({(r['model'],r['stem'],r['candidate_id']) for r in inventory})==len(inventory)
    summaries=model_summary(events,deleted,relations,paired);paired_summary=pair_summary(paired)
    physical=defaultdict(list)
    for d in deleted:physical[d['deleted_candidate_id']].append(d)
    unique_deleted=[{'model':ds[0]['model'],'stem':ds[0]['stem'],'candidate_id':cid,'flat_index':ds[0]['deleted_flat_index'],
        'GT_indices':json.dumps(sorted({d['gt_index'] for d in ds})),'candidate_GT_row_count':len(ds)} for cid,ds in sorted(physical.items())]
    tables={'gt_events':events,'deleted_good_candidates':deleted,'unique_deleted_candidates':unique_deleted,
        'pair_audit':audit,'possible_related_retained':relations,'diagnostic_only_relations':diagnostic,
        'retained_inventory':inventory,'gt_overlap_audit':overlaps,'model_scene_summary':summaries,
        'all_gt_pairing':paired,'common_risk_pairing':[r for r in paired if r['both_precovered']],
        'paired_summary':paired_summary,'numerical_audit':[r for r in audit if r['near_threshold_diagnostic_only'] or r['relation_predicate_disagreement'] or (r['eligible_same_layer_score'] and r['score_tie'])]}
    fields={'diagnostic_only_relations':list(audit[0]),'numerical_audit':list(audit[0])}
    for name,rows in tables.items():write_csv(out/(name+'.csv'),rows,fields.get(name))
    save_json(out/'capture_provenance.json',provenance)
    info={'status':'passed','event_count':len(events),'distinct_GT_count':len({(r['stem'],r['gt_index']) for r in events}),
        'distinct_images':len({r['stem'] for r in events}),'capture_files_loaded_once':len(groups),
        'good_candidate_GT_rows':len(deleted),'unique_good_candidates':len(physical),'primary_relation_GT_rows':len(relations),
        'unique_primary_candidate_pairs':len({(r['deleted_candidate_id'],r['retained_candidate_id']) for r in relations}),
        'pair_audit_rows':len(audit),'diagnostic_only_relation_rows':len(diagnostic),
        'all_GT_paired':len(paired),'common_risk_GT':sum(r['both_precovered'] for r in paired),
        'actual_GPU_suppression_pairs_proven':0,'all_tables_rows':{n:len(rr) for n,rr in tables.items()}}
    save_json(out/'analysis_summary.json',info)
    report(out,info,summaries,paired_summary,events,deleted,relations)
    return info

def report(out,info,summary,paired,events,deleted,relations):
    lines=['# Week09 Day05：完整val NMS覆盖丢失的离线关系','',
        '只用Day04已有捕获和真实索引。没有连接服务器、加载checkpoint、执行模型前向或重新运行NMS；未训练、调参或运行test。','',
        '## 队列与计数单位','',
        f"队列为score_filtered已覆盖且layer_nms未覆盖，CPU float32 IoU≥0.75。IR24项、Fusion34项，共58个模型×GT事件，对应{info['distinct_GT_count']}个不同GT、{info['distinct_images']}张图、{info['capture_files_loaded_once']}份模型捕获。每份捕获只读取一次，多个事件共用它，不按GT重复加载。",
        f"被删好框表有{info['good_candidate_GT_rows']}个候选×GT行，去重后{info['unique_good_candidates']}个物理候选；主关系{info['primary_relation_GT_rows']}条，物理候选对去重后{info['unique_primary_candidate_pairs']}对。它们不是GT数。",
        '','## 关系与数值口径','',
        '原NMS全部输出作为保留集合，包含后来被150预算截掉的框。主关系条件为同层、原分数更高或完全同分、原坐标CPU float32相互IoU严格>0.7。保留集合由真实索引决定，不重跑CPU NMS。',
        'float64几何诊断和按层偏移后float32诊断单列。主口径不满足、仅诊断满足的行在diagnostic_only_relations.csv，不并入主关系数。1e-6仅标记阈值附近，不放宽0.7或0.75。跨层的偏移导致重叠消失是设计行为，不计为数值敏感。',
        '数值敏感GT定义：事件覆盖或被删好框的f32/f64达标判定不同，或存在同层且分数不低的比较在NMS阈值附近／三种几何口径判定分歧。同分主关系与所有同层分数合格比较的同分分别记录。',
        '所有关系统一称可能相关的保留框；缺少原GPU逐次抑制记录，没有实际抑制配对证据。按层偏移只是本地重建的数值诊断，不是原CUDA内部事件。','',
        '## 模型与场景','',
        '|模型|场景|NMS前覆盖GT|丢失GT|条件丢失率|有主关系GT|每个好框都有主关系GT|未解释GT|主关系同分GT|数值敏感GT|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['model']}|{r['scene_id']}|{r['nms_precovered_gt']}|{r['nms_lost_gt']}|{r['conditional_loss_rate']}|{r['gt_with_primary_relation']}|{r['gt_every_deleted_good_has_primary_relation']}|{r['gt_partly_or_fully_unexplained_primary']}|{r['gt_with_primary_score_tie']}|{r['gt_numerically_sensitive']}|")
    lines+=['','未解释GT表示至少一个被删达标框没有主关系；不等于完全没有任何可能关系。即使每个好框都有关系，也不能视为实际抑制事件已证实。','',
        '## 同GT配对：不同风险分母与共同队列','',
        '|范围|场景|GT数|IR前覆盖|Fusion前覆盖|IR丢失|Fusion丢失|共同丢失|仅IR丢失|仅Fusion丢失|均未丢失|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in paired:lines.append(f"|{r['cohort']}|{r['scene_id']}|{r['GT_denominator']}|{r['IR_precovered_gt']}|{r['Fusion_precovered_gt']}|{r['IR_lost_gt']}|{r['Fusion_lost_gt']}|{r['both_lost']}|{r['IR_only_lost']}|{r['Fusion_only_lost']}|{r['neither_lost']}|")
    common=next(r for r in paired if r['scene_id']=='ALL' and r['cohort']=='both_precovered')
    lines+=['',f"共同风险队列{common['GT_denominator']}个GT：IR丢失{common['IR_lost_gt']}，Fusion丢失{common['Fusion_lost_gt']}；同一分母下差值{common['Fusion_minus_IR_lost_gt']}个GT。仅IR丢失{common['IR_only_lost']}、仅Fusion丢失{common['Fusion_only_lost']}。差异是否仍在应看这个配对队列，而不是直接把24/326与34/321当作公平同GT配对。",
        '共同队列也只是条件性描述，未控制NMS前高IoU候选数、分数、尺度或其他几何差异，不是因果效应。all_gt_pairing.csv保留全部416个GT的风险状态；not_at_risk不能与仍然覆盖混为一谈。','',
        '## 选择条件带来的必然现象','',
        '队列本来就要求NMS后所有保留框对指定GT IoU<0.75，而被删好框≥0.75。因此相关保留框对指定GT定位更差是队列定义自然带来的；关系又筛选更高分或同分，高比例分数／定位冲突不能当作所有候选的打分质量结论或独立根因证明。',
        '原NMS丢失索引是操作事实；同层、高分和重叠只给出可能关系；实际哪个框抑制了哪个框仍需直接事件或额外严格归因证据。本轮不实现独立贪心NMS归因。','',
        '## 其他GT重叠','']
    for model in ('IR','Fusion'):
        r=next(r for r in summary if r['model']==model and r['scene_id']=='ALL')
        lines.append(f"- {model}：{r['gt_with_related_retained_other_gt_overlap_larger']}/{r['nms_lost_gt']}个事件存在主关系保留框与其他GT重叠大于指定GT。")
    lines+=['','完整GT重叠在gt_overlap_audit.csv；最大重叠对象只是几何描述，不能认定语义归属或证明邻近目标混淆。','',
        '## 哪些信息有助于下一步','',
        '有用的是共同风险GT上的非对称丢失、场景分布、找不到主关系的候选、同分与数值敏感、以及相关框对其他GT的几何重叠；“有重叠高分保留框、且它对本GT更差”的高比例本身多由规则与队列约束产生。',
        '下一项最小可验证假设：在共同风险队列的模型非对称丢失GT中，失败模型在NMS前的达标候选支持数／层分布与分数排序处境可能较弱。未来若获准，只用现有捕获对这些同GT进行配对描述，并先分场景检查，保留未解释及数值敏感项；不先调阈值、修改融合模块或启动训练。本轮不执行该后续分析。','',
        '## 核验与交付','',
        '输入SHA、原索引、源码／设备／dtype与历史文件前后保护见run_metadata.json；逐候选与GT关系的独立面积公式核验单独保存。主关系、诊断关系、数值审计及全部GT配对均保留；无关系项不猜测补齐。完整运行命令与退出码见run.log、execution_status.json。完成验收后停止，等待反馈。']
    with (out/'day05_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(HERE):raise ValueError('Day05 output only')
    out.mkdir(parents=True,exist_ok=False)
    protected=[q for r in (ROOT/'week09/day01',ROOT/'week09/day02',ROOT/'week09/day03',ROOT/'week09/day04') for q in r.rglob('*') if q.is_file() and '__pycache__' not in q.parts]
    before={str(q):sha(q) for q in protected}
    meta={'scope':'full_val_NMS_loss_events_only','status':'running','command':subprocess.list2cmdline(sys.orig_argv),
        'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'device':'cpu','primary_dtype':'float32','diagnostic_dtype':'float64',
        'environment':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'platform':platform.platform()},
        'source_sha256':{str(q):sha(q) for q in HERE.glob('*.py')},'boxes_python_source_sha256':sha(inspect.getfile(ops)),
        'historical_sha256_before':before,'server_connected':False,'checkpoint_loaded':False,'model_forward_performed':False,'NMS_executed':False}
    code=1
    with (out/'run.log').open('x',encoding='utf-8') as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
       try:
        print('COMMAND',meta['command'],flush=True);torch.set_num_threads(2)
        manifest=load_json(SOURCE/'artifact_manifest.json')
        if not all(sha(SOURCE/n)==h for n,h in manifest.items()):raise ValueError('Day04 artifact SHA mismatch')
        meta['input_manifest_sha256']=sha(SOURCE/'artifact_manifest.json');meta['input_artifact_files_verified']=len(manifest)
        with patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forward forbidden')), \
             patch.object(ops,'nms',side_effect=RuntimeError('NMS forbidden')), \
             patch.object(ops,'batched_nms',side_effect=RuntimeError('NMS forbidden')):
            info=execute(out)
        meta['status']='completed';meta['analysis_summary']=info;code=0
        print('NMS_RELATION_ANALYSIS_COMPLETED',json.dumps(info),flush=True)
       except BaseException:
        meta['status']='failed';meta['error']=traceback.format_exc();traceback.print_exc()
       finally:
        meta['historical_sha256_after']={str(q):sha(q) for q in protected}
        meta['historical_files_unchanged']=before==meta['historical_sha256_after']
        if not meta['historical_files_unchanged']:code=1;meta['status']='failed'
        meta['exit_code']=code
        save_json(out/'run_metadata.json',meta);save_json(out/'execution_status.json',{'status':meta['status'],'exit_code':code})
        print('OFFLINE_EXIT',code,'historical_files_unchanged',meta['historical_files_unchanged'],flush=True)
    return code
if __name__=='__main__':sys.exit(main())
