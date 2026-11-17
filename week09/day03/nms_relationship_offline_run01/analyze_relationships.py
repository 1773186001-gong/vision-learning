"""Fixed six deleted candidates; saved-capture geometry only. No NMS or model execution."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
import torchvision
from torchvision.ops import boxes as box_ops

HERE=Path(__file__).resolve().parent
DAY03=HERE.parent
CAPTURES=DAY03/"outputs/remaining_five_merged_run01"
CASES=(("030269",1,"Fusion","reused_030269_G1",(2201,)),
       ("130126",3,"IR","new_130126_G3",(3216,3516,3521,9216,9516)))
NEAR_WINDOW=1e-6  # Audit flag only; never added to the NMS or GT coverage threshold.

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def snapshot(paths):return {str(p.resolve()):sha(p) for p in sorted(paths)}

def save_json(path,value):
    with path.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def write_csv(path,rows,fields):
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def exact(a,b):
    if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise AssertionError("Saved tensor mismatch")

def scalar_iou(a,b):
    """Independent Python binary64 area formula, only for numerical cross-check."""
    iw=max(0.,min(a[2],b[2])-max(a[0],b[0]));ih=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter=iw*ih
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
    return inter/union if union>0 else 0.

def describe_box(stem,gi,model,role,flat,pos,boxes,scores,levels,cat,r,gt_iou32,gt_iou64,keep_position):
    output_pos=keep_position.get(pos)
    return {"stem":stem,"gt_index":gi,"model":model,"role":role,
        "candidate_id":cat[flat]["candidate_id"],"flat_index":flat,
        "level_key":r["layer_keys"][int(levels[pos])],"nms_input_position_0based":pos,
        "xyxy":json.dumps(boxes[pos].tolist()),"objectness":float(scores[pos]),
        "specified_gt_iou_float32_cpu":float(gt_iou32[pos,gi]),
        "specified_gt_iou_float64_cpu_diagnostic":float(gt_iou64[pos,gi]),
        "nms_output_position_0based":output_pos,
        "passes_final_budget":output_pos is not None and output_pos<len(r["final_boxes"])}

def analyze_case(stem,gi,model,sub,wanted):
    path=CAPTURES/sub/f"{model}_raw_capture.pth"
    data=torch.load(path,map_location="cpu",weights_only=True)  # Only whitelisted raw capture, never a checkpoint.
    assert len(data["records"])==len(data["stages"])==1
    r,stages=data["records"][0],data["stages"][0]
    before,after,final=(stages[k] for k in ("score_filtered","layer_nms","post_nms_topn"))
    boxes,scores,levels,keep=r["nms_boxes"],r["nms_scores"],r["nms_levels"],r["nms_indices"]
    assert boxes.device.type=="cpu" and boxes.dtype==scores.dtype==torch.float32
    assert keep.dtype==levels.dtype==torch.int64 and len(torch.unique(keep))==len(keep)
    assert bool(((keep>=0)&(keep<len(boxes))).all())
    assert bool(torch.isfinite(boxes).all()) and bool((boxes[:,2:]>boxes[:,:2]).all())
    assert r["layer_keys"]==("0","1","pool") and r["image_size"]==(640,640)
    assert r["config"]["nms_thresh"]==.7 and r["config"]["post_nms"]==150
    exact(before["boxes"],boxes);exact(before["scores"],scores)
    for key in ("ids","boxes","scores"):exact(after[key],before[key][keep])
    for key in ("ids","boxes","scores"):exact(final[key],after[key][:150])
    exact(r["final_boxes"],final["boxes"]);exact(r["final_scores"],final["scores"])
    exact(data["observed"]["rpn_forward_boxes"],r["final_boxes"])
    cat=data["catalogues"][0]
    mapped_levels=torch.repeat_interleave(torch.arange(3),torch.tensor(r["counts"]))
    exact(levels,mapped_levels[before["ids"]])
    for pos,flat in enumerate(before["ids"].tolist()):
        assert cat[flat]["flat_index"]==flat and cat[flat]["candidate_id"]==f"{r['prefix']}/{flat}"
        assert cat[flat]["level_key"]==r["layer_keys"][int(levels[pos])]
    all_gt=data["observed"]["transform_gt_all"]
    assert all_gt.dtype==torch.float32 and len(all_gt)>gi
    exact(all_gt[gi],r["gt"]);exact(data["observed"]["selected_gt"],r["gt"])
    assert bool(torch.isfinite(all_gt).all()) and bool((all_gt[:,2:]>all_gt[:,:2]).all())
    gt32=box_ops.box_iou(boxes,all_gt);gt64=box_ops.box_iou(boxes.double(),all_gt.double())
    pair32=box_ops.box_iou(boxes,boxes[keep]);pair64=box_ops.box_iou(boxes.double(),boxes[keep].double())
    # Installed same-version Python implementation uses this coordinate trick for these small CUDA inputs.
    # This CPU reconstruction is a numerical sensitivity diagnostic, not captured CUDA kernel inputs/events.
    offset=levels.to(boxes)*(boxes.max()+torch.tensor(1,dtype=boxes.dtype))
    shifted=boxes+offset[:,None]
    shifted32=box_ops.box_iou(shifted,shifted[keep])
    threshold=r["config"]["nms_thresh"];threshold32=float(boxes.new_tensor(threshold))
    keep_position={int(pos):k for k,pos in enumerate(keep.tolist())}
    positions={int(flat):pos for pos,flat in enumerate(before["ids"].tolist())}
    assert len(positions)==len(before["ids"])
    inventory=[]
    for pos in keep.tolist():
        inventory.append(describe_box(stem,gi,model,"original_nms_retained",int(before["ids"][pos]),pos,
            boxes,scores,levels,cat,r,gt32,gt64,keep_position))
    audit=[];summary=[];related=[];overlaps=[];entities={};max_error=0.
    for ident in wanted:
        pos=positions[ident]
        assert pos not in keep_position and data["first_removed"][0][ident]=="layer_nms"
        assert float(gt32[pos,gi])>=.75
        deleted=describe_box(stem,gi,model,"deleted_high_iou",ident,pos,boxes,scores,levels,cat,r,gt32,gt64,keep_position)
        entities[(ident,"deleted_high_iou")]=deleted
        hits=[];near=[]
        for k,j in enumerate(keep.tolist()):
            flat=int(before["ids"][j]);ret=inventory[k]
            eq=bool(scores[j]==scores[pos]);higher=bool(scores[j]>scores[pos]);same_level=bool(levels[j]==levels[pos])
            f32,f64,off32=float(pair32[pos,k]),float(pair64[pos,k]),float(shifted32[pos,k])
            error=abs(scalar_iou(boxes[pos].tolist(),boxes[j].tolist())-f64);max_error=max(max_error,error)
            assert error<=1e-12
            raw32_gt=bool(pair32[pos,k]>threshold)
            raw64_gt=f64>threshold
            offset32_gt=bool(shifted32[pos,k]>threshold)
            possible=same_level and (eq or higher) and (raw32_gt or raw64_gt or offset32_gt)
            near_threshold=same_level and (eq or higher) and min(abs(f32-threshold32),abs(f64-threshold),abs(off32-threshold32))<=NEAR_WINDOW
            row={"stem":stem,"gt_index":gi,"model":model,
                 **{f"deleted_{key}":value for key,value in deleted.items() if key not in ("stem","gt_index","model","role")},
                 **{f"retained_{key}":value for key,value in ret.items() if key not in ("stem","gt_index","model","role")},
                 "same_level":same_level,"score_relation":"equal" if eq else "higher" if higher else "lower",
                 "score_tie":eq,"retained_minus_deleted_score":float(scores[j])-float(scores[pos]),
                 "mutual_iou_float32_cpu":f32,"mutual_iou_float64_cpu_diagnostic":f64,
                 "mutual_iou_shifted_float32_cpu_diagnostic":off32,
                 "original_nms_threshold":threshold,"threshold_as_float32":threshold32,
                 "float32_iou_minus_float32_threshold":f32-threshold32,
                 "float64_iou_minus_threshold":f64-threshold,
                 "shifted_float32_iou_minus_float32_threshold":off32-threshold32,
                 "raw_float32_strict_overlap":raw32_gt,"raw_float64_strict_overlap":raw64_gt,
                 "shifted_float32_strict_overlap":offset32_gt,
                 "overlap_predicate_disagreement":same_level and len({raw32_gt,raw64_gt,offset32_gt})>1,
                 "near_threshold_diagnostic_only":near_threshold,"possible_related_retained":possible,
                 "retained_has_higher_score_but_worse_specified_gt_iou":higher and float(gt32[j,gi])<float(gt32[pos,gi]),
                 "relationship_evidence":"possible_related_retained_only; no_actual_suppression_event"}
            audit.append(row)
            if near_threshold:near.append(row)
            if possible:
                hits.append(row);related.append(row);entities[(flat,"possibly_related_retained")]=ret
        eligible=[row for row in audit if row["deleted_candidate_id"]==deleted["candidate_id"] and row["same_level"] and row["score_relation"]!="lower"]
        summary.append({**deleted,"original_nms_retained_count":len(keep),"final_retained_count":len(final["ids"]),
            "possible_related_count":len(hits),"related_score_tie_count":sum(v["score_tie"] for v in hits),
            "related_higher_score_worse_gt_count":sum(v["retained_has_higher_score_but_worse_specified_gt_iou"] for v in hits),
            "near_threshold_pair_count":len(near),
            "nearest_threshold_margin_among_same_level_not_lower_score":min((min(abs(v["float32_iou_minus_float32_threshold"]),abs(v["float64_iou_minus_threshold"]),abs(v["shifted_float32_iou_minus_float32_threshold"])) for v in eligible),default=None),
            "unexplained_by_requested_retained_relation":not bool(hits),
            "actual_suppression_pair_proven":False})
    for (flat,role),entity in entities.items():
        pos=positions[flat]
        for index in range(len(all_gt)):
            error=abs(scalar_iou(boxes[pos].tolist(),all_gt[index].tolist())-float(gt64[pos,index]));max_error=max(max_error,error)
            assert error<=1e-12
            overlaps.append({"stem":stem,"model":model,"specified_gt_index":gi,"role":role,
                "candidate_id":entity["candidate_id"],"flat_index":flat,"level_key":entity["level_key"],
                "candidate_xyxy":entity["xyxy"],"compared_gt_index":index,"compared_gt_xyxy":json.dumps(all_gt[index].tolist()),
                "is_specified_gt":index==gi,"iou_float32_cpu":float(gt32[pos,index]),
                "iou_float64_cpu_diagnostic":float(gt64[pos,index]),"semantic_assignment_proven":False})
    info={"stem":stem,"gt_index":gi,"model":model,"capture":str(path),"capture_sha256":sha(path),
        "nms_input_count":len(boxes),"original_nms_output_count":len(keep),"final_count":len(final["ids"]),
        "original_nms_output_ids":after["ids"].tolist(),"final_ids":final["ids"].tolist(),
        "gt_count":len(all_gt),"all_gt_xyxy":all_gt.tolist(),"layer_keys":list(r["layer_keys"]),
        "candidate_ids_checked":list(wanted),"nms_source_output_membership_exact":True,
        "max_independent_scalar_vs_torch_float64_error":max_error,
        "shift_diagnostic_offset_per_level":{key:float(i*(boxes.max()+1)) for i,key in enumerate(r["layer_keys"])},
        "original_gpu_inner_nms_trace_saved":False}
    return audit,related,summary,inventory,overlaps,info

def report(out,related,summaries,overlaps,infos,metadata):
    lines=["# Day03：NMS删除达标框与原保留框的离线关系","",
        "只分析030269/G1 Fusion的ID2201，以及130126/G3 IR的ID3216、3516、3521、9216、9516。所有ID都带模型和图片上下文。",
        "仅加载自己保存的两份raw_capture张量；未连接服务器、加载checkpoint、执行模型前向或重新运行NMS。原GPU保留决策完全由保存的索引确定。",
        "","## 数据与计算口径","",
        "当前离线设备为CPU；输入坐标和原objectness为float32。GT IoU主口径为CPU float32 torchvision.box_iou，与Day02一致；float64是另列的几何诊断，并用独立Python binary64面积公式复核（atol=1e-12）。",
        "关系条件为同层、原分数更高或完全同分、相互IoU严格大于原NMS阈值0.7。相等0.7不符合严格条件。float32阈值表示为0.699999988079071；没有放宽阈值。",
        "另列float32按层坐标偏移后的IoU敏感性诊断：根据同版本本地boxes.py重建偏移坐标，没有执行NMS。这不是原CUDA内核输入的直接捕获。原CUDA内部逐次抑制事件和当时boxes.py哈希没有保存，不据此声称复现内核逐位决策。",
        "仅当上述几何口径之一满足严格重叠条件时列入可能关系，三口径的判定差异均单独标记。1e-6仅用于标记阈值附近的审计项，不加入关系判定或GT达标判定。保留位置从0开始。",""]
    for info in infos:
        lines.append(f"- {info['stem']}/G{info['gt_index']} {info['model']}：真实NMS输入{info['nms_input_count']}框、原NMS输出{info['original_nms_output_count']}框、最终{info['final_count']}框；全图GT={info['gt_count']}。")
    lines += ["","## 可能相关的保留框（不是已证实抑制者）","",
        "|案例/模型|删除ID/层|删除分数|删除框对指定GT IoU|保留ID/层|保留分数|保留框对指定GT IoU|相互IoU f32 CPU|NMS输出位置|通过最终截断|",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in related:
        lines.append(f"|{r['stem']}/G{r['gt_index']} {r['model']}|{r['deleted_flat_index']}/{r['deleted_level_key']}|{r['deleted_objectness']:.16f}|{r['deleted_specified_gt_iou_float32_cpu']:.12f}|{r['retained_flat_index']}/{r['retained_level_key']}|{r['retained_objectness']:.16f}|{r['retained_specified_gt_iou_float32_cpu']:.12f}|{r['mutual_iou_float32_cpu']:.12f}|{r['retained_nms_output_position_0based']}|{r['retained_passes_final_budget']}|")
    lines += ["","全部双侧坐标、原NMS输入位置、相互IoU三种口径及阈值余量见possible_related_retained.csv。pair_audit.csv保留每个删除ID对全部原NMS输出的比较，包括不满足层/分数/重叠条件的行；retained_inventory.csv包括被最终150预算截掉的保留框。", "","## 双方实际坐标",""]
    unique={}
    for r in related:
        for role in ("deleted","retained"):
            key=(r["stem"],r["model"],r[f"{role}_flat_index"])
            unique[key]=r[f"{role}_xyxy"]
    for (stem,model,flat),xyxy in unique.items():lines.append(f"- {stem} {model} ID{flat}：{xyxy}。")
    lines += ["","## 全图其他GT的几何重叠","",
        "|案例/模型|相关保留ID|指定GT IoU|最大其他GT IoU|该其他GT索引（0起始）|",
        "|---|---:|---:|---:|---|"]
    for stem,model,flat in sorted({(r["stem"],r["model"],r["retained_flat_index"]) for r in related}):
        subset=[r for r in overlaps if r["stem"]==stem and r["model"]==model and r["flat_index"]==flat]
        specified=next(r for r in subset if r["is_specified_gt"]);other=[r for r in subset if not r["is_specified_gt"]]
        maximum=max((r["iou_float32_cpu"] for r in other),default=0.)
        best=[r["compared_gt_index"] for r in other if r["iou_float32_cpu"]==maximum]
        lines.append(f"|{stem} {model}|{flat}|{specified['iou_float32_cpu']:.12f}|{maximum:.12f}|{best}|")
    lines += ["","gt_overlaps.csv记录所有相关框、删除框与全图每个GT的重叠。最大重叠索引仅为几何描述，不是框的语义归属，也不证明NMS在邻近目标之间做了错误合并。", "","## 数值审计与解释边界",""]
    for s in summaries:
        lines.append(f"- {s['stem']} {s['model']} ID{s['flat_index']}：可能关系{s['possible_related_count']}条，同分关系{s['related_score_tie_count']}条；阈值附近关系对{s['near_threshold_pair_count']}条；同层且分数不低的比较中距阈值最近余量={s['nearest_threshold_margin_among_same_level_not_lower_score']:.16g}。未解释={s['unexplained_by_requested_retained_relation']}。")
    worse=sum(r["retained_has_higher_score_but_worse_specified_gt_iou"] for r in related)
    disagreement=sum(r["overlap_predicate_disagreement"] for r in related)
    lines += ["",f"本次共{len(related)}条可能关系，其中{worse}条呈现保留框分数更高、对指定GT定位更差；可能关系中的三口径阈值判定不一致数={disagreement}。",
        "已证实：六个达标候选进入原NMS后不在输出中；列出的相关框确在原NMS全部输出中，其层、分数、坐标、相互重叠及对GT定位差异均可由保存数据计算。",
        "尚未证实：每个删除框究竟由哪个框实际抑制。即使只有一个符合关系条件的最终保留框，也没有原CUDA逐次抑制事件作直接证据，本轮未实现独立贪心NMS归因。分数排序、几何相容关系不能命名为实际抑制配对。",
        "当前证据足以提出一个下一项明确假设：在这些具体个例中，objectness排序更偏好对指定GT定位较差、但与达标候选重叠较大的框，这可能参与高IoU候选的NMS丢失。该假设限定于对应案例，仍需直接抑制证据验证，不推广到整个验证集，也不据此修改融合结构、NMS阈值或预算。",
        "若有删除ID未找到满足条件的原保留框，需将其保留为未解释项；当前六个ID的计数见上表，不能用猜测补齐。",
        "","## 运行与保留核验","","~~~powershell",metadata["command"],"~~~", "",
        "运行环境、输入/源码SHA、实际数值误差和前后保留核验见run_metadata.json、numeric_verification.json；实际退出码见execution_status.json，完整日志见run.log。本轮完成后停止，未实施下一实验。"]
    with (out/"analysis_report.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s);stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:stream.flush()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(HERE):raise ValueError("All new artifacts must stay in this new analysis directory")
    out.mkdir(parents=True,exist_ok=False)
    protected=[p for p in DAY03.rglob("*") if p.is_file() and not p.is_relative_to(HERE) and "__pycache__" not in p.parts]
    before=snapshot(protected)
    meta={"status":"running","command":subprocess.list2cmdline(sys.orig_argv),
        "started_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"device":"cpu",
        "environment":{"python":sys.version,"executable":sys.executable,"torch":torch.__version__,
            "torchvision":torchvision.__version__,"platform":platform.platform()},
        "model_forward_performed":False,"checkpoint_loaded":False,"server_connected":False,"nms_executed":False,
        "input_dtype":"float32","diagnostic_dtype":"float64","tie_policy":"mark exact original score equality; no reordering",
        "near_threshold_audit_window":NEAR_WINDOW,"threshold_modified":False,
        "source_sha256":{str(Path(__file__)):sha(__file__),str(Path(inspect.getfile(box_ops))):sha(inspect.getfile(box_ops))},
        "protected_sha256_before":before}
    code=1
    with (out/"run.log").open("x",encoding="utf-8") as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
       try:
        print("COMMAND",meta["command"],flush=True)
        manifest=json.loads((CAPTURES/"artifact_manifest.json").read_text(encoding="utf-8"))
        assert all(sha(CAPTURES/n)==h for n,h in manifest.items())
        results=[]
        # These guards fail if accidental model execution or Python NMS is introduced.
        with patch.object(torch.nn.Module,"_call_impl",side_effect=RuntimeError("Model forward forbidden")), \
             patch.object(box_ops,"nms",side_effect=RuntimeError("NMS execution forbidden")), \
             patch.object(box_ops,"batched_nms",side_effect=RuntimeError("NMS execution forbidden")):
            for case in CASES:results.append(analyze_case(*case))
        audit=[r for v in results for r in v[0]];related=[r for v in results for r in v[1]]
        summaries=[r for v in results for r in v[2]];inventory=[r for v in results for r in v[3]]
        overlaps=[r for v in results for r in v[4]];infos=[v[5] for v in results]
        assert len(summaries)==6 and len(audit)==105+5*171 and len(inventory)==105+171
        near=[r for r in audit if r["near_threshold_diagnostic_only"]]
        for name,rows,fields in (("pair_audit.csv",audit,list(audit[0])),
                ("possible_related_retained.csv",related,list(audit[0])),
                ("threshold_near_pairs.csv",near,list(audit[0])),
                ("candidate_summary.csv",summaries,list(summaries[0])),
                ("retained_inventory.csv",inventory,list(inventory[0])),
                ("gt_overlaps.csv",overlaps,list(overlaps[0]))):write_csv(out/name,rows,fields)
        with (out/"nms_source_readonly.txt").open("x",encoding="utf-8") as f:
            f.write(inspect.getsource(box_ops.batched_nms)+"\n"+inspect.getsource(box_ops._batched_nms_coordinate_trick)+"\n"+box_ops.nms.__doc__)
        meta["cases"]=infos
        check={"status":"passed","audited_deleted_candidates":len(summaries),"pair_audit_rows":len(audit),
            "possible_relation_rows":len(related),"retained_inventory_rows":len(inventory),"gt_overlap_rows":len(overlaps),
            "near_threshold_rows":len(near),"all_audit_score_tie_rows":sum(r["score_tie"] for r in audit),
            "all_audit_same_level_predicate_disagreement_rows":sum(r["overlap_predicate_disagreement"] for r in audit),
            "unexplained_candidate_ids":[r["candidate_id"] for r in summaries if r["unexplained_by_requested_retained_relation"]],
            "actual_suppression_pairs_proven":0,"independent_scalar_float64_atol":1e-12,
            "max_scalar_float64_error":max(i["max_independent_scalar_vs_torch_float64_error"] for i in infos),
            "original_capture_manifest_files_verified":len(manifest),"guarded_no_model_forward_or_python_nms":True}
        save_json(out/"numeric_verification.json",check)
        report(out,related,summaries,overlaps,infos,meta)
        for s in summaries:
            print(f"CANDIDATE_VERIFIED {s['stem']}/G{s['gt_index']} {s['model']} ID{s['flat_index']} possible_relations={s['possible_related_count']} ties={s['related_score_tie_count']} unexplained={s['unexplained_by_requested_retained_relation']}",flush=True)
        print("NUMERIC_VERIFIED",json.dumps(check),flush=True)
        meta["status"]="completed";code=0
       except BaseException:
        meta["status"]="failed";meta["failure"]=traceback.format_exc();traceback.print_exc()
       finally:
        meta["protected_sha256_after"]=snapshot(protected)
        meta["protected_files_unchanged"]=before==meta["protected_sha256_after"]
        if not meta["protected_files_unchanged"]:meta["status"]="failed";code=1
        meta["exit_code"]=code;meta["finished_at_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(out/"run_metadata.json",meta);save_json(out/"execution_status.json",{"status":meta["status"],"exit_code":code})
        print("OFFLINE_ANALYSIS_EXIT",code,"historical_files_unchanged",meta["protected_files_unchanged"],flush=True)
    return code

if __name__=="__main__":sys.exit(main())
