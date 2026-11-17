"""Single authorized validation image 030269/G1. No training or test evaluation."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import csv
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"week09/day01"))
import rpn_diagnostic as day1
import trace_rpn_stages as core
import torch
import torchvision
from torchvision.ops import box_iou
STEM,GT_INDEX="030269",1
D2=ROOT/"week09/day02/outputs/proposal_visualization_run01"

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def save_json(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
    return value

def legacy_iou(boxes,gt):
    """Same Day02 convention: torchvision box_iou, float32 tensors, CPU."""
    return box_iou(cpu(gt).float()[None],cpu(boxes).float())[0]

def source_file(suffix,baseline):
    return next(v for k,v in baseline["input_sha256_before"].items()
                if k.replace("\\","/").endswith(suffix))

def snapshot_files(paths):
    return {str(p.resolve()):sha(p) for p in sorted(set(paths))}

def preflight():
    baseline=json.loads((D2/"run_metadata.json").read_text(encoding="utf-8"))
    required={"torch":torch.__version__,"torchvision":torchvision.__version__,
              "python":sys.version,"hostname":platform.node()}
    for name,value in required.items():
        if value!=baseline["environment"][name]:
            raise RuntimeError(f"Environment changed: {name}={value}; review compatibility before forward")
    if platform.node()!="DESKTOP-BFIO0RA" or not torch.cuda.is_available():
        raise RuntimeError("Verified original CUDA server required")
    if torch.cuda.get_device_name(0)!=baseline["environment"]["gpu"]:
        raise RuntimeError("GPU differs")
    sources=["week08/day01/llvip_dataset.py","week08/day02/paired_letterbox.py",
             "week08/day02/paired_horizontal_flip.py","week08/day03/ir_common.py",
             "week08/day05/fusion_model.py","week08/day05/fusion_common.py",
             "week09/day01/rpn_diagnostic.py","data/LLVIP_1000/manifest.csv"]
    for relative in sources:
        if sha(ROOT/relative)!=source_file(relative,baseline):
            raise RuntimeError("Protected source/input differs: "+relative)
    rpn_path=Path(inspect.getfile(core.rpn_module))
    if sha(rpn_path)!=source_file("/models/detection/rpn.py",baseline):
        raise RuntimeError("RPN source changed")
    lock_check=json.loads((ROOT/"week09/day03/checks/lock_release_regression_run01/verification.json").read_text(encoding="utf-8"))
    if lock_check["exit_code"]!=0 or sha(Path(core.__file__))!=lock_check["source_sha256"]["trace_rpn_stages.py"]:
        raise RuntimeError("Reviewed lock-fixed core checksum differs")
    for name,m in baseline["models"].items():
        path=ROOT/("week08/day03/outputs/ir_baseline_run01/best.pth" if name=="IR" else
                   "week08/day05/outputs/fusion_concat_run01/best.pth")
        if sha(path)!=m["checkpoint_sha256"]:raise RuntimeError(name+" best checksum differs")
    selected=[r for r in read_csv(D2/"selected_cases.csv")
              if r["stem"]==STEM and int(r["gt_index"])==GT_INDEX]
    if len(selected)!=1:raise RuntimeError("Expected exactly the fixed selected case")
    return baseline,selected[0],rpn_path

def config_check(current,expected):
    if json.loads(json.dumps(current))!=expected:
        raise RuntimeError("Original RPN/transform configuration differs from Day02")
    if current["active_pre_nms_top_n"]!=150 or current["active_post_nms_top_n"]!=150:
        raise RuntimeError("Original budgets must remain 150 per level / 150 per image")
    if current["score_thresh"]!=.05 or current["nms_thresh"]!=.7:
        raise RuntimeError("Original score/NMS thresholds differ")

def hook_state(modules):
    return [(module,name,dict(getattr(module,name))) for module in modules
            for name in ("_forward_hooks","_forward_pre_hooks")]

def hooks_restored(snapshot):
    return all(dict(getattr(module,name))==old for module,name,old in snapshot)

def save_capture(out,name,cap,observed):
    data={"scope":"030269/G1 only","observed":cpu(observed),
          "records":[cpu(asdict(r)) for r in cap.records] if cap else [],
          "stages":[{k:cpu(asdict(v)) for k,v in t.stages.items()} for t in cap.traces] if cap else [],
          "catalogues":[t.catalogue for t in cap.traces] if cap else [],
          "first_removed":[t.first_removed for t in cap.traces] if cap else [],
          "clip_deltas":[t.clip_deltas for t in cap.traces] if cap else []}
    with (out/f"{name}_raw_capture.pth").open("xb") as f:torch.save(data,f)

@torch.inference_mode()
def capture_model(name,args,image,target,selected,baseline,metadata):
    checkpoint=ROOT/("week08/day03/outputs/ir_baseline_run01/best.pth" if name=="IR" else
                     "week08/day05/outputs/fusion_concat_run01/best.pth")
    info={"checkpoint":str(checkpoint),"checkpoint_sha256_before":sha(checkpoint),"status":"loading"}
    metadata["models"][name]=info
    restore=day1.ir.checkpoint_model if name=="IR" else day1.fusion.restore
    model,ck=restore(checkpoint,args.device);del ck
    model.eval()
    detector=model if name=="IR" else model.detector
    original_config=day1.rpn_configuration(detector)
    config_check(original_config,baseline["models"][name]["rpn_config"])
    state_before=day1.tensor_digest(model)
    original_filter=detector.rpn.filter_proposals
    original_top=detector.rpn._get_top_n_idx
    method_state={k:(k in detector.rpn.__dict__,detector.rpn.__dict__.get(k))
                  for k in ("filter_proposals","_get_top_n_idx")}
    original_nms=core.rpn_module.box_ops.batched_nms
    old_hooks=hook_state([model,detector.transform,detector.rpn])
    observed={"model_forward_calls":0,"transform_calls":0,"rpn_calls":0}
    cap=None;finished=False
    def model_pre(module,inputs):
        observed["model_forward_calls"]+=1
        if observed["model_forward_calls"]!=1:raise RuntimeError("Only one original eval forward is authorized")
    def transform_hook(module,inputs,output):
        images,targets=output
        observed["transform_calls"]+=1
        observed["transform_sizes"]=list(images.image_sizes)
        observed["transform_gt_all"]=targets[0]["boxes"].detach().clone()
        if observed["transform_calls"]!=1 or images.image_sizes!=[(640,640)]:
            raise RuntimeError("Unexpected transform calls or geometry")
        core.same(targets[0]["boxes"],target["boxes"].to(args.device),"original letterbox GT")
    try:
        with ExitStack() as stack:
            def rpn_pre(module,inputs):
                nonlocal cap
                images,features,targets=inputs
                if cap is not None:raise RuntimeError("Multiple RPN calls")
                keys=tuple(features.keys())
                if keys!=("0","1","pool"):raise RuntimeError("Actual feature key order changed")
                if list(images.image_sizes)!=observed["transform_sizes"]:
                    raise RuntimeError("RPN geometry differs from transform")
                core.same(targets[0]["boxes"],observed["transform_gt_all"],"same-call transformed GT")
                g=targets[0]["boxes"][GT_INDEX]
                expected=g.new_tensor([float(selected[f"gt_{k}"]) for k in ("xmin","ymin","xmax","ymax")])
                core.same(g,expected,"selected Day02 GT")
                observed["selected_gt"]=g.detach().clone()
                observed["layer_keys"]=keys
                observed["feature_shapes"]={k:list(v.shape) for k,v in features.items()}
                cap=stack.enter_context(core.TraceCapture(module,keys,g,
                    prefix=f"{args.output.name}/{name}/{STEM}",image_atol=1e-6))
            def rpn_hook(module,inputs,output):
                observed["rpn_calls"]+=1
                proposals,losses=output
                if losses or len(proposals)!=1 or cap is None or len(cap.records)!=1:
                    raise RuntimeError("Unexpected original RPN output")
                observed["rpn_forward_boxes"]=proposals[0].detach().clone()
                core.same(proposals[0],cap.records[0].final_boxes,"original RPN forward boxes")
            for module,kind,fn in ((model,"pre",model_pre),(detector.transform,"post",transform_hook),
                                   (detector.rpn,"pre",rpn_pre),(detector.rpn,"post",rpn_hook)):
                handle=(module.register_forward_pre_hook(fn) if kind=="pre" else module.register_forward_hook(fn))
                stack.callback(handle.remove)
            moved=([image["infrared"].to(args.device)] if name=="IR" else
                   [{k:v.to(args.device) for k,v in image.items()}])
            moved_targets=[{k:v.to(args.device) for k,v in target.items()}]
            model(moved,moved_targets) # exactly one complete original eval forward
        if observed["model_forward_calls"]!=1 or observed["rpn_calls"]!=1 or cap is None or len(cap.traces)!=1:
            raise RuntimeError("Incomplete original forward capture")
        finished=True
    finally:
        # Preserve raw original information even if replay or a forward assertion failed.
        restored=(hooks_restored(old_hooks) and not core._LOCK.locked() and cap is not None and not cap.active
                  and detector.rpn.filter_proposals==original_filter and detector.rpn._get_top_n_idx==original_top
                  and core.rpn_module.box_ops.batched_nms is original_nms
                  and all((k in detector.rpn.__dict__,detector.rpn.__dict__.get(k))==v for k,v in method_state.items()))
        state_after=day1.tensor_digest(model)
        info.update(status="captured" if finished else "failed",forward_calls=observed["model_forward_calls"],
                    config=original_config,state_sha256_before=state_before,state_sha256_after=state_after,
                    checkpoint_sha256_after=sha(checkpoint),restoration_verified=restored,
                    config_unchanged=day1.rpn_configuration(detector)==original_config)
        save_capture(args.output,name,cap,observed)
        save_json(args.output/f"{name}_model_verification.json",info)
    if not restored or state_before!=state_after or not info["config_unchanged"] or info["checkpoint_sha256_after"]!=info["checkpoint_sha256_before"]:
        raise RuntimeError("State/checkpoint/config/restoration verification failed")
    record,trace=cap.records[0],cap.traces[0]
    old=next(r for r in read_csv(D2/"proposal_details.csv") if r["model"]==name and r["stem"]==STEM and int(r["gt_index"])==GT_INDEX)
    f32=legacy_iou(record.final_boxes,record.gt)
    maximum=float(f32.max()) if len(f32) else 0.
    difference=maximum-float(old["best_iou"])
    info.update(final_max_iou_float32_cpu=maximum,day02_max_iou_float32_cpu=float(old["best_iou"]),
                actual_difference=difference,atol=1e-6,rtol=0.,
                final_max_iou_float64=trace.stages["post_nms_topn"].summary["max_iou"])
    save_json(args.output/f"{name}_historical_comparison.json",info)
    if abs(difference)>1e-6 or (maximum>=.75)!=(float(old["best_iou"])>=.75):
        raise RuntimeError("Day02 IoU mismatch; no tolerance relaxation or attribution")
    print(f"MODEL_VERIFIED {name} forward_calls=1 f32_cpu={maximum:.16f} day02_delta={difference:.16g} f64={info['final_max_iou_float64']:.16f} restored=True",flush=True)
    del model,detector
    torch.cuda.empty_cache()
    return record,trace

def export_trace(out,name,record,trace):
    """All float32 columns are CPU historical convention; float64 comes from core."""
    from dataclasses import replace
    t=replace(trace,levels=cpu(trace.levels),
              stages={k:replace(v,ids=cpu(v.ids),boxes=cpu(v.boxes),scores=cpu(v.scores),iou=cpu(v.iou))
                      for k,v in trace.stages.items()})
    gt=cpu(record.gt)
    summaries=[];watched=set()
    for stage in t.stages.values():
        f32=legacy_iou(stage.boxes,gt)
        best32=int(f32.argmax()) if len(f32) else None
        best64=int(stage.iou.argmax()) if len(stage.iou) else None
        for b in (best32,best64):
            if b is not None:watched.add(int(stage.ids[b]))
        watched.update(stage.ids[stage.iou>=.75].tolist())
        watched.update(stage.ids[f32>=.75].tolist())
        row={"model":name,"stem":STEM,"gt_index":GT_INDEX,"stage":stage.name,
             "candidate_count":len(stage.ids),"counts_by_level":json.dumps(stage.summary["counts_by_level"]),
             "max_iou_float32_cpu":float(f32.max()) if len(f32) else 0.,
             "best_id_float32_cpu":t.catalogue[int(stage.ids[best32])]["candidate_id"] if best32 is not None else None,
             "count_iou_ge_075_float32_cpu":int((f32>=.75).sum()),
             "max_iou_float64_diagnostic":stage.summary["max_iou"],
             "best_id_float64_diagnostic":stage.summary["best_candidate_id"],
             "count_iou_ge_075_float64_diagnostic":stage.summary["count_iou_ge_075"]}
        summaries.append(row)
    rows=[]
    # Same candidate histories for all stage-best and all high-IoU IDs, not coordinate matching.
    for identity in sorted(watched):
        for row in t.history(identity):
            s=t.stages[row["stage"]];hit=torch.where(s.ids==identity)[0]
            row.update(model=name,stem=STEM,gt_index=GT_INDEX,flat_index=identity,
                       level_key=t.catalogue[identity]["level_key"],
                       role="stage_best_or_high_iou; fixed_identity_track",
                       iou_float64_diagnostic=row.pop("iou",None))
            row["iou_float32_cpu"]=float(legacy_iou(s.boxes[hit],gt)[0]) if len(hit) else None
            row["xyxy"]=json.dumps(row.get("xyxy"))
            for key in ("score","stage_position_0based","score_rank_min","score_rank_max","level_rank_min"):
                row.setdefault(key,None)
            rows.append(row)
    day1.write_csv(out/f"{name}_stage_summary.csv",summaries)
    if rows:
        day1.write_csv(out/f"{name}_candidate_trajectory.csv",rows)
    else:
        with (out/f"{name}_candidate_trajectory.csv").open("x",encoding="utf-8-sig",newline="") as f:
            csv.writer(f).writerow(["model","stem","gt_index","candidate_id","stage","kept"])
    if t.catalogue:
        day1.write_csv(out/f"{name}_candidate_catalogue.csv",t.catalogue)
    else:
        with (out/f"{name}_candidate_catalogue.csv").open("x",encoding="utf-8-sig",newline="") as f:
            csv.writer(f).writerow(["candidate_id","flat_index","level_key","level_index","index_within_level"])
    if t.clip_deltas:day1.write_csv(out/f"{name}_clip_changes.csv",t.clip_deltas)
    return summaries

def loss_transitions(rows,column):
    return [{"from":a["stage"],"to":b["stage"]} for a,b in zip(rows,rows[1:])
            if a[column]>0 and b[column]==0]

def report(out,tables,metadata):
    lines=["# Day03：030269/G1 单图RPN筛选阶段追踪","",
           "仅完整validation图片030269；指定GT索引1（0起始）。IR/Fusion各一次原eval前向，不训练、不使用test、不改预算。","",
           "## 各阶段实际观测","",
           "f32列：与Day02一致的CPU float32 torchvision.box_iou；f64列：核心模块float64诊断IoU，两者不可混用。",
           "候选ID稳定对应原展平索引，表内最佳框每阶段重新选取，不表示始终是同一框。","",
           "|模型|阶段|候选数|最大IoU f32 CPU|≥0.75数 f32|最大IoU f64|≥0.75数 f64|",
           "|---|---|---:|---:|---:|---:|---:|"]
    for name,rows in tables.items():
        for r in rows:
            lines.append(f"|{name}|{r['stage']}|{r['candidate_count']}|{r['max_iou_float32_cpu']:.9f}|{r['count_iou_ge_075_float32_cpu']}|{r['max_iou_float64_diagnostic']:.9f}|{r['count_iou_ge_075_float64_diagnostic']}|")
    lines+=["","## 高IoU框是否曾存在、何时全部消失",""]
    for name,rows in tables.items():
        losses=loss_transitions(rows,"count_iou_ge_075_float32_cpu")
        high=[r["stage"] for r in rows if r["count_iou_ge_075_float32_cpu"]>0]
        lines.append(f"- {name}（历史f32口径）：有高IoU候选的阶段为{high}；由有到无的转变为{losses}。")
        if losses:
            lines.append(f"  第一次全部消失发生于{losses[0]['to']}；需同时查看完整阶段表，不能忽略后续是否恢复。")
        else:
            lines.append("  没有记录到由有到无的阶段转变；这不等于证明某种上游机制。")
        diagnostic=loss_transitions(rows,"count_iou_ge_075_float64_diagnostic")
        lines.append(f"  f64诊断口径的由有到无转变：{diagnostic}。")
    lines+=["","## 验收与边界","",
            "重放的NMS输入、最终框、顺序和分数与同次原方法精确一致；最终框与RPN forward实际输出精确一致。阶段表/固定ID轨迹、原始捕获均保存，未配对猜测NMS抑制者。"]
    for name,m in metadata["models"].items():
        lines.append(f"- {name}：Day02 f32 IoU={m['day02_max_iou_float32_cpu']:.16f}，本次={m['final_max_iou_float32_cpu']:.16f}，实际差值={m['actual_difference']:.16g}；atol=1e-6、rtol=0；包装/hooks恢复={m['restoration_verified']}。")
    lines+=["","这是一个GT的阶段观察，不能推广到六图或整个验证集。阶段索引能证明某个候选在哪个操作被删除，不能证明融合特征或回归就是根因；原NMS只返回保留索引，未记录实际抑制配对。",
            "本次从解码候选开始，不覆盖anchor回归前的机制。对于未通过Top-N的框没有作反事实裁剪冒充实际阶段。",
            "受保护输入、权重、模型持久state及配置前后校验见run_metadata.json和各模型verification文件。","",
            "## 实际命令","", "~~~text",metadata["command"],"~~~","",
            "原始捕获：IR_raw_capture.pth、Fusion_raw_capture.pth。阶段表、候选目录和相关ID轨迹使用对应模型前缀。"]
    with (out/"day03_report.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    args=parser.parse_args();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    metadata={"status":"running","scope":"030269/G1 single_validation_image_only",
              "command":subprocess.list2cmdline(sys.orig_argv),"models":{},
              "started_at":dt.datetime.now(dt.timezone.utc).isoformat(),
              "training":False,"test_inference":False,"budget_modified":False}
    protected=[];before={}
    with (args.output/"run.log").open("x",encoding="utf-8") as log:
        with __import__("contextlib").redirect_stdout(day1.Tee(sys.stdout,log)),__import__("contextlib").redirect_stderr(day1.Tee(sys.stderr,log)):
            try:
                baseline,selected,rpn_path=preflight()
                day1.ir.seed_all(42);torch.set_num_threads(2)
                metadata["environment"]={"python":sys.version,"executable":sys.executable,
                    "hostname":platform.node(),"platform":platform.platform(),"torch":torch.__version__,
                    "torchvision":torchvision.__version__,"device":args.device,"gpu":torch.cuda.get_device_name(0),
                    "cudnn":torch.backends.cudnn.version(),"seed":42,"threads":2,
                    "cudnn_deterministic":torch.backends.cudnn.deterministic,
                    "cudnn_benchmark":torch.backends.cudnn.benchmark}
                data=day1.fusion.loader(ROOT/"data/LLVIP_1000","val",shuffle=False).dataset
                index=int(selected["image_index"]);record=data.get_record(index)
                if record.stem!=STEM:raise RuntimeError("Manifest index differs")
                image,target=data[index] # one complete paired image only, no GT crop
                image_ir,target_ir=day1.ir.loader(ROOT/"data/LLVIP_1000","val").dataset[index]
                core.same(image["infrared"],image_ir,"paired/IR input equivalence")
                for key in target:core.same(target[key],target_ir[key],"paired/IR target")
                roots=[ROOT/"week08",ROOT/"week09/day01",ROOT/"week09/day02",ROOT/"week09/day03/checks"]
                protected=[p for r in roots for p in r.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
                protected += [p for p in (ROOT/"week09/day03").iterdir() if p.is_file()]
                protected += [record.visible_path,record.infrared_path,record.annotation_path,
                              ROOT/"data/LLVIP_1000/manifest.csv",rpn_path]
                before=snapshot_files(protected);metadata["protected_sha256_before"]=before
                metadata["source_sha256"]={str(p):sha(p) for p in [Path(__file__),Path(core.__file__),rpn_path]}
                metadata["selected_case"]=selected
                print("COMMAND",metadata["command"],flush=True)
                print("ENVIRONMENT",json.dumps(metadata["environment"]),flush=True)
                print(f"PREFLIGHT_PASSED protected_files={len(before)} image={STEM} GT={GT_INDEX}",flush=True)
                tables={}
                for name in ("IR","Fusion"):
                    rec,trace=capture_model(name,args,image,target,selected,baseline,metadata)
                    tables[name]=export_trace(args.output,name,rec,trace)
                report(args.output,tables,metadata)
                metadata["status"]="completed";code=0
                print("COMPLETED single_image=030269 GT=1 IR_forwards=1 Fusion_forwards=1",flush=True)
            except BaseException:
                code=1;metadata["status"]="failed";metadata["failure"]=traceback.format_exc()
                traceback.print_exc()
                save_json(args.output/"failure.json",{"status":"failed","error":metadata["failure"],
                    "attribution_allowed":False,"message":"Preserve raw captures; no automatic rerun or tolerance relaxation"})
                print("FAILED: preserve original information; attribution stopped",flush=True)
            finally:
                if before:
                    metadata["protected_sha256_after"]=snapshot_files(protected)
                    metadata["protected_files_unchanged"]=metadata["protected_sha256_after"]==before
                    if not metadata["protected_files_unchanged"]:
                        code=1;metadata["status"]="failed"
                metadata["finished_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
                metadata["exit_code"]=code
                save_json(args.output/"run_metadata.json",metadata)
                save_json(args.output/"execution_status.json",{"status":metadata["status"],"exit_code":code,
                    "scope":metadata["scope"]})
    return code

if __name__=="__main__":sys.exit(main())


