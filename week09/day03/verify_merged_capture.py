"""Offline verification and Chinese report. Loads only saved tensor captures, never models."""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
from pathlib import Path
import subprocess
import sys
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import capture_single_val as single
from verify_single_capture import exact,area_iou64
import torch
from torchvision.ops import box_iou,clip_boxes_to_image,remove_small_boxes

def transitions(rows,col):
    high=[r["stage"] for r in rows if int(r[col])>0]
    lost=[b["stage"] for a,b in zip(rows,rows[1:]) if int(a[col])>0 and int(b[col])==0]
    appeared=[b["stage"] for a,b in zip(rows,rows[1:]) if int(a[col])==0 and int(b[col])>0]
    return {"high_stages":high,"first_high":high[0] if high else None,
            "all_lost_at":lost,"reappeared_or_first_appeared_after_decode":appeared}

def verify_case(directory,selected,model):
    directory=Path(directory);stem=selected["stem"];gi=int(selected["gt_index"])
    data=torch.load(directory/f"{model}_raw_capture.pth",map_location="cpu",weights_only=True)
    assert len(data["records"])==len(data["stages"])==1
    r,stages=data["records"][0],data["stages"][0]
    assert tuple(stages)==single.core.STAGES
    assert r["image_size"]==(640,640) and r["layer_keys"]==("0","1","pool")
    cfg=r["config"]
    assert (cfg["pre_nms"],cfg["post_nms"],cfg["score_thresh"],cfg["nms_thresh"])==(150,150,.05,.7)
    gt=torch.tensor([float(selected[f"gt_{k}"]) for k in ("xmin","ymin","xmax","ymax")],dtype=r["gt"].dtype)
    exact(r["gt"],gt);single.core.validate_geometry(r["boxes"],gt,(640,640))
    observed=data["observed"]
    assert observed["model_forward_calls"]==observed["transform_calls"]==observed["rpn_calls"]==1
    exact(observed["selected_gt"],gt);exact(observed["transform_gt_all"][gi],gt)
    exact(observed["rpn_forward_boxes"],r["final_boxes"])
    ids=torch.arange(len(r["boxes"]));levels=torch.repeat_interleave(torch.arange(3),torch.tensor(r["counts"]))
    catalogue=data["catalogues"][0]
    assert len(catalogue)==len(ids) and len({e["candidate_id"] for e in catalogue})==len(ids)
    offset=0
    for level,count in enumerate(r["counts"]):
        for local in range(count):
            e=catalogue[offset+local]
            assert e==dict(candidate_id=f"{r['prefix']}/{offset+local}",flat_index=offset+local,
                level_key=r["layer_keys"][level],level_index=level,index_within_level=local)
        offset+=count
    exact(stages["decoded"]["ids"],ids);exact(stages["decoded"]["boxes"],r["boxes"])
    top=r["top_indices"];single.core.indices(top,len(ids),"saved Top-N")
    exact(stages["pre_nms_topn"]["ids"],ids[top]);exact(stages["pre_nms_topn"]["boxes"],r["boxes"][top])
    assert all(int((levels[top]==i).sum())==min(150,n) for i,n in enumerate(r["counts"]))
    exact(stages["clipped"]["ids"],ids[top]);exact(stages["clipped"]["scores"],stages["pre_nms_topn"]["scores"])
    exact(stages["clipped"]["boxes"],clip_boxes_to_image(r["boxes"][top],r["image_size"]))
    before=area_iou64(stages["pre_nms_topn"]["boxes"],gt);after=area_iou64(stages["clipped"]["boxes"],gt)
    assert bool((after+1e-6>=before).all())
    def selected(previous,current,idx):
        single.core.indices(idx,len(previous["ids"]),"saved stage")
        for key in ("ids","boxes","scores"):exact(current[key],previous[key][idx])
    prev=stages["clipped"]
    selected(prev,stages["small_box_filtered"],remove_small_boxes(prev["boxes"],cfg["min_size"]))
    prev=stages["small_box_filtered"]
    selected(prev,stages["score_filtered"],torch.where(prev["scores"]>=cfg["score_thresh"])[0])
    prev=stages["score_filtered"]
    for key,raw in (("boxes","nms_boxes"),("scores","nms_scores")):exact(prev[key],r[raw])
    exact(levels[prev["ids"]],r["nms_levels"])
    selected(prev,stages["layer_nms"],r["nms_indices"])
    selected(stages["layer_nms"],stages["post_nms_topn"],torch.arange(min(150,len(stages["layer_nms"]["ids"]))))
    exact(stages["post_nms_topn"]["boxes"],r["final_boxes"]);exact(stages["post_nms_topn"]["scores"],r["final_scores"])
    rows=single.read_csv(directory/f"{model}_stage_summary.csv");assert len(rows)==7
    maxerr=0.;removed={}
    for i,row in enumerate(rows):
        assert (row["stem"],int(row["gt_index"]),row["model"],row["stage"])==(stem,gi,model,single.core.STAGES[i])
        s=stages[row["stage"]];f32=box_iou(gt.float()[None],s["boxes"].float())[0];f64=area_iou64(s["boxes"],gt)
        err=float((s["iou"]-f64).abs().max()) if len(f64) else 0.;maxerr=max(maxerr,err);assert err<=1e-12
        assert int(row["candidate_count"])==len(s["ids"])
        for values,suffix in ((f32,"float32_cpu"),(f64,"float64_diagnostic")):
            maximum=float(values.max()) if len(values) else 0.
            assert abs(float(row[f"max_iou_{suffix}"])-maximum)<=(0. if suffix=="float32_cpu" else 1e-12)
            assert int(row[f"count_iou_ge_075_{suffix}"])==int((values>=.75).sum())
            best=catalogue[int(s["ids"][int(values.argmax())])]["candidate_id"] if len(values) else ""
            assert row[f"best_id_{suffix}"]==best
        if i:
            prev=stages[rows[i-1]["stage"]]
            for ident in set(prev["ids"].tolist())-set(s["ids"].tolist()):removed[ident]=row["stage"]
            if row["stage"]!="clipped":
                assert float(row["max_iou_float32_cpu"])<=float(rows[i-1]["max_iou_float32_cpu"])+1e-6
                assert int(row["count_iou_ge_075_float32_cpu"])<=int(rows[i-1]["count_iou_ge_075_float32_cpu"])
    assert removed==data["first_removed"][0]
    tracks=single.read_csv(directory/f"{model}_candidate_trajectory.csv")
    watched=set()
    for s in stages.values():
        f32=box_iou(gt.float()[None],s["boxes"].float())[0]
        if len(s["ids"]):watched.update([int(s["ids"][int(f32.argmax())]),int(s["ids"][int(s["iou"].argmax())])])
        watched.update(s["ids"][f32>=.75].tolist());watched.update(s["ids"][s["iou"]>=.75].tolist())
    assert len(tracks)==7*len(watched) and {int(t["flat_index"]) for t in tracks}==watched
    assert len({(t["flat_index"],t["stage"]) for t in tracks})==len(tracks)
    for t in tracks:
        ident=int(t["flat_index"]);s=stages[t["stage"]];hit=torch.where(s["ids"]==ident)[0]
        assert (t["kept"]=="True")==bool(len(hit)) and t["candidate_id"]==catalogue[ident]["candidate_id"]
        assert t["first_removed_stage"]==(removed.get(ident) or "")
        if len(hit):
            p=int(hit[0]);assert json.loads(t["xyxy"])==s["boxes"][p].tolist()
            assert float(t["score"])==float(s["scores"][p])
            assert float(t["iou_float32_cpu"])==float(box_iou(gt.float()[None],s["boxes"][hit].float())[0,0])
    hist=next(d for d in single.read_csv(single.D2/"proposal_details.csv")
              if d["stem"]==stem and int(d["gt_index"])==gi and d["model"]==model)
    final=float(rows[-1]["max_iou_float32_cpu"]);old=float(hist["best_iou"]);delta=final-old
    assert abs(delta)<=1e-6 and (final>=.75)==(old>=.75)
    comparison=json.loads((directory/f"{model}_historical_comparison.json").read_text(encoding="utf-8"))
    assert comparison["actual_difference"]==delta and comparison["final_max_iou_float32_cpu"]==final
    assert comparison["restoration_verified"] and comparison["config_unchanged"]
    assert comparison["state_sha256_before"]==comparison["state_sha256_after"]
    assert comparison["checkpoint_sha256_before"]==comparison["checkpoint_sha256_after"]
    # Exact NMS replay was checked on the original device by the unchanged core;
    # here apply captured indices, not another CPU NMS with potentially different tie decisions.
    detail={"stem":stem,"gt_index":gi,"model":model,"stage_reconstruction_verified":True,
            "float32_cpu_final":final,"day02_float32_cpu":old,"actual_difference":delta,
            "atol":1e-6,"rtol":0.,"coverage_075_current":final>=.75,"coverage_075_day02":old>=.75,
            "float64_diagnostic_final":float(rows[-1]["max_iou_float64_diagnostic"]),
            "max_float64_cpu_vs_saved_error":maxerr,"float64_reconstruction_atol":1e-12,
            "f32_transitions":transitions(rows,"count_iou_ge_075_float32_cpu"),
            "f64_transitions":transitions(rows,"count_iou_ge_075_float64_diagnostic"),
            "stage_best_ids_f32":[r["best_id_float32_cpu"] for r in rows],
            "decoded_to_final_max_change":final-float(rows[0]["max_iou_float32_cpu"])}
    return detail,rows,tracks

def merge_report(output,cases,selected):
    output=Path(output);items=[];allrows=[];alltracks=[];summaryrows=[]
    lines=["# Day03 六例固定验证GT：RPN筛选追踪","",
        "030269/G1复用已验收捕获，新增五张完整val图片，每张IR/Fusion各一次原eval前向，共10次新增前向。未训练、未运行test、未修改预算或阈值。",
        "","本表主口径为Day02一致的CPU float32 IoU；float64仅作独立诊断，另存于CSV和核验JSON。候选数按真实索引追踪；每阶段的最佳框可能不同。",
        "","## 十二项逐阶段观测","","|案例/模型|阶段|候选数|最大IoU f32 CPU|≥0.75数|最佳ID（原flat索引）|",
        "|---|---|---:|---:|---:|---:|"]
    for case,row in zip(cases,selected):
        assert case["stem"]==row["stem"] and int(case["gt_index"])==int(row["gt_index"])
        for name in ("IR","Fusion"):
            detail,rows,tracks=verify_case(output/case["directory"],row,name)
            detail["new_forward_calls"]=0 if case["status"]=="reused_verified" else 1
            items.append(detail);allrows.extend(rows);alltracks.extend(tracks)
            summaryrows.append({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in detail.items()})
            for r in rows:
                lines.append(f"|{row['stem']}/G{row['gt_index']} {name}|{r['stage']}|{r['candidate_count']}|{float(r['max_iou_float32_cpu']):.12f}|{r['count_iou_ge_075_float32_cpu']}|{r['best_id_float32_cpu'].rsplit('/',1)[-1]}|")
    lines += ["","## 逐项解释与固定候选轨迹",""]
    for d in items:
        tag=f"{d['stem']}/G{d['gt_index']} {d['model']}";tr=d["f32_transitions"]
        lines += [f"### {tag}","",
            f"- 高IoU候选首次出现：{tr['first_high'] or '七阶段均无'}；有达标框的阶段：{tr['high_stages']}。",
            f"- 达标框由有到全部消失：{tr['all_lost_at']}；解码后由无到有：{tr['reappeared_or_first_appeared_after_decode']}。",
            f"- 阶段最佳flat索引依次为：{[s.rsplit('/',1)[-1] for s in d['stage_best_ids_f32']]}。",
            f"- 解码到最终最大IoU变化（最终减解码）：{d['decoded_to_final_max_change']:.12f}；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。",
            f"- Day02={d['day02_float32_cpu']:.16f}；当前={d['float32_cpu_final']:.16f}；差值={d['actual_difference']:.16g}。",""]
        relevant=[t for t in alltracks if t["stem"]==d["stem"] and t["model"]==d["model"] and t["kept"]=="True" and float(t["iou_float32_cpu"])>=.75]
        fixed=sorted({int(t["flat_index"]) for t in relevant})
        lines.append(f"曾达标的固定flat索引为{fixed}。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。")
        for ident in fixed:
            track=[t for t in alltracks if t["stem"]==d["stem"] and t["model"]==d["model"] and int(t["flat_index"])==ident]
            kept=[t["stage"] for t in track if t["kept"]=="True"]
            first=track[0]["first_removed_stage"] or "最终保留"
            lines.append(f"- ID {ident}：实际经过{kept}；首次删除={first}。")
        lines.append("")
    lines += ["## 解码质量与筛选变化的对照","",
        "|案例|IR解码最大IoU|Fusion解码最大IoU|IR最终|Fusion最终|IR筛选变化|Fusion筛选变化|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for i in range(0,len(items),2):
        a,b=items[i:i+2];ra=allrows[i*7];rb=allrows[(i+1)*7]
        lines.append(f"|{a['stem']}/G{a['gt_index']}|{float(ra['max_iou_float32_cpu']):.9f}|{float(rb['max_iou_float32_cpu']):.9f}|{a['float32_cpu_final']:.9f}|{b['float32_cpu_final']:.9f}|{a['decoded_to_final_max_change']:.9f}|{b['decoded_to_final_max_change']:.9f}|")
    lost={};never=[]
    for d in items:
        if d["f32_transitions"]["first_high"] is None:never.append(f"{d['stem']}/G{d['gt_index']} {d['model']}")
        for stage in d["f32_transitions"]["all_lost_at"]:lost.setdefault(stage,[]).append(f"{d['stem']}/G{d['gt_index']} {d['model']}")
    lines += ["","## 重复模式与证据边界","",
        f"达标候选全部消失的实际阶段分组：{json.dumps(lost,ensure_ascii=False)}。",
        f"七阶段均未达标的项：{never}。这些是按规则选择的六例，不能估计全val发生比例。",
        "操作索引与同ID坐标能证实在哪一步删除、裁剪后是否达标；这些观察不能证实融合特征、anchor回归或某个NMS抑制者就是根因。没有记录抑制者配对，也不从两个框重叠猜测实际抑制关系。",
        "解码阶段已有差异时，上游候选生成机制仍需另行验证；候选存在却筛选丢失时，后续可优先只读检查对应分数排名与筛选去留。当前证据不支持直接修改融合模块。",
        "裁剪范围为含Letterbox padding的完整640×640；GT完整位于范围内。对有限有效框，同ID裁剪IoU在精确几何下不降，float64核验atol=1e-6。未过Top-N的框不做反事实裁剪。",
        "","## 边界样本170455/G1 IR",""]
    boundary=next(d for d in items if d["stem"]=="170455" and d["model"]=="IR")
    lines += [f"历史IoU={boundary['day02_float32_cpu']:.16f}，当前={boundary['float32_cpu_final']:.16f}，实际差={boundary['actual_difference']:.16g}；历史覆盖={boundary['coverage_075_day02']}，当前覆盖={boundary['coverage_075_current']}。0.75阈值没有附加容差。",
        "","所有核验见merged_verification.json，各模型原始捕获与原state/config/恢复证据保存在子目录；全程来源、真实命令、退出码和受保护文件SHA前后值见run_metadata.json、process_status.json及run.log。完成本步骤后停止，不启动新实验。"]
    single.day1.write_csv(output/"merged_stage_summary.csv",allrows)
    single.day1.write_csv(output/"merged_candidate_trajectory.csv",alltracks)
    single.day1.write_csv(output/"merged_case_summary.csv",summaryrows)
    with (output/"six_case_report.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    return {"status":"passed","items":items,"item_count":len(items),"all_lost_stage_groups":lost,
            "never_high_items":never,"actual_command":subprocess.list2cmdline(sys.orig_argv),
            "new_forward_calls":sum(d["new_forward_calls"] for d in items),"model_forward_in_verifier":False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--capture-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);code=1
    with (args.output/"run.log").open("x",encoding="utf-8") as log:
      with contextlib.redirect_stdout(single.day1.Tee(sys.stdout,log)),contextlib.redirect_stderr(single.day1.Tee(sys.stderr,log)):
       try:
        source=args.capture_dir.resolve();before=single.snapshot_files([q for q in source.rglob("*") if q.is_file()])
        meta=json.loads((source/"run_metadata.json").read_text(encoding="utf-8"))
        assert meta["status"]=="completed" and meta["exit_code"]==0 and meta["new_forward_calls"]==10
        assert meta["protected_sha256_before"]==meta["protected_sha256_after"]
        manifest=json.loads((source/"artifact_manifest.json").read_text(encoding="utf-8"))
        assert all(single.sha(source/n)==h for n,h in manifest.items())
        rows=single.read_csv(single.D2/"selected_cases.csv");assert len(meta["cases"])==6
        items=[]
        for case,row in zip(meta["cases"],rows):
            for name in ("IR","Fusion"):
                d,_,_=verify_case(source/case["directory"],row,name);items.append(d)
                print("OFFLINE_VERIFIED",row["stem"],name,"day02_delta",d["actual_difference"],flush=True)
        reused=source/meta["cases"][0]["directory"]
        provenance=json.loads((reused/"reuse_provenance.json").read_text(encoding="utf-8"))
        original=HERE/"outputs/single_030269_run01"
        for path,h in provenance["source_sha256"].items():
            file=Path(path).name;assert single.sha(reused/file)==single.sha(original/file)==h
        merged=json.loads((source/"merged_verification.json").read_text(encoding="utf-8"))
        assert len(merged["items"])==12
        for actual,saved in zip(items,merged["items"]):
            assert all(saved[k]==v for k,v in actual.items())
        assert before==single.snapshot_files([q for q in source.rglob("*") if q.is_file()])
        protected=json.loads((HERE/"checks/local_unit_protected_before.json").read_text())
        assert all(single.sha(q)==h for q,h in protected.items())
        single.save_json(args.output/"verification.json",{"status":"passed","items":items,"artifact_sha256_verified":len(manifest),
            "local_historical_files_unchanged":len(protected),"reuse_original_sha256_verified":len(provenance["source_sha256"]),
            "actual_command":subprocess.list2cmdline(sys.orig_argv),"model_forward_performed":False,"checkpoint_loaded":False})
        code=0;print("OFFLINE_ALL_12_PASSED no_model_forward no_checkpoint_load",flush=True)
       except BaseException:traceback.print_exc()
       single.save_json(args.output/"execution_status.json",{"exit_code":code})
    return code
if __name__=="__main__":sys.exit(main())
