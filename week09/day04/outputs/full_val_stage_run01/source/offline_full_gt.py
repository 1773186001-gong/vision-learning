"""Six saved validation images, all GT, CPU only. Never imports model restoration adapters."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import csv
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
from torchvision.ops import boxes as box_ops
from coverage_statistics import STAGES,CONVENTIONS,measure_image,summarize

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
D1=ROOT/"week09/day01/outputs/val_rpn_run01"
D2=ROOT/"week09/day02/outputs/proposal_visualization_run01"
D3=ROOT/"week09/day03/outputs/remaining_five_merged_run01"

def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read_csv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def load_json(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def save_json(p,v):
    with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def write_csv(p,rows,fields=None):
    with p.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
def exact(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):raise ValueError("Captured tensor mismatch")
def historical_hash(mapping,suffix):
    found=[v for p,v in mapping.items() if p.replace("\\","/").endswith(suffix)]
    if len(found)!=1:raise ValueError("Missing/ambiguous source hash: "+suffix)
    return found[0]

def verify_raw(data):
    if len(data["records"])!=1 or len(data["stages"])!=1:raise ValueError("Expected single image capture")
    r,s=data["records"][0],data["stages"][0]
    if tuple(s)!=STAGES or r["layer_keys"]!=("0","1","pool") or r["image_size"]!=(640,640):raise ValueError("Incomplete stages/geometry/keys")
    if (r["config"]["pre_nms"],r["config"]["post_nms"],r["config"]["score_thresh"],r["config"]["nms_thresh"],r["config"]["min_size"])!=(150,150,.05,.7,.001):raise ValueError("Budget/config changed")
    ids=torch.arange(len(r["boxes"]));levels=torch.repeat_interleave(torch.arange(3),torch.tensor(r["counts"]))
    if sum(r["counts"])!=len(ids) or r["logits"].numel()!=len(ids):raise ValueError("Decoded dimensions")
    exact(s["decoded"]["ids"],ids);exact(s["decoded"]["boxes"],r["boxes"])
    cat=data["catalogues"][0]
    if len(cat)!=len(ids) or any(e["flat_index"]!=i or e["candidate_id"]!=f"{r['prefix']}/{i}" or e["level_key"]!=r["layer_keys"][int(levels[i])] for i,e in enumerate(cat)):raise ValueError("Catalogue identities")
    top=r["top_indices"]
    if top.dtype!=torch.int64 or len(top.unique())!=len(top) or bool(((top<0)|(top>=len(ids))).any()):raise ValueError("Top indices")
    exact(s["pre_nms_topn"]["ids"],ids[top]);exact(s["pre_nms_topn"]["boxes"],r["boxes"][top])
    if any(int((levels[top]==i).sum())!=min(150,n) for i,n in enumerate(r["counts"])):raise ValueError("Per-level budget")
    exact(s["clipped"]["ids"],ids[top]);exact(s["clipped"]["boxes"],box_ops.clip_boxes_to_image(r["boxes"][top],r["image_size"]))
    exact(s["clipped"]["scores"],s["pre_nms_topn"]["scores"])
    def select(old,new,index):
        for field in ("ids","boxes","scores"):exact(new[field],old[field][index])
    select(s["clipped"],s["small_box_filtered"],box_ops.remove_small_boxes(s["clipped"]["boxes"],r["config"]["min_size"]))
    prev=s["small_box_filtered"];select(prev,s["score_filtered"],torch.where(prev["scores"]>=r["config"]["score_thresh"])[0])
    for f in ("boxes","scores"):exact(s["score_filtered"][f],r["nms_"+f])
    exact(levels[s["score_filtered"]["ids"]],r["nms_levels"])
    keep=r["nms_indices"]
    if keep.dtype!=torch.int64 or len(keep.unique())!=len(keep) or bool(((keep<0)|(keep>=len(r["nms_boxes"]))).any()):raise ValueError("NMS indices")
    select(s["score_filtered"],s["layer_nms"],keep)
    select(s["layer_nms"],s["post_nms_topn"],torch.arange(min(150,len(keep))))
    exact(s["post_nms_topn"]["boxes"],r["final_boxes"]);exact(s["post_nms_topn"]["scores"],r["final_scores"])
    exact(r["final_boxes"],data["observed"]["rpn_forward_boxes"])
    if any(data["observed"][k]!=1 for k in ("model_forward_calls","transform_calls","rpn_calls")):raise ValueError("Original forward evidence")
    exact(data["observed"]["selected_gt"],r["gt"])
    return r,s,cat,data["observed"]["transform_gt_all"]

def reuse_preflight():
    run=load_json(D3/"run_metadata.json");d1=load_json(D1/"rpn_metrics.json");d2=load_json(D2/"run_metadata.json")
    if run["status"]!="completed" or run["exit_code"]!=0 or run["protected_sha256_before"]!=run["protected_sha256_after"]:raise ValueError("Unaccepted capture run")
    manifest=load_json(D3/"artifact_manifest.json")
    if not all(sha(D3/n)==h for n,h in manifest.items()):raise ValueError("Capture artifact manifest mismatch")
    source_evidence=[]
    suffixes=["week08/day01/llvip_dataset.py","week08/day02/paired_letterbox.py","week08/day02/paired_horizontal_flip.py",
              "week08/day03/ir_common.py","week08/day05/fusion_model.py","week08/day05/fusion_common.py",
              "week09/day01/rpn_diagnostic.py","data/LLVIP_1000/manifest.csv"]
    for relative in suffixes:
        current=sha(ROOT/relative);expected=historical_hash(d1["inputs"]["file_sha256"],relative)
        if current!=expected or current!=historical_hash(run["protected_sha256_before"],relative) or current!=historical_hash(d2["input_sha256_before"],relative):raise ValueError("Source/input changed: "+relative)
        source_evidence.append({"path":str(ROOT/relative),"sha256":current,"day01_day02_day03_match":True})
    for path,h in run["source_sha256"].items():
        local=ROOT/"week09/day03"/Path(path).name
        if sha(local)!=h:raise ValueError("Day03 capture source differs: "+str(local))
    for relative in ("models/detection/rpn.py","ops/boxes.py","models/detection/transform.py"):
        local=Path(torchvision.__file__).parent/relative
        if sha(local)!=historical_hash(d1["inputs"]["file_sha256"],"torchvision/"+relative):raise ValueError("Local geometry source changed")
        source_evidence.append({"path":str(local),"sha256":sha(local),"day01_source_match":True})
    val=[r for r in read_csv(ROOT/"data/LLVIP_1000/manifest.csv") if r["split"]=="val"]
    selected=read_csv(D2/"selected_cases.csv")
    if len(run["cases"])!=len(selected) or len({c["stem"] for c in run["cases"]})!=len(selected):raise ValueError("Selected cases mismatch")
    for case,row in zip(run["cases"],selected):
        if (case["stem"],int(case["gt_index"]))!=(row["stem"],int(row["gt_index"])):raise ValueError("Case order differs")
    return run,d1,d2,val,selected,source_evidence

def run_pilot():
    run,d1,d2,val,selected,source_evidence=reuse_preflight()
    old_rows=read_csv(D1/"rpn_per_gt.csv");day1={(r["model"],r["stem"],int(r["gt_index"])):r for r in old_rows}
    if len(day1)!=len(old_rows):raise ValueError("Duplicate Day01 GT keys")
    for model in ("IR","Fusion"):
        if len([k for k in day1 if k[0]==model])!=d1["models"][model]["total_valid_gt"]:raise ValueError("Day01 GT count")
    day3={(r["model"],r["stem"],int(r["gt_index"]),r["stage"]):r for r in read_csv(D3/"merged_stage_summary.csv")}
    allrows=[];images=[];reuse=[];comparisons=[];day3_checks=[]
    for case,row in zip(run["cases"],selected):
        index=int(row["image_index"]);mr=val[index]
        if (mr["stem"],mr["scene_id"])!=(row["stem"],row["scene_id"]):raise ValueError("Val identity mismatch")
        input_hashes={}
        for field in ("visible","infrared","annotation_voc"):
            local=ROOT/"data/LLVIP_1000"/mr[field]
            h=sha(local);suffix=str(local.relative_to(ROOT)).replace("\\","/")
            if h!=historical_hash(run["protected_sha256_before"],suffix) or h!=historical_hash(d2["input_sha256_before"],suffix):raise ValueError("Selected image hash mismatch")
            input_hashes[str(local)]=h
        pair_gt=None
        for model in ("IR","Fusion"):
            directory=D3/case["directory"];raw=directory/f"{model}_raw_capture.pth"
            data=torch.load(raw,map_location="cpu",weights_only=True)
            rec,stages,cat,gt=verify_raw(data)
            if len(gt)!=int(mr["person_boxes"]):raise ValueError("All-GT count differs from manifest")
            if pair_gt is not None:exact(gt,pair_gt)
            pair_gt=gt
            model_info=load_json(directory/f"{model}_historical_comparison.json")
            if not model_info["restoration_verified"] or not model_info["config_unchanged"] or model_info["state_sha256_before"]!=model_info["state_sha256_after"]:raise ValueError("State/config/restoration evidence failed")
            checkpoint=d1["models"][model]["checkpoint"]["sha256_before"]
            if model_info["checkpoint_sha256_before"]!=checkpoint or model_info["checkpoint_sha256_after"]!=checkpoint:raise ValueError("Checkpoint historical hash differs")
            if model_info["config"]!=d1["models"][model]["rpn_config"]:raise ValueError("Original RPN/transform config differs")
            rows,im=measure_image(model,row["stem"],mr["scene_id"],index,gt,stages,cat,rec["image_size"])
            allrows.extend(rows);images.append(im)
            for gi,g in enumerate(gt):
                key=(model,row["stem"],gi);historic=day1[key]
                if historic["scene_id"]!=mr["scene_id"] or int(historic["image_index"])!=index:raise ValueError("Day01 identity differs")
                exact(g,torch.tensor([float(historic[k]) for k in ("xmin","ymin","xmax","ymax")]))
                final=next(r for r in rows if r["gt_index"]==gi and r["stage"]=="post_nms_topn")
                delta=final["max_iou_float32_cpu"]-float(historic["max_iou_top300"])
                if abs(delta)>1e-6 or final["covered_075_float32_cpu"]!=bool(int(historic["covered_top300_iou0.75"])):raise ValueError("Day01 final comparison differs; no tolerance relaxation")
                if final["proposal_count"]!=int(historic["proposal_count"]):raise ValueError("Day01 proposal count differs")
                comparisons.append({"model":model,"stem":row["stem"],"gt_index":gi,"day01_max_iou":float(historic["max_iou_top300"]),
                    "pilot_max_iou":final["max_iou_float32_cpu"],"actual_difference":delta,"atol":1e-6,"rtol":0,
                    "coverage_equal":True})
            for r in rows:
                if r["gt_index"]!=int(row["gt_index"]):continue
                old=day3[model,row["stem"],r["gt_index"],r["stage"]]
                for convention in CONVENTIONS:
                    for field in ("max_iou_","count_iou_ge_075_","best_candidate_id_"):
                        key=field+convention
                        old_key=("best_id_" if field=="best_candidate_id_" else field)+convention
                        v=r[key];o=old[old_key]
                        if field=="max_iou_":
                            err=float(v)-float(o)
                            if abs(err)>(0 if convention=="float32_cpu" else 1e-12):raise ValueError("Selected Day03 IoU mismatch")
                        elif field=="count_iou_ge_075_":
                            if v!=int(o):raise ValueError("Day03 high count differs")
                        elif v!=o:raise ValueError("Day03 best ID differs")
                day3_checks.append({"model":model,"stem":row["stem"],"gt_index":r["gt_index"],"stage":r["stage"],"float32_exact":True,"float64_atol":1e-12,"best_ids_and_counts_equal":True})
            reuse.append({"model":model,"stem":row["stem"],"scene_id":mr["scene_id"],"image_index":index,
                "all_gt_count":len(gt),"raw_path":str(raw),"raw_sha256":sha(raw),"inputs_sha256":input_hashes,
                "checkpoint_sha256_historical":checkpoint,"state_sha256_historical":model_info["state_sha256_after"],
                "original_config":model_info["config"],"all_seven_stages_present":True,"real_indices_verified":True,
                "all_gt_match_day01":True,"current_server_or_weight_file_rechecked":False})
    summaries=summarize(allrows,images)
    remaining=[dict(image_index=i,**r) for i,r in enumerate(val) if r["stem"] not in {c["stem"] for c in run["cases"]}]
    evidence={"status":"passed_for_local_offline_reuse","scope":"six_image_all_gt_pilot_only","captures":reuse,
        "source_evidence":source_evidence,"capture_manifest_files_verified":len(load_json(D3/"artifact_manifest.json")),
        "source_run_metadata_sha256":sha(D3/"run_metadata.json"),"day01_metrics_sha256":sha(D1/"rpn_metrics.json"),
        "day01_gt_csv_sha256":sha(D1/"rpn_per_gt.csv"),"future_server_preflight_required":True}
    validation={"status":"passed","selected_day03_stage_rows_compared":len(day3_checks),"day03_checks":day3_checks,
        "day01_final_gt_model_items_compared":len(comparisons),"day01_comparisons":comparisons,
        "full_val_reference":{m:{"images":d1['models'][m]['processed_images'],"valid_gt":d1['models'][m]['total_valid_gt'],
            "covered_075":d1['models'][m]['top_k']['300']['hits']['0.75']} for m in ("IR","Fusion")},
        "remaining_images":len(remaining),"expected_new_full_image_forwards_after_authorization":len(remaining)*2,
        "remaining_gt_per_model_from_manifest":sum(int(r["person_boxes"]) for r in remaining)}
    return allrows,images,summaries,evidence,validation,remaining

def report(out,rows,images,summaries,validation):
    stage=summaries["stage_summary"];net=summaries["net_difference"]
    lines=["# Week09 Day04：已有六图全GT离线试运行","",
        "这只是Day03已选六图的全GT试运行，不是完整validation结果。六图存在选择偏差，不能估计全val比例。未连接服务器、加载checkpoint、执行新模型前向或NMS，也没有修改阈值／预算。",
        "","## 实际样本","","|图片|场景|GT数（每模型）|","|---|---|---:|"]
    for im in images:
        if im["model"]=="IR":lines.append(f"|{im['stem']}|{im['scene_id']}|{im['gt_count']}|")
    for model in ("IR","Fusion"):
        ims=[i for i in images if i['model']==model]
        lines.append(f"\n{model}：{len(ims)}图，{sum(i['gt_count'] for i in ims)}个GT，空GT图{sum(i['empty_gt'] for i in ims)}张。")
    lines += ["","## CPU float32历史一致口径","",
        "|阶段|IR覆盖/GT|Fusion覆盖/GT|IR−Fusion净差|净差较前阶段变化|",
        "|---|---:|---:|---:|---:|"]
    for n in net:
        if n["scope"]=="overall" and n["iou_convention"]=="float32_cpu":
            lines.append(f"|{n['stage']}|{n['IR_covered_gt']}/{n['valid_gt']}|{n['Fusion_covered_gt']}/{n['valid_gt']}|{n['IR_minus_Fusion']}|{n['net_change_from_previous']}|")
    lines += ["","净差变化=(Fusion丢失−IR丢失)+(IR新增覆盖−Fusion新增覆盖)，只是算术分解，不是因果贡献。总体覆盖率按GT汇总，不平均逐图比例。",
        "","## 覆盖丢失及条件率","","|模型|阶段|丢失GT|前阶段已覆盖GT|条件率|部分好框减少但仍覆盖GT|",
        "|---|---|---:|---:|---:|---:|"]
    for r in stage:
        if r["scope"]=="overall" and r["iou_convention"]=="float32_cpu" and r["stage"]!="decoded":
            lines.append(f"|{r['model']}|{r['stage']}|{r['coverage_lost_gt']}|{r['previous_covered_gt']}|{r['lost_rate_previous_covered']}|{r['partial_high_loss_coverage_kept_gt']}|")
    lines += ["","## 七阶段始终未覆盖与配对",""]
    for r in summaries["never_covered_summary"]:
        if r["scope"]=="overall" and r["iou_convention"]=="float32_cpu":lines.append(f"- {r['model']}始终未覆盖：{r['never_covered_gt']}/{r['valid_gt']}。逐GT名单见never_covered_gt.csv。")
    for r in summaries["paired_summary"]:
        if r["scope"]=="overall" and r["iou_convention"]=="float32_cpu":lines.append(f"- {r['stage']}：两者覆盖{r['both']}、仅IR{r['IR_only']}、仅Fusion{r['Fusion_only']}、均无{r['neither']}，GT分母{r['valid_gt']}。")
    lines += ["","完整场景表、float64诊断表及逐GT完整覆盖序列另存CSV；两种精度不混用。裁剪后的首次达标与恢复不会被忽略，0.75阈值没有加容差。候选数量减少不等于GT覆盖丢失。",
        "","## 真实验收与范围","",
        f"原六个指定GT共{validation['selected_day03_stage_rows_compared']}行与Day03逐项一致（f32精确、f64 atol=1e-12，数量与最佳ID一致）；全部GT共{validation['day01_final_gt_model_items_compared']}项末阶段与Day01逐GT核对，实际差值见verification.json，不放宽1e-6容差。",
        f"全val参考来自既有Day01：{validation['full_val_reference']}。当前没有执行全val捕获。满足未来预检后，尚需{validation['remaining_images']}张图、预计{validation['expected_new_full_image_forwards_after_authorization']}次新增完整图片前向。",
        "原核心只支持一个指定GT的捕获期诊断，但保存的候选与GT无关、全图GT齐全，因此本次离线扩展到所有GT，不修改原模型或旧核心。未来应适配空GT采集，不能为占位而伪造GT；具体分块与恢复方案见FULL_VAL_CAPTURE_PLAN.md。",
        "本次只说明这些选定图片内的分布与流水线是否能工作，不解释融合根因，不扩展NMS抑制者配对。完成本地试运行后停止，等待反馈。"]
    with (out/"pilot_report.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s);stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:stream.flush()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(HERE):raise ValueError("New output must stay within Day04")
    out.mkdir(parents=True,exist_ok=False)
    old=[q for d in (ROOT/"week09/day01",ROOT/"week09/day02",ROOT/"week09/day03") for q in d.rglob("*") if q.is_file() and "__pycache__" not in q.parts]
    before={str(q):sha(q) for q in old}
    meta={"scope":"six_saved_images_all_gt_pilot_not_full_validation","status":"running","command":subprocess.list2cmdline(sys.orig_argv),
        "started_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"model_forward_performed":False,"checkpoint_loaded":False,
        "server_connected":False,"nms_executed":False,"environment":{"python":sys.version,"executable":sys.executable,
        "torch":torch.__version__,"torchvision":torchvision.__version__,"device":"cpu","platform":platform.platform()},
        "source_sha256":{str(q):sha(q) for q in HERE.glob('*.py')},"historical_sha256_before":before}
    code=1
    with (out/"run.log").open("x",encoding="utf-8") as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
       try:
        print("COMMAND",meta["command"],flush=True);torch.set_num_threads(2)
        with patch.object(torch.nn.Module,"_call_impl",side_effect=RuntimeError("Model forward forbidden")), \
             patch.object(box_ops,"nms",side_effect=RuntimeError("NMS forbidden")), \
             patch.object(box_ops,"batched_nms",side_effect=RuntimeError("NMS forbidden")):
            rows,images,summaries,evidence,validation,remaining=run_pilot()
        write_csv(out/"per_gt_stage.csv",rows);write_csv(out/"per_image.csv",images)
        for name,rr in summaries.items():write_csv(out/f"{name}.csv",rr)
        seq=summaries["gt_coverage_sequences"]
        write_csv(out/"never_covered_gt.csv",[r for r in seq if r['never_covered']],list(seq[0]))
        write_csv(out/"remaining_val_manifest.csv",remaining)
        save_json(out/"reuse_evidence.json",evidence);save_json(out/"verification.json",validation)
        report(out,rows,images,summaries,validation)
        meta.update(status="completed",images_per_model={m:len([i for i in images if i['model']==m]) for m in ('IR','Fusion')},
            valid_gt_per_model={m:sum(i['gt_count'] for i in images if i['model']==m) for m in ('IR','Fusion')},per_gt_stage_rows=len(rows))
        print("PILOT_VERIFIED",json.dumps({k:meta[k] for k in ('images_per_model','valid_gt_per_model','per_gt_stage_rows')}),flush=True)
        print("HISTORICAL_COMPARISONS",validation['selected_day03_stage_rows_compared'],validation['day01_final_gt_model_items_compared'],flush=True)
        code=0
       except BaseException:
        meta["status"]="failed";meta["error"]=traceback.format_exc();traceback.print_exc()
       finally:
        meta["historical_sha256_after"]={str(q):sha(q) for q in old};meta["historical_files_unchanged"]=before==meta["historical_sha256_after"]
        if not meta["historical_files_unchanged"]:code=1;meta['status']='failed'
        meta['exit_code']=code
        save_json(out/'run_metadata.json',meta);save_json(out/'execution_status.json',{'status':meta['status'],'exit_code':code})
        print("OFFLINE_PILOT_EXIT",code,"historical_files_unchanged",meta['historical_files_unchanged'],flush=True)
    return code
if __name__=="__main__":sys.exit(main())
