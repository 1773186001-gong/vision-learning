"""Offline acceptance of saved single-image captures; never loads weights or forwards a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import traceback
import torch
from torchvision.ops import box_iou,clip_boxes_to_image,remove_small_boxes
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def digest(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read_csv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def area_iou64(boxes,gt):
    b,g=boxes.double(),gt.double()
    intersection=(torch.minimum(b[:,2:],g[2:])-torch.maximum(b[:,:2],g[:2])).clamp(min=0).prod(1)
    return intersection/((b[:,2:]-b[:,:2]).clamp(min=0).prod(1)+(g[2:]-g[:2]).prod()-intersection)
def exact(a,b):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();o=args.capture_dir.resolve();dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    class Tee:
        def __init__(self,a,b):self.a=a;self.b=b
        def write(self,s):self.a.write(s);self.b.write(s);self.b.flush();return len(s)
        def flush(self):self.a.flush();self.b.flush()
    with (dest/"run.log").open("x",encoding="utf-8") as log:
        import contextlib
        with contextlib.redirect_stdout(Tee(sys.stdout,log)):
            try:
                manifest=json.loads((o.parent/"transfer_manifest.json").read_text())
                assert all(digest(o.parent/n)==info["sha256"] for n,info in manifest.items())
                before={str(p):digest(p) for p in o.iterdir() if p.is_file()}
                meta=json.loads((o/"run_metadata.json").read_text(encoding="utf-8"))
                assert meta["status"]=="completed" and meta["exit_code"]==0
                assert meta["protected_sha256_before"]==meta["protected_sha256_after"]
                day02_path=ROOT/"week09/day02/outputs/proposal_visualization_run01/run_metadata.json"
                day02_meta=json.loads(day02_path.read_text(encoding="utf-8"))
                shared={p:h for p,h in day02_meta["input_sha256_before"].items()
                        if p in meta["protected_sha256_before"]}
                assert shared and all(meta["protected_sha256_before"][p]==h for p,h in shared.items())
                selected_input_keys=[p for p in shared if "030269" in p]
                assert len(selected_input_keys)==3
                info={"status":"passed","scope":"saved_single_030269_G1_offline_only",
                      "model_forward_performed":False,"checkpoint_loaded":False,"models":{},
                      "actual_command":subprocess.list2cmdline(sys.orig_argv),
                      "verification_source_sha256":digest(__file__),
                      "day02_shared_pre_forward_hashes_matched":len(shared),
                      "selected_rgb_ir_xml_pre_forward_hashes_match_day02":selected_input_keys}
                max64_error=0.
                for model in ("IR","Fusion"):
                    data=torch.load(o/f"{model}_raw_capture.pth",map_location="cpu",weights_only=True)
                    assert len(data["records"])==len(data["stages"])==1
                    r,stages=data["records"][0],data["stages"][0]
                    assert r["image_size"]==(640,640)
                    assert r["layer_keys"]==("0","1","pool")
                    exact(data["observed"]["rpn_forward_boxes"],r["final_boxes"])
                    assert data["observed"]["model_forward_calls"]==data["observed"]["rpn_calls"]==data["observed"]["transform_calls"]==1
                    sums=read_csv(o/f"{model}_stage_summary.csv")
                    assert len(sums)==7
                    ids=torch.arange(len(r["boxes"]));levels=torch.repeat_interleave(torch.arange(3),torch.tensor(r["counts"]))
                    exact(stages["decoded"]["ids"],ids);exact(stages["decoded"]["boxes"],r["boxes"])
                    top=r["top_indices"];exact(stages["pre_nms_topn"]["ids"],ids[top])
                    assert all(int((levels[top]==i).sum())==min(150,n) for i,n in enumerate(r["counts"]))
                    exact(stages["clipped"]["ids"],ids[top])
                    exact(stages["clipped"]["boxes"],clip_boxes_to_image(r["boxes"][top],r["image_size"]))
                    old=stages["pre_nms_topn"];cur=stages["clipped"]
                    assert bool((area_iou64(cur["boxes"],r["gt"])+1e-6>=area_iou64(old["boxes"],r["gt"])).all())
                    idx=remove_small_boxes(cur["boxes"],r["config"]["min_size"])
                    exact(stages["small_box_filtered"]["ids"],cur["ids"][idx])
                    old=stages["small_box_filtered"];idx=torch.where(old["scores"]>=r["config"]["score_thresh"])[0]
                    exact(stages["score_filtered"]["ids"],old["ids"][idx])
                    current=stages["score_filtered"]
                    exact(current["boxes"],r["nms_boxes"]);exact(current["scores"],r["nms_scores"])
                    exact(levels[current["ids"]],r["nms_levels"])
                    exact(stages["layer_nms"]["ids"],current["ids"][r["nms_indices"]])
                    exact(stages["post_nms_topn"]["ids"],stages["layer_nms"]["ids"][:150])
                    exact(stages["post_nms_topn"]["boxes"],r["final_boxes"])
                    exact(stages["post_nms_topn"]["scores"],r["final_scores"])
                    previous=None
                    for row in sums:
                        s=stages[row["stage"]]
                        f32=box_iou(r["gt"].float()[None],s["boxes"].float())[0]
                        diagnostic=area_iou64(s["boxes"],r["gt"])
                        err=float((diagnostic-s["iou"]).abs().max()) if len(diagnostic) else 0.
                        max64_error=max(max64_error,err);assert err<=1e-12
                        assert int(row["candidate_count"])==len(s["ids"])
                        assert float(row["max_iou_float32_cpu"])==(float(f32.max()) if len(f32) else 0.)
                        assert int(row["count_iou_ge_075_float32_cpu"])==int((f32>=.75).sum())
                        assert abs(float(row["max_iou_float64_diagnostic"])-(float(diagnostic.max()) if len(diagnostic) else 0.))<1e-12
                        assert int(row["count_iou_ge_075_float64_diagnostic"])==int((diagnostic>=.75).sum())
                        if previous and row["stage"]!="clipped":
                            assert float(row["max_iou_float32_cpu"])<=float(previous["max_iou_float32_cpu"])+1e-6
                            assert int(row["count_iou_ge_075_float32_cpu"])<=int(previous["count_iou_ge_075_float32_cpu"])
                        previous=row
                    comparisons=json.loads((o/f"{model}_historical_comparison.json").read_text(encoding="utf-8"))
                    assert comparisons["actual_difference"]==0
                    assert comparisons["restoration_verified"] and comparisons["config_unchanged"]
                    assert comparisons["state_sha256_before"]==comparisons["state_sha256_after"]
                    assert comparisons["checkpoint_sha256_before"]==comparisons["checkpoint_sha256_after"]
                    losses=[{"from":a["stage"],"to":b["stage"]} for a,b in zip(sums,sums[1:])
                            if int(a["count_iou_ge_075_float32_cpu"])>0 and int(b["count_iou_ge_075_float32_cpu"])==0]
                    # Every exported tracked identity must agree with raw membership and coordinates.
                    rows=read_csv(o/f"{model}_candidate_trajectory.csv")
                    for row in rows:
                        identity=int(row["flat_index"]);s=stages[row["stage"]]
                        hit=torch.where(s["ids"]==identity)[0]
                        assert (row["kept"]=="True")==bool(len(hit))
                        if len(hit):assert json.loads(row["xyxy"])==s["boxes"][int(hit[0])].tolist()
                    initial_high=stages["decoded"]["ids"][stages["decoded"]["iou"]>=.75].tolist()
                    histories={}
                    for identity in initial_high:
                        histories[str(identity)]={"level":r["layer_keys"][int(levels[identity])],
                            "first_removed":data["first_removed"][0].get(identity),
                            "track":[{"stage":row["stage"],"kept":row["kept"],"score":row["score"],
                                      "iou_f32":row["iou_float32_cpu"],"iou_f64":row["iou_float64_diagnostic"]}
                                     for row in rows if int(row["flat_index"])==identity]}
                    info["models"][model]={"forward_calls":1,"stage_reconstruction_verified":True,
                        "day02_iou_difference":0.,"first_all_high_iou_loss":losses[0] if losses else None,
                        "high_iou_initial_identities":histories}
                    print(model,"ALL_CHECKS_PASSED",json.dumps(info["models"][model],ensure_ascii=True))
                protected=json.loads((HERE/"checks/local_unit_protected_before.json").read_text())
                assert all(digest(p)==h for p,h in protected.items())
                assert all(digest(p)==h for p,h in before.items())
                info.update(max_float64_cpu_vs_saved_diagnostic_error=max64_error,float64_atol=1e-12,
                            historical_files_unchanged=len(protected),server_files_sha256_verified=len(manifest),
                            original_input_sha256_before_and_after=before)
                with (dest/"verification.json").open("x",encoding="utf-8") as f:json.dump(info,f,indent=2,ensure_ascii=False)
                print("OFFLINE_ACCEPTANCE_PASSED: original server files unchanged; no model forward")
                code=0
            except BaseException:
                print(traceback.format_exc());code=1
            with (dest/"execution_status.json").open("x",encoding="utf-8") as f:json.dump({"exit_code":code},f)
    return code
if __name__=="__main__":sys.exit(main())

