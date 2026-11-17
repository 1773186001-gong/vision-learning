"""Independently read exported rows against original capture indices and coordinates; no NMS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
sys.dont_write_bytecode=True
import torch
from torchvision.ops import box_iou

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"outputs/remaining_five_merged_run01"
FIXED=(("030269",1,"Fusion","reused_030269_G1",(2201,)),
       ("130126",3,"IR","new_130126_G3",(3216,3516,3521,9216,9516)))

def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def numeric_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=HERE/"results")
    p.add_argument("--output",type=Path,default=HERE/"independent_verification")
    args=p.parse_args();dest=args.output.resolve()
    assert dest.is_relative_to(HERE)
    dest.mkdir(exist_ok=False)
    before={str(q):digest(q) for q in args.results.rglob("*") if q.is_file()}
    code=1;info={"status":"failed","command":subprocess.list2cmdline(sys.orig_argv),
        "model_forward_performed":False,"checkpoint_loaded":False,"nms_executed":False}
    try:
        audit=rows(args.results/"pair_audit.csv");related=rows(args.results/"possible_related_retained.csv")
        assert len(audit)==960 and len({(r["deleted_candidate_id"],r["retained_candidate_id"]) for r in audit})==960
        assert [r for r in audit if r["possible_related_retained"]=="True"]==related
        assert [r for r in audit if r["near_threshold_diagnostic_only"]=="True"]==rows(args.results/"threshold_near_pairs.csv")
        maxerror=0.;geometry=[]
        for stem,gi,model,sub,ids in FIXED:
            d=torch.load(SOURCE/sub/f"{model}_raw_capture.pth",weights_only=True,map_location="cpu")
            rec=d["records"][0];pre=d["stages"][0]["score_filtered"]
            positions={int(ident):i for i,ident in enumerate(pre["ids"].tolist())}
            keep=rec["nms_indices"].tolist();keep_positions={i:k for k,i in enumerate(keep)}
            assert len(keep)==(105 if stem=="030269" else 171)
            case_rows=[r for r in audit if (r["stem"],r["model"])==(stem,model)]
            assert len(case_rows)==len(ids)*len(keep)
            boxes=rec["nms_boxes"];scores=rec["nms_scores"];levels=rec["nms_levels"]
            shifted=boxes+((boxes.max()+1)*levels.to(boxes))[:,None]
            inventory=rows(args.results/"retained_inventory.csv")
            inv=[r for r in inventory if (r["stem"],r["model"])==(stem,model)]
            assert [int(r["flat_index"]) for r in inv]==pre["ids"][rec["nms_indices"]].tolist()
            assert sum(r["passes_final_budget"]=="False" for r in inv)==max(0,len(keep)-150)
            for row in case_rows:
                deleted=int(row["deleted_flat_index"]);retained=int(row["retained_flat_index"])
                assert deleted in ids
                i,j=positions[deleted],positions[retained]
                assert i not in keep_positions and j in keep_positions
                assert row["deleted_candidate_id"]==d["catalogues"][0][deleted]["candidate_id"]
                assert row["retained_candidate_id"]==d["catalogues"][0][retained]["candidate_id"]
                assert int(row["deleted_nms_input_position_0based"])==i
                assert int(row["retained_nms_input_position_0based"])==j
                assert row["deleted_nms_output_position_0based"]==""
                assert int(row["retained_nms_output_position_0based"])==keep_positions[j]
                assert (row["retained_passes_final_budget"]=="True")==bool(keep_positions[j]<len(rec["final_boxes"]))
                for role,pos in (("deleted",i),("retained",j)):
                    assert json.loads(row[f"{role}_xyxy"])==boxes[pos].tolist()
                    assert float(row[f"{role}_objectness"])==float(scores[pos])
                    assert row[f"{role}_level_key"]==rec["layer_keys"][int(levels[pos])]
                    f32=float(box_iou(boxes[pos:pos+1],rec["gt"][None])[0,0])
                    f64=numeric_iou(boxes[pos].tolist(),rec["gt"].tolist())
                    assert float(row[f"{role}_specified_gt_iou_float32_cpu"])==f32
                    maxerror=max(maxerror,abs(float(row[f"{role}_specified_gt_iou_float64_cpu_diagnostic"])-f64))
                v32=box_iou(boxes[i:i+1],boxes[j:j+1])[0,0]
                v64=numeric_iou(boxes[i].tolist(),boxes[j].tolist())
                off32=box_iou(shifted[i:i+1],shifted[j:j+1])[0,0]
                assert float(row["mutual_iou_float32_cpu"])==float(v32)
                assert float(row["mutual_iou_shifted_float32_cpu_diagnostic"])==float(off32)
                maxerror=max(maxerror,abs(float(row["mutual_iou_float64_cpu_diagnostic"])-v64))
                possible=bool(levels[i]==levels[j] and scores[j]>=scores[i]) and (bool(v32>.7) or v64>.7 or bool(off32>.7))
                assert (row["possible_related_retained"]=="True")==possible
                assert (row["score_tie"]=="True")==bool(scores[i]==scores[j])
            for row in rows(args.results/"gt_overlaps.csv"):
                if (row["stem"],row["model"])!=(stem,model):continue
                pos=positions[int(row["flat_index"])];index=int(row["compared_gt_index"])
                g=d["observed"]["transform_gt_all"][index]
                assert json.loads(row["compared_gt_xyxy"])==g.tolist()
                assert float(row["iou_float32_cpu"])==float(box_iou(boxes[pos:pos+1],g[None])[0,0])
                maxerror=max(maxerror,abs(float(row["iou_float64_cpu_diagnostic"])-numeric_iou(boxes[pos].tolist(),g.tolist())))
            for flat in sorted({int(r["retained_flat_index"]) for r in related if (r["stem"],r["model"])==(stem,model)}):
                b=boxes[positions[flat]].double();g=rec["gt"].double()
                delta=(b[:2]+b[2:]-g[:2]-g[2:])/2
                geometry.append({"stem":stem,"gt_index":gi,"model":model,"retained_flat_index":flat,
                    "dx_right_positive":float(delta[0]),"dy_down_positive":float(delta[1]),
                    "width_ratio_to_gt":float((b[2]-b[0])/(g[2]-g[0])),
                    "height_ratio_to_gt":float((b[3]-b[1])/(g[3]-g[1]))})
        assert maxerror<=1e-12
        assert before=={str(q):digest(q) for q in args.results.rglob("*") if q.is_file()}
        m=json.loads((args.results/"run_metadata.json").read_text(encoding="utf-8"))
        assert m["protected_sha256_before"]==m["protected_sha256_after"]
        assert all(digest(Path(q))==h for q,h in m["protected_sha256_before"].items())
        info.update(status="passed",pair_rows_verified=960,possible_rows_verified=len(related),
            max_binary64_error=maxerror,binary64_atol=1e-12,retained_frame_geometry=geometry,
            historical_files_unchanged=len(m["protected_sha256_before"]),results_files_unchanged=len(before))
        code=0
    except BaseException:info["error"]=traceback.format_exc()
    info["exit_code"]=code
    with (dest/"verification.json").open("x",encoding="utf-8") as f:json.dump(info,f,ensure_ascii=False,indent=2)
    text=json.dumps(info,ensure_ascii=False,indent=2)
    with (dest/"run.log").open("x",encoding="utf-8") as f:f.write(text+"\n")
    print(text)
    return code
if __name__=="__main__":sys.exit(main())
