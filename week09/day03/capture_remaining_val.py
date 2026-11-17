"""Exactly five remaining selected validation images; reuse accepted 030269 capture."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import datetime as dt
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import capture_single_val as single
import torch
from verify_merged_capture import verify_case, merge_report

FIXED=(("030269",1,21),("030308",1,24),("030205",0,16),
       ("170100",1,120),("130126",3,85),("170455",1,145))
REVIEWED_SINGLE_SHA="1ad565f0dcc6d4ba124177583e8d81aa484672537d7c78d90f00bde382319de4"
REVIEWED_CORE_SHA="3a14d39c4c3b045e83f1ba420bd6bb92d1867ea1b3b085e3f616b3b63ee46b60"

def selected_cases(rows):
    actual=tuple((r["stem"],int(r["gt_index"]),int(r["image_index"])) for r in rows)
    if actual!=FIXED:raise RuntimeError("Selected cases or order differs from reviewed six cases")
    return rows

def adapter(stem,gt_index):
    if (stem,gt_index) not in [(s,g) for s,g,_ in FIXED[1:]]:
        raise ValueError("Only remaining five cases may create a live adapter; 030269 is reuse only")
    spec=importlib.util.spec_from_file_location(f"remaining_{stem}_{gt_index}",HERE/"capture_single_val.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.STEM,module.GT_INDEX=stem,gt_index
    def save_capture(out,name,cap,observed):
        data={"scope":f"{stem}/G{gt_index} only","observed":module.cpu(observed),
              "records":[module.cpu(asdict(r)) for r in cap.records] if cap else [],
              "stages":[{k:module.cpu(asdict(v)) for k,v in t.stages.items()} for t in cap.traces] if cap else [],
              "catalogues":[t.catalogue for t in cap.traces] if cap else [],
              "first_removed":[t.first_removed for t in cap.traces] if cap else [],
              "clip_deltas":[t.clip_deltas for t in cap.traces] if cap else []}
        with (out/f"{name}_raw_capture.pth").open("xb") as f:torch.save(data,f)
    module.save_capture=save_capture
    return module

def reused_copy(source,dest):
    meta=json.loads((source/"run_metadata.json").read_text(encoding="utf-8"))
    if meta["status"]!="completed" or meta["exit_code"]!=0:raise RuntimeError("Accepted single capture not completed")
    if (meta["selected_case"]["stem"],int(meta["selected_case"]["gt_index"]))!=FIXED[0][:2]:
        raise RuntimeError("Wrong reused case")
    manifest=json.loads((HERE/"checks/remaining_expected_reuse_sha256.json").read_text(encoding="utf-8"))
    files=[p for p in source.iterdir() if p.is_file()]
    before=single.snapshot_files(files)
    dest.mkdir()
    for p in files:
        key=f"single_030269_run01/{p.name}"
        if key not in manifest or single.sha(p)!=manifest[key]["sha256"]:
            raise RuntimeError("Reused source transfer checksum differs: "+key)
        shutil.copy2(p,dest/p.name)
        if single.sha(dest/p.name)!=single.sha(p):raise RuntimeError("Reuse copy hash mismatch")
    if len(files)!=len(manifest):raise RuntimeError("Reused artifact count differs from accepted manifest")
    single.save_json(dest/"reuse_provenance.json",{"source":str(source),"source_sha256":before,
        "copied_sha256":single.snapshot_files([dest/p.name for p in files]),"new_model_forward_calls":0,
        "note":"Original run metadata belongs to previous run, not new inference"})
    return before

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    args=parser.parse_args();args.output=args.output.resolve();args.reuse=args.reuse.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    meta={"status":"running","scope":"five_new_full_val_images_plus_saved_030269",
          "command":subprocess.list2cmdline(sys.orig_argv),"cases":[],"new_forward_calls":0,
          "started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"training":False,"test_inference":False,
          "budgets_modified":False,"atol":1e-6,"rtol":0.,"failure_policy":"preserve; stop; no rerun"}
    protected=[];before={};code=1
    with (args.output/"run.log").open("x",encoding="utf-8") as log:
      with redirect_stdout(single.day1.Tee(sys.stdout,log)),redirect_stderr(single.day1.Tee(sys.stderr,log)):
       try:
        print("COMMAND",meta["command"],flush=True)
        if single.sha(HERE/"capture_single_val.py")!=REVIEWED_SINGLE_SHA or single.sha(HERE/"trace_rpn_stages.py")!=REVIEWED_CORE_SHA:
            raise RuntimeError("Reviewed helper/core changed")
        rows=selected_cases(single.read_csv(single.D2/"selected_cases.csv"))
        mod=adapter(*FIXED[1][:2]);baseline,_,rpn_path=mod.preflight()
        single.day1.ir.seed_all(42);torch.set_num_threads(2)
        meta["environment"]={"hostname":platform.node(),"python":sys.version,"executable":sys.executable,
            "platform":platform.platform(),"torch":torch.__version__,"torchvision":single.torchvision.__version__,
            "gpu":torch.cuda.get_device_name(0),"cudnn":torch.backends.cudnn.version(),"seed":42,"threads":2,
            "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark}
        dataset=single.day1.fusion.loader(single.ROOT/"data/LLVIP_1000","val",shuffle=False).dataset
        ir_dataset=single.day1.ir.loader(single.ROOT/"data/LLVIP_1000","val").dataset
        roots=[single.ROOT/"week08",single.ROOT/"week09/day01",single.ROOT/"week09/day02",HERE]
        protected=[p for root in roots for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and not p.is_relative_to(args.output)]
        for row in rows:
            rec=dataset.get_record(int(row["image_index"]))
            if rec.stem!=row["stem"]:raise RuntimeError("Dataset index mismatch")
            for p in (rec.visible_path,rec.infrared_path,rec.annotation_path):
                if single.sha(p)!=single.source_file(str(p.relative_to(single.ROOT)).replace("\\","/"),baseline):
                    raise RuntimeError("Input differs from Day02: "+str(p))
                protected.append(p)
        protected.extend([single.ROOT/"data/LLVIP_1000/manifest.csv",rpn_path])
        before=single.snapshot_files(protected);meta["protected_sha256_before"]=before
        meta["source_sha256"]={str(p):single.sha(p) for p in HERE.glob("*.py")}
        meta["selected_cases"]=rows
        print("PREFLIGHT_PASSED",json.dumps(meta["environment"]),"protected=",len(before),flush=True)
        reused=args.output/"reused_030269_G1";reused_copy(args.reuse,reused)
        for name in ("IR","Fusion"):verify_case(reused,rows[0],name)
        meta["cases"].append({"stem":"030269","gt_index":1,"directory":reused.name,"new_forwards":0,"status":"reused_verified"})
        for row in rows[1:]:
            stem,gi=row["stem"],int(row["gt_index"])
            out=args.output/f"new_{stem}_G{gi}";out.mkdir()
            case={"stem":stem,"gt_index":gi,"image_index":int(row["image_index"]),"directory":out.name,
                  "new_forwards":0,"status":"running","models":{}}
            meta["cases"].append(case)
            module=adapter(stem,gi)
            case_baseline,selected,_=module.preflight()
            image,target=dataset[int(row["image_index"])]
            ir_image,ir_target=ir_dataset[int(row["image_index"])]
            single.core.same(image["infrared"],ir_image,"paired/IR input equivalence")
            for k in target:single.core.same(target[k],ir_target[k],"paired/IR target")
            case_args=argparse.Namespace(output=out,device=args.device)
            print(f"CASE_BEGIN {stem}/G{gi} image_index={row['image_index']}",flush=True)
            try:
                for name in ("IR","Fusion"):
                    record,trace=module.capture_model(name,case_args,image,target,selected,case_baseline,case)
                    module.export_trace(out,name,record,trace)
                    verify_case(out,row,name)
                case["status"]="completed"
            finally:
                case["new_forwards"]=sum(m.get("forward_calls",0) for m in case["models"].values())
                meta["new_forward_calls"]=sum(c["new_forwards"] for c in meta["cases"])
                single.save_json(out/"case_metadata.json",case)
            print(f"CASE_VERIFIED {stem}/G{gi} new_forwards={case['new_forwards']}",flush=True)
        if meta["new_forward_calls"]!=10:raise RuntimeError("Expected exactly 10 newly observed original eval forwards")
        # A failed invariant produces failure records and no interpretation report.
        after=single.snapshot_files(protected)
        if before!=after:raise RuntimeError("Protected source/input/artifacts changed")
        summary=merge_report(args.output,meta["cases"],rows)
        single.save_json(args.output/"merged_verification.json",summary)
        meta["status"]="completed";code=0
        print("COMPLETED new_original_eval_forwards=10 reused_original_eval_forwards=0 verified_gt_model_items=12",flush=True)
       except BaseException:
        meta["status"]="failed";meta["failure"]=traceback.format_exc();traceback.print_exc()
        single.save_json(args.output/"failure.json",{"error":meta["failure"],"attribution_allowed":False,"no_automatic_rerun":True})
       finally:
        if before:
            meta["protected_sha256_after"]=single.snapshot_files(protected)
            meta["protected_files_unchanged"]=before==meta["protected_sha256_after"]
            if not meta["protected_files_unchanged"]:meta["status"]="failed";code=1
        meta["exit_code"]=code;meta["finished_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
        single.save_json(args.output/"run_metadata.json",meta)
        single.save_json(args.output/"execution_status.json",{"status":meta["status"],"exit_code":code,"new_forward_calls":meta["new_forward_calls"]})
    return code

if __name__=="__main__":sys.exit(main())
