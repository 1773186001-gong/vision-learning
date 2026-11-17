"""Day02: six selected validation GTs, original post-RPN boxes and sigmoid objectness."""
from __future__ import annotations
import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "week09/day01"))
import rpn_diagnostic as day1
import torch
import torchvision
import numpy as np
from torchvision.ops import box_iou
from torchvision.models.detection.rpn import RegionProposalNetwork
from PIL import Image

K_FIELD = "max_iou_top300"
BOX_FIELDS = ("xmin", "ymin", "xmax", "ymax")

def save_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def selection(path):
    rows = read_csv(path)
    required = {"model", "stem", "scene_id", "gt_index", "image_index", K_FIELD,
                "max_iou_top1000", *BOX_FIELDS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Day01 CSV columns differ: inspect before selecting")
    pairs = {}
    for row in rows:
        key = (row["stem"], int(row["gt_index"]))
        if row["model"] not in ("IR", "Fusion"):
            raise ValueError("Unexpected model")
        group = pairs.setdefault(key, {})
        if row["model"] in group:
            raise ValueError("Duplicate model/GT")
        group[row["model"]] = row
    values = []
    for (stem, gi), pair in pairs.items():
        if set(pair) != {"IR", "Fusion"}:
            raise ValueError("Unpaired GT")
        a, b = pair["IR"], pair["Fusion"]
        if any(a[k] != b[k] for k in ("scene_id", "image_index", *BOX_FIELDS)):
            raise ValueError("GT geometry/order differs")
        if any(float(pair[m][K_FIELD]) != float(pair[m]["max_iou_top1000"]) for m in pair):
            raise ValueError("Day01 all-proposal equivalence not satisfied")
        values.append({"stem": stem, "gt_index": gi, "scene_id": a["scene_id"],
                       "image_index": int(a["image_index"]),
                       "IR_day01_max_iou": float(a[K_FIELD]),
                       "Fusion_day01_max_iou": float(b[K_FIELD]),
                       **{f"gt_{k}": float(a[k]) for k in BOX_FIELDS}})
    if len(values) != 416:
        raise ValueError("Expected the original 416 validation GT pairs")
    ir = sorted([v for v in values if v["IR_day01_max_iou"] >= .75 > v["Fusion_day01_max_iou"]],
                key=lambda v: (v["scene_id"] != "03",
                               -(v["IR_day01_max_iou"] - v["Fusion_day01_max_iou"]),
                               v["stem"], v["gt_index"]))
    fusion = sorted([v for v in values if v["Fusion_day01_max_iou"] >= .75 > v["IR_day01_max_iou"]],
                    key=lambda v: (-(v["Fusion_day01_max_iou"] - v["IR_day01_max_iou"]),
                                   v["stem"], v["gt_index"]))
    chosen = []
    for category, pool, count, reason in (
        ("IR_advantage", ir, 3, "IR>=0.75>Fusion; scene03 first, then descending IR-Fusion IoU"),
        ("Fusion_advantage", fusion, 2, "Fusion>=0.75>IR; descending Fusion-IR IoU")):
        for v in pool[:count]:
            chosen.append({**v, "category": category, "selection_reason": reason})
    used = {(v["stem"], v["gt_index"]) for v in chosen}
    boundary = [v for v in values if (v["stem"], v["gt_index"]) not in used
                and any(.70 <= v[f"{m}_day01_max_iou"] <= .80 for m in ("IR", "Fusion"))]
    boundary.sort(key=lambda v: (
        not ((v["IR_day01_max_iou"] >= .75) != (v["Fusion_day01_max_iou"] >= .75)),
        min(abs(v["IR_day01_max_iou"] - .75), abs(v["Fusion_day01_max_iou"] - .75)),
        v["stem"], v["gt_index"]))
    for v in boundary[:1]:
        chosen.append({**v, "category": "boundary",
                       "selection_reason": "unused GT; one side in [0.70,0.80]; crossing first, then min distance to 0.75"})
    for i, v in enumerate(chosen, 1):
        v["case_id"] = f"{i:02d}_{v['category']}_{v['stem']}_G{v['gt_index']}"
    return chosen, {"IR_advantage": len(ir), "Fusion_advantage": len(fusion),
                    "unused_boundary": len(boundary), "selected": len(chosen),
                    "pairing": "stem+0-based gt_index",
                    "ranking_source": K_FIELD,
                    "tie_breaking": "stem ascending, then gt_index ascending; no duplicates"}

class CaptureScores(day1.CaptureRPN):
    """Call the exact original bound method, return its output untouched, restore in finally."""
    def reset(self):
        super().reset()
        self.filter_calls = 0
        self.filter_boxes = None
        self.filter_scores = None

    def __enter__(self):
        self.had_instance_override = "filter_proposals" in self.detector.rpn.__dict__
        self.old_override = self.detector.rpn.__dict__.get("filter_proposals")
        self.original = self.detector.rpn.filter_proposals
        def readonly_wrapper(*args, **kwargs):
            output = self.original(*args, **kwargs)
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("filter_proposals return structure differs")
            boxes, scores = output
            if len(boxes) != len(scores):
                raise ValueError("Box/score batch mismatch")
            self.filter_calls += 1
            self.filter_boxes = [x.detach().cpu().clone() for x in boxes]
            self.filter_scores = [x.detach().cpu().clone() for x in scores]
            for b, s in zip(self.filter_boxes, self.filter_scores):
                if s.ndim != 1 or len(b) != len(s) or not bool(torch.isfinite(s).all()):
                    raise ValueError("Invalid returned scores")
                if bool(((s < 0) | (s > 1)).any()) or bool((s[1:] > s[:-1]).any()):
                    raise ValueError("Expected descending sigmoid objectness")
            return output
        self.detector.rpn.filter_proposals = readonly_wrapper
        try:
            return super().__enter__()
        except BaseException:
            self.restore_method()
            for handle in self.handles:
                handle.remove()
            raise

    def restore_method(self):
        if self.had_instance_override:
            self.detector.rpn.filter_proposals = self.old_override
        else:
            delattr(self.detector.rpn, "filter_proposals")
        assert self.detector.rpn.filter_proposals == self.original

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.restore_method()

def hashes(paths):
    return {str(p.resolve()): day1.ir.sha(p) for p in sorted(set(paths))}

def geometry(gt, proposal):
    gx1, gy1, gx2, gy2 = gt
    gcx, gcy, gw, gh = (gx1+gx2)/2, (gy1+gy2)/2, gx2-gx1, gy2-gy1
    result = dict(gt_center_x=gcx, gt_center_y=gcy, gt_width=gw, gt_height=gh)
    if proposal is None:
        return {**result, **{k: None for k in ("proposal_center_x", "proposal_center_y",
                "dx", "dy", "dx_norm", "dy_norm", "proposal_width", "proposal_height",
                "width_ratio", "height_ratio", "width_error", "height_error",
                "left_edge_error", "top_edge_error", "right_edge_error", "bottom_edge_error")}}
    x1, y1, x2, y2 = proposal
    cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
    return {**result, "proposal_center_x": cx, "proposal_center_y": cy,
            "dx": cx-gcx, "dy": cy-gcy, "dx_norm": (cx-gcx)/gw, "dy_norm": (cy-gcy)/gh,
            "proposal_width": w, "proposal_height": h, "width_ratio": w/gw, "height_ratio": h/gh,
            "width_error": w-gw, "height_error": h-gh,
            "left_edge_error": x1-gx1, "top_edge_error": y1-gy1,
            "right_edge_error": x2-gx2, "bottom_edge_error": y2-gy2}

@torch.inference_mode()
def forward_selected(name, args, dataset, chosen, baseline):
    checkpoint = args.ir_checkpoint if name == "IR" else args.fusion_checkpoint
    sha_before = day1.ir.sha(checkpoint)
    if sha_before != baseline["models"][name]["checkpoint"]["sha256_before"]:
        raise ValueError(f"{name}: checkpoint differs from Day01")
    restore = day1.ir.checkpoint_model if name == "IR" else day1.fusion.restore
    model, ck = restore(checkpoint, args.device)
    model.eval()
    if ck["config"]["manifest_sha256"] != day1.ir.sha(args.data_root/"manifest.csv"):
        raise ValueError("Training manifest differs")
    del ck
    detector = model if name == "IR" else model.detector
    config = day1.rpn_configuration(detector)
    for k in ("pre_nms_top_n", "post_nms_top_n", "score_thresh", "nms_thresh",
              "transform_min_size", "transform_max_size", "transform_mean", "transform_std",
              "anchor_sizes", "anchor_aspect_ratios", "roi_feature_keys"):
        if json.loads(json.dumps(config[k])) != baseline["models"][name]["rpn_config"][k]:
            raise ValueError(f"{name}: original configuration changed: {k}")
    state_before = day1.tensor_digest(model)
    results, raw, details, geometries = {}, {}, [], []
    original_ir = day1.ir.loader(args.data_root, "val").dataset if name == "IR" else None
    with CaptureScores(detector) as capture:
        for index in sorted({v["image_index"] for v in chosen}):
            image, target = dataset[index]
            record = dataset.get_record(index)
            cases = [v for v in chosen if v["image_index"] == index]
            assert all(v["stem"] == record.stem for v in cases)
            if original_ir is not None:
                im_ir, t_ir = original_ir[index]
                torch.testing.assert_close(im_ir, image["infrared"], rtol=0, atol=0)
                for key in target:
                    torch.testing.assert_close(target[key], t_ir[key], rtol=0, atol=0)
            moved = ([image["infrared"].to(args.device)] if name == "IR" else
                     [{k: v.to(args.device) for k, v in image.items()}])
            moved_targets = [{k: v.to(args.device) for k, v in target.items()}]
            capture.reset()
            model(moved, moved_targets)
            if capture.transform_calls != 1 or capture.rpn_calls != 1 or capture.filter_calls != 1:
                raise ValueError("Expected one original transform/filter/RPN call")
            sizes, gt_batch = capture.transformed
            assert sizes == [(640, 640)] and len(capture.proposals) == 1
            gt, proposals, scores = gt_batch[0], capture.proposals[0], capture.filter_scores[0]
            torch.testing.assert_close(proposals, capture.filter_boxes[0], rtol=0, atol=0)
            torch.testing.assert_close(gt, target["boxes"], rtol=0, atol=0)
            day1.check_boxes(gt, sizes[0], "GT")
            day1.check_boxes(proposals, sizes[0], "proposals")
            assert len(proposals) <= config["active_post_nms_top_n"]
            raw[record.stem] = {"image_index": index, "image_size_hw": sizes[0],
                                "gt_boxes": gt.tolist(), "proposal_boxes": proposals.tolist(),
                                "objectness_sigmoid": scores.tolist()}
            for case in cases:
                gi = case["gt_index"]
                g = gt[gi]
                torch.testing.assert_close(g, torch.tensor([case[f"gt_{k}"] for k in BOX_FIELDS]), rtol=0, atol=0)
                overlaps = box_iou(g[None], proposals)[0]
                bi = int(overlaps.argmax()) if len(proposals) else None
                best_iou = float(overlaps[bi]) if bi is not None else 0.0
                box = proposals[bi].tolist() if bi is not None else None
                score = float(scores[bi]) if bi is not None else None
                old_iou = case[f"{name}_day01_max_iou"]
                delta = best_iou-old_iou
                changed = (best_iou >= .75) != (old_iou >= .75)
                detail = {"case_id": case["case_id"], "category": case["category"], "model": name,
                          "stem": record.stem, "scene_id": record.scene_id, "gt_index": gi,
                          "proposal_count": len(proposals), "best_proposal_index_0based": bi,
                          "objectness_rank_1based": bi+1 if bi is not None else None,
                          "best_iou": best_iou, "best_objectness": score,
                          "day01_iou": old_iou, "day02_minus_day01_iou": delta,
                          "coverage_at_0.75_changed": changed,
                          **{f"gt_{k}": float(g[j]) for j, k in enumerate(BOX_FIELDS)},
                          **{f"proposal_{k}": box[j] if box else None for j, k in enumerate(BOX_FIELDS)}}
                geo = {**detail, **geometry(g.tolist(), box)}
                results[case["case_id"]] = geo
                details.append(detail)
                geometries.append(geo)
                print("CASE", json.dumps(detail, ensure_ascii=False), flush=True)
                if args.device == "cuda:0" and (abs(delta) > 1e-6 or changed):
                    raise ValueError("Same-GPU Day01 consistency failed; preserve logs, do not relabel cases")
    state_after = day1.tensor_digest(model)
    assert state_before == state_after
    assert config == day1.rpn_configuration(detector)
    assert sha_before == day1.ir.sha(checkpoint)
    print(f"MODEL_DONE model={name} images={len(raw)} targets={len(details)} state_unchanged=True wrapper_restored=True", flush=True)
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha_before,
                "state_sha256_before": state_before, "state_sha256_after": state_after,
                "rpn_config": config, "wrapper_restored": True, "processed_images": len(raw)}
    del model, detector
    if args.device == "cuda:0":
        torch.cuda.empty_cache()
    return results, raw, details, geometries, metadata

def draw_boxes(ax, gt, prop):
    from matplotlib.patches import Rectangle
    for box, color, style, label in ((gt, "#00a83b", "-", "GT"), (prop, "#1976d2", "--", "Best RPN proposal")):
        if box is not None:
            x1,y1,x2,y2 = box
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linestyle=style,linewidth=2,label=label))

def make_figure(path, case, dataset, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    record = dataset.get_record(case["image_index"])
    image, target = dataset[case["image_index"]]
    gt = target["boxes"][case["gt_index"]].tolist()
    shared = image["infrared"].permute(1,2,0).numpy()
    boxes = [[records[m][f"proposal_{k}"] for k in BOX_FIELDS]
             for m in ("IR","Fusion") if records[m]["proposal_xmin"] is not None]
    all_boxes = np.asarray([gt]+boxes)
    x1, y1 = all_boxes[:,:2].min(axis=0)
    x2, y2 = all_boxes[:,2:].max(axis=0)
    margin_x, margin_y = max((x2-x1)*.45,12), max((y2-y1)*.25,12)
    crop = (max(0,x1-margin_x),min(640,x2+margin_x),max(0,y1-margin_y),min(640,y2+margin_y))
    fig, axes = plt.subplots(2,2,figsize=(14,12))
    with Image.open(record.visible_path) as im:
        axes[0,0].imshow(im.convert("RGB"))
    with Image.open(record.infrared_path) as im:
        axes[0,1].imshow(im.convert("L"),cmap="gray",vmin=0,vmax=255)
    axes[0,0].set_title("Original RGB (1280 x 1024; no coordinate overlay)")
    axes[0,1].set_title("Original IR (1280 x 1024; no coordinate overlay)")
    for ax in axes[0]:
        ax.axis("off")
    for ax, model in zip(axes[1], ("IR","Fusion")):
        row=records[model]
        prop=[row[f"proposal_{k}"] for k in BOX_FIELDS] if row["proposal_xmin"] is not None else None
        ax.imshow(shared)
        draw_boxes(ax, gt, prop)
        ax.set_xlim(0,640); ax.set_ylim(640,0)
        ax.set_title(f"{model}: IoU={row['best_iou']:.6f}, objectness={row['best_objectness']:.6f}" if prop
                     else f"{model}: no proposal")
        ax.set_xlabel(f"dx={row['dx']:.2f}, dy={row['dy']:.2f} px; dx/w={row['dx_norm']:+.3f}, dy/h={row['dy_norm']:+.3f}\n"
                      f"proposal/GT width={row['width_ratio']:.3f}, height={row['height_ratio']:.3f}" if prop else "NA")
        ax.legend(loc="lower left",fontsize=8)
        ax.set_ylabel("640x640 shared IR view; x right, y down")
        ix=.02 if (gt[0]+gt[2])/2>320 else .60
        iy=.02 if (gt[1]+gt[3])/2<320 else .59
        inset=ax.inset_axes([ix,iy,.38,.38])
        inset.imshow(shared)
        draw_boxes(inset,gt,prop)
        inset.set_xlim(crop[0],crop[1]); inset.set_ylim(crop[3],crop[2])
        inset.set_xticks([]);inset.set_yticks([])
        inset.text(.5,.98,"Same zoom",ha="center",va="top",transform=inset.transAxes,
                   fontsize=8,bbox={"facecolor":"white","edgecolor":"none","alpha":.9},clip_on=True)
    fig.suptitle(f"{case['case_id']} | GT index is zero-based\nBottom panels share IR background and coordinates; Fusion receives paired RGB+IR",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(path,dpi=150)
    plt.close(fig)

def write_report(out, cases, results, metadata, command):
    lines=["# Week09 Day02：RPN Proposal 定位误差可视化","",
           "只读推理完成：仅选中validation图片，未训练、未改结构/权重/原RPN预算，未运行test。","",
           "## 选择规则与坐标","",
           "按Day01逐GT表的stem+0起始gt_index配对，采用全部实际候选（max_iou_top300=top1000）。IR优势优先场景03再按差值；Fusion优势按差值；临界案例排除前面已选GT，优先跨阈值，再按任一侧距离0.75的最小值排序，平局按stem/gt_index。",
           "dx=Proposal中心x−GT中心x，正值向右；dy=Proposal中心y−GT中心y，正值向下。归一化分别除以GT宽/高；宽高比例=Proposal/GT。全部测量和底部叠框都在原Letterbox及模型transform后的640×640坐标，不将原图坐标混入。",
           "两张底图相同且放大区域相同；Fusion仍输入RGB+IR。绿色实线GT、蓝色虚线最佳RPN候选。顶部原始RGB/IR仅作场景参考，不叠加640框。","",
           "## 真实命令","", "~~~text",command,"~~~","",
           "## 每个案例",""]
    for case in cases:
        cid=case["case_id"]
        lines += [f"### {cid}","",f"选择原因：{case['selection_reason']}。场景{case['scene_id']}。","",
                  "|模型|Day01 IoU|本次IoU|objectness|最佳框xyxy|分数排名/候选总数|",
                  "|---|---:|---:|---:|---|---|"]
        for m in ("IR","Fusion"):
            r=results[m][cid]
            box=[r[f"proposal_{k}"] for k in BOX_FIELDS]
            lines.append(f"|{m}|{r['day01_iou']:.9f}|{r['best_iou']:.9f}|{r['best_objectness']:.9f}|{[round(x,6) for x in box]}|{r['objectness_rank_1based']}/{r['proposal_count']}|")
        gt=[case[f"gt_{k}"] for k in BOX_FIELDS]
        lines += ["",f"GT xyxy={gt}。","",
                  "|模型|dx|dy|dx/GT宽|dy/GT高|宽比例|高比例|上边误差|下边误差|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in ("IR","Fusion"):
            r=results[m][cid]
            lines.append("|"+m+"|"+"|".join(f"{r[k]:+.6f}" for k in ("dx","dy","dx_norm","dy_norm","width_ratio","height_ratio","top_edge_error","bottom_edge_error"))+"|")
        for m in ("IR","Fusion"):
            r=results[m][cid]
            direction_x="右" if r["dx"]>0 else "左"
            direction_y="下" if r["dy"]>0 else "上"
            lines.append("")
            lines.append(f"{m}数值观察：候选中心向{direction_x}{abs(r['dx']):.3f}px、向{direction_y}{abs(r['dy']):.3f}px；"
                         f"宽为GT的{r['width_ratio']:.3f}倍，高为{r['height_ratio']:.3f}倍。"
                         f"上/下边界相对GT误差分别为{r['top_edge_error']:+.3f}/{r['bottom_edge_error']:+.3f}px（正值向下）。")
        lines += ["",f"![{cid}](case_{cid}.png)",""]
    lines += ["## 重复偏差与Day03建议","",
              "本次6个GT按优势差值与阈值主动选取，并非随机样本，不能用此处符号计数估计整个验证集的偏差比例。"]
    for m in ("IR","Fusion"):
        rr=list(results[m].values())
        lines.append(f"- {m}：本次{len(rr)}例中候选偏窄{sum(r['width_ratio']<1 for r in rr)}例、偏矮{sum(r['height_ratio']<1 for r in rr)}例、中心向下{sum(r['dy']>0 for r in rr)}例；仅为选中样本描述。")
    lines += ["","临界案例应结合精确IoU与像素误差阅读：0.75是覆盖判定阈值，不代表几何质量在阈值两侧突然发生机制变化。",
              "objectness是RPN原filter_proposals返回的sigmoid分数，不是最终person分类置信度，也不是IoU概率；最佳框按IoU最大选取，分数排名仅作解释。",
              "Day03优先保持融合结构和原权重，追踪这些固定val案例从解码框、逐层pre-NMS筛选、score过滤、NMS到最终post预算的去留与IoU变化；再结合RPN框回归前后变化定位问题。当前证据不足以直接修改融合模块。",
              "本次只能描述RPN输出的几何差异，不能区分特征、框回归和筛选机制的因果贡献，也不能据此解释全部历史test AP75差异。","",
              "## 完整性与输出","",
              "filter_proposals临时包装调用原方法并原样返回，捕获其后处理boxes和对应scores；与RPN forward实际返回框逐位核对。finally移除hooks并恢复实例原始方法查找状态。各模型持久state和checkpoint前后SHA一致，配置与Day01一致。",
              "全部后处理框及分数另存rpn_proposals.json，最佳框在proposal_details.csv，几何量在geometry_analysis.csv；没有使用最终RoI框。",
              "环境、输入文件前后校验、实际命令和各模型配置见run_metadata.json，完成状态与退出码见execution_status.json，逐图过程见run.log。"]
    with (out/"day02_report.md").open("x",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")

def run(args):
    day1.ir.seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device=="cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but unavailable; no fallback")
    baseline=json.loads(args.day01_metrics.read_text(encoding="utf-8"))
    if baseline["run"]["status"]!="completed" or baseline["run"]["scope"]!="full_validation":
        raise ValueError("Day01 full validation evidence required")
    source=inspect.getsource(RegionProposalNetwork.filter_proposals)
    if not all(s in source for s in ("objectness_prob = torch.sigmoid(objectness)",
            "return final_boxes, final_scores", "keep = keep[: self.post_nms_top_n()]")):
        raise RuntimeError("Audit installed filter_proposals implementation before running")
    cases,pools=selection(args.day01_csv)
    if not cases:
        raise ValueError("No valid cases")
    dataset=day1.fusion.loader(args.data_root,"val",shuffle=False).dataset
    source_files=[Path(__file__),ROOT/"week09/day01/rpn_diagnostic.py",
        ROOT/"week08/day03/ir_common.py",ROOT/"week08/day05/fusion_model.py",
        ROOT/"week08/day05/fusion_common.py",ROOT/"week08/day01/llvip_dataset.py",
        ROOT/"week08/day02/paired_letterbox.py",ROOT/"week08/day02/paired_horizontal_flip.py",
        args.data_root/"manifest.csv",Path(inspect.getfile(RegionProposalNetwork))]
    protected=list((ROOT/"week09/day01").rglob("*"))
    protected=[p for p in protected if p.is_file() and "__pycache__" not in p.parts]+source_files+[args.ir_checkpoint,args.fusion_checkpoint]
    for index in sorted({c["image_index"] for c in cases}):
        rec=dataset.get_record(index)
        protected += [rec.visible_path,rec.infrared_path,rec.annotation_path]
    before=hashes(protected)
    for suffix in ["week09/day01/rpn_diagnostic.py","week08/day03/ir_common.py",
                   "week08/day05/fusion_model.py","week08/day05/fusion_common.py",
                   "week08/day01/llvip_dataset.py","week08/day02/paired_letterbox.py",
                   "week08/day02/paired_horizontal_flip.py","data/LLVIP_1000/manifest.csv"]:
        old=next(v for k,v in baseline["inputs"]["file_sha256"].items() if k.replace("\\","/").endswith(suffix))
        assert day1.ir.sha(ROOT/suffix)==old, suffix
    day1.write_csv(args.output/"selected_cases.csv",cases)
    command=subprocess.list2cmdline(sys.orig_argv)
    metadata={"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"command":command,
              "environment":{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
                 "hostname":platform.node(),"torch":torch.__version__,"torchvision":torchvision.__version__,
                 "numpy":np.__version__,"device":args.device,
                 "gpu":torch.cuda.get_device_name(0) if args.device=="cuda:0" else None,
                 "cudnn_deterministic":torch.backends.cudnn.deterministic,
                 "cudnn_benchmark":torch.backends.cudnn.benchmark,
                 "seed":args.seed,"threads":args.threads},
              "selection_pools":pools,"selection_order":[c["case_id"] for c in cases],
              "inference_order":"unique selected image_index ascending; IR then Fusion",
              "filter_proposals_source":source,"input_sha256_before":before,"models":{},
              "training":False,"test_inference":False,"rpn_budget_changed":False}
    print("COMMAND",command,flush=True)
    print("ENVIRONMENT",json.dumps(metadata["environment"]),flush=True)
    print("SELECTION",json.dumps(cases,ensure_ascii=False),flush=True)
    results,raw,details,geos={},{},[],[]
    for name in ("IR","Fusion"):
        a,b,c,d,e=forward_selected(name,args,dataset,cases,baseline)
        results[name]=a;raw[name]=b;details+=c;geos+=d;metadata["models"][name]=e
    day1.write_csv(args.output/"proposal_details.csv",details)
    day1.write_csv(args.output/"geometry_analysis.csv",geos)
    save_json(args.output/"rpn_proposals.json",raw)
    for case in cases:
        target=args.output/f"case_{case['case_id']}.png"
        if target.exists():
            raise FileExistsError(target)
        make_figure(target,case,dataset,{m:results[m][case["case_id"]] for m in results})
        print("FIGURE",target.name,flush=True)
    after=hashes(protected)
    assert before==after,"Protected input file changed"
    metadata["input_sha256_after"]=after
    metadata["protected_inputs_unchanged"]=True
    metadata["finished_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
    write_report(args.output,cases,results,metadata,command)
    save_json(args.output/"run_metadata.json",metadata)
    print(f"COMPLETED cases={len(cases)} unique_validation_images={len(raw['IR'])} output={args.output}",flush=True)

def render_saved(args):
    """Repaint saved coordinates only; never restore a model or perform forward."""
    source=args.render_from
    protected=[p for p in source.iterdir() if p.is_file()]
    before=hashes(protected)
    cases=read_csv(source/"selected_cases.csv")
    for c in cases:
        c["gt_index"]=int(c["gt_index"]);c["image_index"]=int(c["image_index"])
    rows=read_csv(source/"geometry_analysis.csv")
    fields=["best_iou","best_objectness","dx","dy","dx_norm","dy_norm","width_ratio","height_ratio"]
    fields += [f"proposal_{k}" for k in BOX_FIELDS]
    for row in rows:
        for k in fields:row[k]=float(row[k]) if row[k] else None
    dataset=day1.fusion.loader(args.data_root,"val",shuffle=False).dataset
    for case in cases:
        records={r["model"]:r for r in rows if r["case_id"]==case["case_id"]}
        assert set(records)=={"IR","Fusion"}
        make_figure(args.output/f"case_{case['case_id']}.png",case,dataset,records)
        print("RENDERED",case["case_id"],flush=True)
    assert before==hashes(protected)
    save_json(args.output/"render_metadata.json",{
        "command":subprocess.list2cmdline(sys.orig_argv),"source_results":str(source),
        "source_sha256_before":before,"source_sha256_after":hashes(protected),
        "render_script_sha256":day1.ir.sha(Path(__file__)),"model_forward_performed":False,
        "change":"Legend moved away from inset; inset label contained within its axes",
        "cases":len(cases)})
    print("RENDER_COMPLETED no_model_forward=True",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    p.add_argument("--data-root",type=Path,default=ROOT/"data/LLVIP_1000")
    p.add_argument("--day01-csv",type=Path,default=ROOT/"week09/day01/outputs/val_rpn_run01/rpn_per_gt.csv")
    p.add_argument("--day01-metrics",type=Path,default=ROOT/"week09/day01/outputs/val_rpn_run01/rpn_metrics.json")
    p.add_argument("--ir-checkpoint",type=Path,default=ROOT/"week08/day03/outputs/ir_baseline_run01/best.pth")
    p.add_argument("--fusion-checkpoint",type=Path,default=ROOT/"week08/day05/outputs/fusion_concat_run01/best.pth")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--render-from",type=Path,help="Saved successful result directory; redraw only, no model")
    p.add_argument("--output",type=Path,required=True,help="New nonexistent output directory")
    args=p.parse_args()
    if args.threads<1:p.error("threads must be positive")
    for k in ("output","data_root","day01_csv","day01_metrics","ir_checkpoint","fusion_checkpoint"):
        setattr(args,k,getattr(args,k).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    if args.render_from is not None:args.render_from=args.render_from.resolve()
    status={"status":"running","exit_code":None}
    with (args.output/"run.log").open("x",encoding="utf-8") as log:
        with contextlib.redirect_stdout(day1.Tee(sys.stdout,log)),contextlib.redirect_stderr(day1.Tee(sys.stderr,log)):
            try:
                if args.render_from is None:
                    run(args)
                else:
                    render_saved(args)
                status={"status":"completed","exit_code":0}
            except BaseException:
                status={"status":"failed","exit_code":1}
                traceback.print_exc()
                print("FAILED: preserve this directory; no completed result claimed",flush=True)
                raise
            finally:
                status["recorded_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
                save_json(args.output/"execution_status.json",status)

if __name__=="__main__":
    main()

