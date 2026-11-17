"""GT-level offline stage statistics. No model construction, checkpoint loading or NMS."""
from collections import defaultdict
import json
import torch
from torchvision.ops import box_iou

STAGES=("decoded","pre_nms_topn","clipped","small_box_filtered","score_filtered","layer_nms","post_nms_topn")
CONVENTIONS=("float32_cpu","float64_diagnostic")

def ratio(n,d):return n/d if d else None

def measure_image(model,stem,scene_id,image_index,gt,stages,catalogue,image_size):
    if gt.ndim!=2 or gt.shape[1]!=4 or gt.dtype!=torch.float32 or gt.device.type!="cpu":
        raise ValueError("Expected CPU float32 complete GT [G,4]")
    h,w=image_size
    if not bool(torch.isfinite(gt).all()) or bool((gt[:,2:]<=gt[:,:2]).any()):raise ValueError("Invalid GT; no silent exclusion")
    if bool((gt<0).any()) or bool((gt[:,[0,2]]>w).any()) or bool((gt[:,[1,3]]>h).any()):raise ValueError("GT outside image")
    if tuple(stages)!=STAGES:raise ValueError("Expected all seven ordered stages")
    image={"model":model,"stem":stem,"scene_id":scene_id,"image_index":image_index,
           "gt_count":len(gt),"empty_gt":len(gt)==0,
           "proposal_counts":json.dumps({k:len(v['ids']) for k,v in stages.items()})}
    rows=[];previous={};preclip=None
    for stage,s in stages.items():
        b,ids=s["boxes"],s["ids"]
        if b.shape!=(len(ids),4) or b.dtype!=torch.float32 or b.device.type!="cpu":raise ValueError("Invalid stage boxes")
        if ids.dtype!=torch.int64 or len(ids.unique())!=len(ids):raise ValueError("Invalid candidate IDs")
        if len(ids) and (int(ids.min())<0 or int(ids.max())>=len(catalogue)):raise ValueError("ID out of catalogue")
        if not bool(torch.isfinite(b).all()) or bool((b[:,2:]<b[:,:2]).any()):raise ValueError("Invalid box geometry")
        values={"float32_cpu":box_iou(gt,b),"float64_diagnostic":box_iou(gt.double(),b.double())}
        if not all(bool(torch.isfinite(v).all()) for v in values.values()):raise ValueError("Nonfinite IoU")
        if stage=="pre_nms_topn":preclip=values["float64_diagnostic"]
        if stage=="clipped" and preclip is not None:
            if preclip.shape!=values["float64_diagnostic"].shape or not torch.equal(stages["pre_nms_topn"]["ids"],ids):raise ValueError("Clip identities changed")
            if bool((values["float64_diagnostic"]+1e-6<preclip).any()):raise ValueError("Contained GT clip IoU decreased")
        for gi in range(len(gt)):
            row={"model":model,"stem":stem,"scene_id":scene_id,"image_index":image_index,"gt_index":gi,
                 "gt_xyxy":json.dumps(gt[gi].tolist()),"stage":stage,"proposal_count":len(ids)}
            for convention,v in values.items():
                vector=v[gi];best=int(vector.argmax()) if len(ids) else None
                maximum=float(vector[best]) if best is not None else 0.;count=int((vector>=.75).sum())
                covered=count>0;old=previous.get((gi,convention))
                if old and stage!="clipped" and (maximum>old[0]+1e-6 or count>old[1]):raise ValueError("Deletion increased quality/coverage")
                row.update({f"max_iou_{convention}":maximum,f"count_iou_ge_075_{convention}":count,
                    f"covered_075_{convention}":covered,
                    f"best_candidate_id_{convention}":catalogue[int(ids[best])]["candidate_id"] if best is not None else None,
                    f"previous_covered_{convention}":old[2] if old else None,
                    f"coverage_lost_{convention}":bool(old and old[2] and not covered),
                    f"coverage_gained_{convention}":bool(old and not old[2] and covered),
                    f"partial_high_loss_coverage_kept_{convention}":bool(old and count<old[1] and covered),
                    f"max_iou_change_{convention}":maximum-old[0] if old else None})
                previous[gi,convention]=(maximum,count,covered)
            rows.append(row)
    return rows,image

def summarize(rows,images):
    keys=[(r["model"],r["stem"],r["gt_index"],r["stage"]) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError("Duplicate model/stem/GT/stage key")
    image_keys=[(r["model"],r["stem"]) for r in images]
    if len(set(image_keys))!=len(image_keys):raise ValueError("Duplicate image")
    bygt=defaultdict(dict)
    for r in rows:bygt[r["model"],r["stem"],r["gt_index"]][r["stage"]]=r
    for im in images:
        expected={(im["model"],im["stem"],g) for g in range(im["gt_count"])}
        actual={k for k in bygt if k[:2]==(im["model"],im["stem"])}
        if expected!=actual:raise ValueError("Missing/extra GT")
    trajectories=[]
    for (model,stem,gi),ss in sorted(bygt.items()):
        if set(ss)!=set(STAGES):raise ValueError("Incomplete GT stages")
        for convention in CONVENTIONS:
            sequence=[ss[s][f"covered_075_{convention}"] for s in STAGES]
            first=next((s for s,h in zip(STAGES,sequence) if h),None)
            losses=[s for s in STAGES if ss[s][f"coverage_lost_{convention}"]]
            gains=[s for s in STAGES if ss[s][f"coverage_gained_{convention}"]]
            trajectories.append({"model":model,"stem":stem,"gt_index":gi,"scene_id":ss[STAGES[0]]["scene_id"],
                "iou_convention":convention,"coverage_sequence":"".join(str(int(h)) for h in sequence),
                "never_covered":not any(sequence),"first_covered_stage":first,"first_all_loss_stage":losses[0] if losses else None,
                "loss_stages":json.dumps(losses),"gain_stages":json.dumps(gains),
                "first_coverage_after_decoded":bool(first and first!="decoded"),
                "recovered_after_loss":any(STAGES.index(g)>STAGES.index(l) for g in gains for l in losses)})
    summary=[];never=[];pair_detail=[];paired=[];net=[]
    scenes=sorted({i["scene_id"] for i in images});scopes=[("overall","ALL")]+[("scene",s) for s in scenes]
    for scope,scene in scopes:
        scoped=lambda r:scope=="overall" or r["scene_id"]==scene
        for model in ("IR","Fusion"):
            ims=[i for i in images if i["model"]==model and scoped(i)]
            denominator=sum(i["gt_count"] for i in ims)
            for convention in CONVENTIONS:
                ts=[t for t in trajectories if t["model"]==model and t["iou_convention"]==convention and scoped(t)]
                never.append({"scope":scope,"scene_id":scene,"model":model,"iou_convention":convention,
                    "never_covered_gt":sum(t["never_covered"] for t in ts),"valid_gt":denominator,
                    "never_covered_rate":ratio(sum(t["never_covered"] for t in ts),denominator)})
                for stage in STAGES:
                    rr=[r for r in rows if r["model"]==model and r["stage"]==stage and scoped(r)]
                    if len(rr)!=denominator:raise ValueError("GT denominator mismatch")
                    covered=sum(r[f"covered_075_{convention}"] for r in rr)
                    lost=sum(r[f"coverage_lost_{convention}"] for r in rr);gained=sum(r[f"coverage_gained_{convention}"] for r in rr)
                    prev=sum(bool(r[f"previous_covered_{convention}"]) for r in rr) if stage!="decoded" else None
                    summary.append({"scope":scope,"scene_id":scene,"model":model,"iou_convention":convention,"stage":stage,
                        "images":len(ims),"empty_gt_images":sum(i["empty_gt"] for i in ims),"valid_gt":denominator,
                        "covered_gt":covered,"coverage_rate":ratio(covered,denominator),"previous_covered_gt":prev,
                        "coverage_lost_gt":lost,"lost_rate_all_gt":ratio(lost,denominator) if stage!="decoded" else None,
                        "lost_rate_previous_covered":ratio(lost,prev) if prev is not None else None,
                        "coverage_gained_gt":gained,"partial_high_loss_coverage_kept_gt":sum(r[f"partial_high_loss_coverage_kept_{convention}"] for r in rr)})
    ir={(stem,gi):ss for (model,stem,gi),ss in bygt.items() if model=="IR"}
    fusion={(stem,gi):ss for (model,stem,gi),ss in bygt.items() if model=="Fusion"}
    if set(ir)!=set(fusion):raise ValueError("IR/Fusion GT keys differ")
    for key,iss in ir.items():
        fss=fusion[key]
        if any(iss[s]["gt_xyxy"]!=fss[s]["gt_xyxy"] or iss[s]["scene_id"]!=fss[s]["scene_id"] for s in STAGES):raise ValueError("Paired GT differs")
        for convention in CONVENTIONS:
            for stage in STAGES:
                a,b=iss[stage][f"covered_075_{convention}"],fss[stage][f"covered_075_{convention}"]
                pair_detail.append({"stem":key[0],"gt_index":key[1],"scene_id":iss[stage]["scene_id"],"stage":stage,
                    "iou_convention":convention,"IR_covered":a,"Fusion_covered":b,
                    "pair_category":"both" if a and b else "IR_only" if a else "Fusion_only" if b else "neither"})
    for scope,scene in scopes:
        for convention in CONVENTIONS:
            prev_net=None
            for stage in STAGES:
                rr=[r for r in pair_detail if scoped(r) and r["stage"]==stage and r["iou_convention"]==convention]
                counts={cat:sum(r["pair_category"]==cat for r in rr) for cat in ("both","IR_only","Fusion_only","neither")}
                a=next(r for r in summary if (r["scope"],r["scene_id"],r["model"],r["iou_convention"],r["stage"])==(scope,scene,"IR",convention,stage))
                b=next(r for r in summary if (r["scope"],r["scene_id"],r["model"],r["iou_convention"],r["stage"])==(scope,scene,"Fusion",convention,stage))
                paired.append({"scope":scope,"scene_id":scene,"iou_convention":convention,"stage":stage,
                    "valid_gt":len(rr),**counts,**{k+"_rate":ratio(v,len(rr)) for k,v in counts.items()}})
                difference=a["covered_gt"]-b["covered_gt"];change=difference-prev_net if prev_net is not None else None
                loss_term=b["coverage_lost_gt"]-a["coverage_lost_gt"] if prev_net is not None else None
                gain_term=a["coverage_gained_gt"]-b["coverage_gained_gt"] if prev_net is not None else None
                if prev_net is not None and change!=loss_term+gain_term:raise ValueError("Net arithmetic decomposition mismatch")
                if difference!=counts["IR_only"]-counts["Fusion_only"]:raise ValueError("Pair net mismatch")
                net.append({"scope":scope,"scene_id":scene,"iou_convention":convention,"stage":stage,"valid_gt":len(rr),
                    "IR_covered_gt":a["covered_gt"],"Fusion_covered_gt":b["covered_gt"],"IR_minus_Fusion":difference,
                    "net_change_from_previous":change,"Fusion_losses_minus_IR_losses":loss_term,
                    "IR_gains_minus_Fusion_gains":gain_term,"interpretation":"arithmetic_decomposition_not_causation"})
                prev_net=difference
    return {"stage_summary":summary,"never_covered_summary":never,"gt_coverage_sequences":trajectories,
            "paired_per_gt":pair_detail,"paired_summary":paired,"net_difference":net}
