"""Independent synthetic-testable RPN trace core. No checkpoint or dataset loading."""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
import inspect
import threading
import torch
from torchvision.models.detection import rpn as rpn_module
from torchvision.ops import boxes as box_ops

_LOCK = threading.Lock()
_MISSING = object()
STAGES = ("decoded", "pre_nms_topn", "clipped", "small_box_filtered",
          "score_filtered", "layer_nms", "post_nms_topn")

class VerificationError(RuntimeError):
    pass

@dataclass(frozen=True)
class Config:
    pre_nms: int
    post_nms: int
    min_size: float
    score_thresh: float
    nms_thresh: float
    iou_atol: float = 1e-6

@dataclass
class CaptureRecord:
    boxes: torch.Tensor
    logits: torch.Tensor
    counts: tuple[int, ...]
    image_size: tuple[int, int]
    layer_keys: tuple[str, ...]
    gt: torch.Tensor
    config: Config
    top_indices: torch.Tensor
    nms_boxes: torch.Tensor
    nms_scores: torch.Tensor
    nms_levels: torch.Tensor
    nms_indices: torch.Tensor
    final_boxes: torch.Tensor
    final_scores: torch.Tensor
    prefix: str

def same(actual, expected, label):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.device != expected.device or not torch.equal(actual, expected)):
        raise VerificationError(f"{label}: tensor coordinates/order/scores differ")

def indices(idx, n, label):
    if idx.ndim != 1 or idx.dtype != torch.int64:
        raise VerificationError(f"{label}: expected int64 vector")
    if len(idx) and (bool((idx < 0).any()) or bool((idx >= n).any())
                     or len(torch.unique(idx)) != len(idx)):
        raise VerificationError(f"{label}: out-of-range or repeated indices")

def validate_geometry(boxes, gt, image_size):
    h, w = image_size
    if h <= 0 or w <= 0 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Invalid image size or box shape")
    if not boxes.is_floating_point() or not bool(torch.isfinite(boxes).all()):
        raise ValueError("Decoded candidates must be finite floating boxes")
    if len(boxes) and bool((boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("Initial degenerate or inverted decoded box; no silent sanitizing")
    if gt.shape != (4,) or not bool(torch.isfinite(gt).all()):
        raise ValueError("One finite GT is required")
    if bool((gt[2:] <= gt[:2]).any()):
        raise ValueError("GT must have positive width/height")
    if bool((gt < 0).any()) or gt[0] > w or gt[2] > w or gt[1] > h or gt[3] > h:
        raise ValueError("GT must be wholly inside the full image including letterbox padding")

def diagnostic_iou(boxes, gt):
    """Float64 diagnostic geometry; zero-area clipped boxes have IoU zero."""
    b, g = boxes.to(torch.float64), gt.to(device=boxes.device, dtype=torch.float64)
    wh = (torch.minimum(b[:, 2:], g[2:]) - torch.maximum(b[:, :2], g[:2])).clamp(min=0)
    intersection = wh.prod(1)
    area = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
    gt_area = (g[2:] - g[:2]).prod()
    return intersection / (area + gt_area - intersection)

def catalogue(prefix, keys, counts, device):
    if len(keys) != len(counts) or len(set(keys)) != len(keys) or any(n < 0 for n in counts):
        raise ValueError("Actual level keys/counts mismatch")
    levels = torch.cat([torch.full((n,), i, dtype=torch.int64, device=device)
                        for i, n in enumerate(counts)]) if counts else torch.empty(0, dtype=torch.int64, device=device)
    entries = []
    flat = 0
    for level, (key, n) in enumerate(zip(keys, counts)):
        for local in range(n):
            entries.append(dict(candidate_id=f"{prefix}/{flat}", flat_index=flat,
                                level_key=key, level_index=level, index_within_level=local))
            flat += 1
    return entries, levels

@dataclass
class Stage:
    name: str
    ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    iou: torch.Tensor
    summary: dict

@dataclass
class Trace:
    catalogue: list[dict]
    levels: torch.Tensor
    stages: dict[str, Stage]
    first_removed: dict[int, str]
    clip_deltas: list[dict]
    verified: bool = True

    def history(self, flat_index):
        if not 0 <= flat_index < len(self.catalogue):
            raise IndexError(flat_index)
        rows = []
        for name, s in self.stages.items():
            hit = torch.where(s.ids == flat_index)[0]
            row = dict(stage=name, candidate_id=self.catalogue[flat_index]["candidate_id"],
                       kept=bool(len(hit)), first_removed_stage=self.first_removed.get(flat_index))
            if len(hit):
                p = int(hit[0]); score = s.scores[p]
                level_mask = self.levels[s.ids] == self.levels[flat_index]
                row.update(xyxy=s.boxes[p].tolist(), iou=float(s.iou[p]), score=float(score),
                           stage_position_0based=p,
                           score_rank_min=1+int((s.scores > score).sum()),
                           score_rank_max=int((s.scores >= score).sum()),
                           level_rank_min=1+int((s.scores[level_mask] > score).sum()))
            rows.append(row)
        return rows

def replay(record: CaptureRecord) -> Trace:
    r = record
    if r.boxes.ndim != 2 or r.logits.numel() != len(r.boxes) or sum(r.counts) != len(r.boxes):
        raise ValueError("Single-image shapes/counts mismatch")
    validate_geometry(r.boxes, r.gt, r.image_size)
    if not bool(torch.isfinite(r.logits).all()):
        raise ValueError("Nonfinite objectness")
    c = r.config
    if c.pre_nms < 0 or c.post_nms < 0 or c.min_size <= 0 or c.iou_atol < 0:
        raise ValueError("Invalid budget/min_size/tolerance")
    if not 0 <= c.score_thresh <= 1 or not 0 <= c.nms_thresh <= 1:
        raise ValueError("Invalid score/NMS threshold")
    cat, levels = catalogue(r.prefix, r.layer_keys, r.counts, r.boxes.device)
    stages, removed, clip_deltas = {}, {}, []
    ids = torch.arange(len(r.boxes), device=r.boxes.device)
    boxes, scores = r.boxes.clone(), torch.sigmoid(r.logits.reshape(-1))
    def snapshot(name):
        iou = diagnostic_iou(boxes, r.gt)
        maximum = float(iou.max()) if len(iou) else 0.
        best = torch.where(iou == maximum)[0] if len(iou) else ids[:0]
        best_ids = ids[best].tolist()
        s = dict(stage=name, candidate_count=len(ids), empty=not len(ids), max_iou=maximum,
                 best_candidate_id=cat[best_ids[0]]["candidate_id"] if best_ids else None,
                 tied_best_candidate_ids=[cat[i]["candidate_id"] for i in best_ids],
                 count_iou_ge_075=int((iou >= .75).sum()),
                 counts_by_level={key:int((levels[ids] == i).sum()) for i,key in enumerate(r.layer_keys)},
                 score_origin="diagnostic_sigmoid_all" if name == "decoded" else "original_selected_sigmoid")
        if len(best):
            value = scores[best[0]]
            s.update(best_score=float(value), best_score_rank_min=1+int((scores>value).sum()),
                     best_score_rank_max=int((scores>=value).sum()))
        stages[name] = Stage(name, ids.clone(), boxes.clone(), scores.clone(), iou.clone(), s)
    def select(keep, name):
        nonlocal ids, boxes, scores
        indices(keep, len(ids), name)
        before = stages[next(reversed(stages))]
        new_ids = ids[keep]
        for i in set(ids.tolist()) - set(new_ids.tolist()):
            removed.setdefault(i, name)
        ids, boxes, scores = new_ids, boxes[keep], scores[keep]
        snapshot(name)
        current = stages[name]
        if (current.summary["max_iou"] > before.summary["max_iou"] + c.iou_atol
                or current.summary["count_iou_ge_075"] > before.summary["count_iou_ge_075"]):
            raise VerificationError("Pure deletion increased IoU/coverage")
    snapshot("decoded")
    keep = r.top_indices
    indices(keep, len(ids), "captured Top-N")
    chosen_levels = levels[keep]
    offset = 0
    for i,n in enumerate(r.counts):
        chosen = keep[chosen_levels == i]
        if len(chosen) != min(c.pre_nms, n):
            raise VerificationError("Captured pre-NMS budget is not per level")
        if len(chosen) and len(chosen) < n:
            all_local = torch.arange(offset, offset+n, device=ids.device)
            omitted = all_local[~torch.isin(all_local, chosen)]
            if bool((r.logits.reshape(-1)[omitted].max() > r.logits.reshape(-1)[chosen].min())):
                raise VerificationError("Captured Top-N omitted a strictly higher logit")
        offset += n
    # Preserve the actual helper order, including its tie choices.
    select(keep, "pre_nms_topn")
    scores = torch.sigmoid(r.logits.reshape(-1)[keep])
    # Replace diagnostic-all scores with the original-shaped selected sigmoid.
    snapshot("pre_nms_topn")
    old_boxes = boxes.clone()
    old_iou = diagnostic_iou(old_boxes, r.gt)
    boxes = box_ops.clip_boxes_to_image(boxes, r.image_size)
    snapshot("clipped")
    new_iou = stages["clipped"].iou
    if bool((new_iou + c.iou_atol < old_iou).any()):
        raise VerificationError("Contained GT clipping lowered IoU beyond tolerance")
    for j, identity in enumerate(ids.tolist()):
        clip_deltas.append(dict(flat_index=identity, candidate_id=cat[identity]["candidate_id"],
                                xyxy_before=old_boxes[j].tolist(), xyxy_after=boxes[j].tolist(),
                                iou_before=float(old_iou[j]), iou_after=float(new_iou[j]),
                                delta=float(new_iou[j]-old_iou[j]),
                                degenerate_after=bool((boxes[j,2:] <= boxes[j,:2]).any())))
    select(box_ops.remove_small_boxes(boxes, c.min_size), "small_box_filtered")
    select(torch.where(scores >= c.score_thresh)[0], "score_filtered")
    same(boxes, r.nms_boxes, "original NMS input boxes")
    same(scores, r.nms_scores, "original NMS input scores")
    same(levels[ids], r.nms_levels, "original NMS input levels")
    select(r.nms_indices, "layer_nms")
    if len(scores) and bool((scores[1:] > scores[:-1]).any()):
        raise VerificationError("Captured NMS order is not descending")
    select(torch.arange(min(c.post_nms,len(ids)),device=ids.device), "post_nms_topn")
    same(boxes,r.final_boxes,"original final boxes")
    same(scores,r.final_scores,"original final scores")
    return Trace(cat,levels,stages,removed,clip_deltas)

def _patch_attribute(stack, obj, name, value):
    old = obj.__dict__.get(name, _MISSING)
    def restore():
        if old is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, old)
    stack.callback(restore)
    setattr(obj, name, value)

class TraceCapture:
    """Single-image, serial numeric call adapter; never constructs or loads a model.

    hooks is an optional sequence of (module, callback) to be registered here.
    Shared NMS reference is patched only during the original filter invocation.
    """
    def __init__(self, rpn, layer_keys, gt, *, prefix="synthetic/unit/image", image_atol=1e-6, hooks=()):
        self.rpn,self.keys,self.gt = rpn,tuple(layer_keys),gt
        self.prefix,self.atol,self.hooks = prefix,image_atol,tuple(hooks)
        self.records,self.traces = [],[]
        self.returned_reference = None
        self.active=False

    def __enter__(self):
        if self.active:
            raise RuntimeError("Capture context is already active")
        stack = ExitStack()
        try:
            for module, callback in self.hooks:
                handle=module.register_forward_hook(callback)
                stack.callback(handle.remove)
            original=self.rpn.filter_proposals
            # The instance may already have a generic (*args, **kwargs) wrapper.
            # Bind using the installed original API, while still calling that instance method.
            signature=inspect.signature(rpn_module.RegionProposalNetwork.filter_proposals)
            def wrapper(*args,**kwargs):
                bound=signature.bind(self.rpn,*args,**kwargs).arguments
                p=bound["proposals"];logits=bound["objectness"]
                shapes=bound["image_shapes"];counts=tuple(bound["num_anchors_per_level"])
                if p.shape[0]!=1 or len(shapes)!=1:
                    raise ValueError("Minimum adapter supports batch_size=1 only")
                validate_geometry(p[0],self.gt,tuple(shapes[0]))
                if not _LOCK.acquire(blocking=False):
                    raise RuntimeError("Concurrent capture is forbidden")
                try:
                    # All work after successful acquire is release-protected,
                    # including configuration and original-method lookups.
                    owner=threading.get_ident()
                    top_calls,nms_calls=[],[]
                    cfg=Config(self.rpn.pre_nms_top_n(),self.rpn.post_nms_top_n(),
                               self.rpn.min_size,self.rpn.score_thresh,self.rpn.nms_thresh,self.atol)
                    top_original=self.rpn._get_top_n_idx
                    nms_original=rpn_module.box_ops.batched_nms
                    def top_wrapper(*a,**k):
                        if threading.get_ident()!=owner:raise RuntimeError("Other-thread Top-N call")
                        result=top_original(*a,**k);top_calls.append(result.detach().clone())
                        return result
                    def nms_wrapper(b,s,l,t):
                        if threading.get_ident()!=owner:raise RuntimeError("Other-thread NMS call")
                        result=nms_original(b,s,l,t)
                        nms_calls.append((b.detach().clone(),s.detach().clone(),l.detach().clone(),result.detach().clone()))
                        return result
                    saved_p=p.detach().clone()
                    saved_logits=logits.detach().clone()
                    with ExitStack() as narrow:
                        _patch_attribute(narrow,self.rpn,"_get_top_n_idx",top_wrapper)
                        _patch_attribute(narrow,rpn_module.box_ops,"batched_nms",nms_wrapper)
                        output=original(*args,**kwargs)
                    self.returned_reference=output
                    if len(top_calls)!=1 or len(nms_calls)!=1 or len(output)!=2:
                        raise VerificationError("Unexpected original helper call count/return")
                    b,s,l,keep=nms_calls[0]
                    rec=CaptureRecord(saved_p[0],saved_logits.reshape(-1),counts,tuple(shapes[0]),self.keys,
                                      self.gt.detach().clone(),cfg,top_calls[0][0],b,s,l,keep,
                                      output[0][0].detach().clone(),output[1][0].detach().clone(),self.prefix)
                    self.records.append(rec)
                    self.traces.append(replay(rec))
                    return output
                finally:
                    _LOCK.release()
            _patch_attribute(stack,self.rpn,"filter_proposals",wrapper)
            self._stack=stack;self.active=True
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self,*exc):
        try:
            return self._stack.__exit__(*exc)
        finally:
            self.active=False

