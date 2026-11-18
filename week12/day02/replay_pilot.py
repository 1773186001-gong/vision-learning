"""Replay the first new RoI NMS job for each model from the frozen loss queue."""

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import torchvision
from torchvision.ops import box_iou, boxes as box_ops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / 'week10/day05'
FREEZE = ROOT / 'week12/day01/outputs/frozen_roi_nms_queue_run01'
SCOPE = ROOT / 'week11/day04/outputs/full_val_scope_run02/scope.json'
MODELS = ('IR', 'Fusion')
sys.path.insert(0, str(BASELINE))
from replay_roi_nms import exact, greedy_replay, load, require, save, sha


def load_queue():
    manifest = load(FREEZE / 'artifact_manifest.json')
    for name in ('replay_queue.json', 'verification.json'):
        require(sha(FREEZE / name) == manifest[name], f'Frozen Day01 artifact differs: {name}')
    verification = load(FREEZE / 'verification.json')
    queue = load(FREEZE / 'replay_queue.json')
    require(verification['status'] == 'passed' and queue['status'] == 'frozen' and
            len(queue['entries']) == 151 and queue['original_nms_iou_threshold'] == .5 and
            queue['gt_iou_threshold'] == .75, 'The frozen model-image scope or thresholds differ')
    require(sha(SCOPE) == verification['input_artifacts'][SCOPE.relative_to(ROOT).as_posix()], 'Scope SHA differs')
    version = load(SCOPE)['environment']
    require(version['device'] == 'cpu' and version['torch'] == torch.__version__ and
            version['torchvision'] == torchvision.__version__, 'CPU library versions differ')
    source = Path(inspect.getfile(box_ops))
    archived = BASELINE / 'outputs/roi_nms_replay_run01/boxes_source.py.txt'
    require(source.read_text(encoding='utf-8') == archived.read_text(encoding='utf-8') and
            'boxes.numel() > (4000' in inspect.getsource(box_ops.batched_nms) and
            not torch.jit.is_tracing() and not torch.jit.is_scripting(), 'Captured small CPU NMS path differs')
    return queue


def replay_job(job, out, counts, *, replay_budget):
    raw_path = ROOT / job['capture_path']
    require(sha(raw_path) == job['capture_sha256'], 'Raw capture SHA differs')
    raw = torch.load(raw_path, map_location='cpu', weights_only=True)
    counts['candidate_capture_loads'] += 1
    require(raw['run_identity'] == job['run_identity'], 'Raw run identity differs')
    nms = raw['nms']
    require(len(nms['args']) == 4 and not nms['kwargs'], 'NMS call signature differs')
    boxes, scores, labels, threshold = nms['args']
    n = len(boxes)
    require(boxes.shape == (n, 4) and labels.shape == scores.shape == (n,) and n == job['nms_input_count'] and
            0 < boxes.numel() <= 4000 and boxes.dtype == scores.dtype == torch.float32 and labels.dtype == torch.int64 and
            bool((labels == 1).all()) and threshold == .5, 'Expected original inputs on the CPU class-offset path')
    require(all(bool(torch.isfinite(t).all()) for t in (boxes, scores)) and bool((boxes[:, 2:] > boxes[:, :2]).all()),
            'Invalid geometry or classification scores in capture')
    cfg = raw['configuration']['roi']
    require((cfg['score'], cfg['nms'], cfg['detections']) == (.001, .5, 100), 'RoI settings differ from fixed scope')
    probs = raw['softmax']['result'][:, 1]
    person = raw['clip']['result'][:, 1]
    score_rows = torch.where(probs > cfg['score'])[0]
    roi_rows = score_rows[raw['small']['result']]
    require(exact(person[score_rows], raw['small']['args'][0]) and exact(person[roi_rows], boxes) and
            exact(probs[roi_rows], scores), 'RoI identity mapping differs from original NMS inputs')
    flat = raw['rpn_stages'][0]['post_nms_topn']['ids'][roi_rows].tolist()
    require(len(flat) == len(set(flat)) == n, 'Duplicate or missing candidate identities')

    # Send only the actual original inputs to the independent NMS implementation.
    require(counts['new_replays'] < replay_budget, 'Replay budget exceeded')
    counts['new_replays'] += 1
    replay = greedy_replay(boxes, scores, labels, threshold)
    original = nms['result']
    predicted_keep = torch.tensor(replay['keep'], dtype=torch.int64)
    sliced = predicted_keep[:cfg['detections']]
    checks = dict(original_keep_order_exact=exact(predicted_keep, original),
                  original_postprocess_exact=all(exact(v[sliced], p[0]) for v, p in
                                                 zip((boxes, scores, labels), raw['actual_post_output'])),
                  original_final_output_exact=all(exact(v[sliced], raw['model_output'][k]) for v, k in
                                                  zip((boxes, scores, labels), ('boxes', 'scores', 'labels'))))
    counts['full_output_verifications'] += 1
    save(out / 'output_comparison.json', dict(checks=checks, captured_keep=original.tolist(), replay_keep=replay['keep']))
    require(all(checks.values()), 'Complete output differs from original capture')

    shifted = boxes + labels.to(boxes)[:, None] * (boxes.max() + 1.)
    require(shifted.tolist() == replay['shifted'], 'Independent Float32 class offsets differ')
    overlap = box_iou(shifted, shifted)
    ambiguous = []
    for comp in replay['comparisons']:
        a, b = comp['selected_position'], comp['other_position']
        actual = float(overlap[a, b])
        require(actual == comp['path_iou'], f'Float32 IoU implementations differ for {a},{b}')
        if abs(actual - threshold) <= 1e-6:
            ambiguous.append(dict(reason='comparison_near_threshold', positions=[a, b], iou=actual))
    ties = [v for v, count in Counter(scores.tolist()).items() if count > 1]
    if ties:
        ambiguous.append(dict(reason='score_tie', tied_scores=ties))
    save(out / 'attribution_checks.json', dict(ambiguities=ambiguous, comparison_count=len(replay['comparisons']),
                                               threshold_tolerance_used=False))
    require(not ambiguous, 'First-suppressor attribution cannot be assigned unambiguously')

    decisions = {d['input_position']: d for d in replay['decisions']}
    require(len(replay['decisions']) == n and set(decisions) == set(range(n)), 'Incomplete or duplicate replay decisions')
    top_positions = torch.where(scores == scores.max())[0].tolist()
    require(top_positions == replay['keep'][:1], 'Image top-score position differs from first selected candidate')
    top_flat = flat[top_positions[0]]
    decisions_with_identity = []
    for i, d in sorted(decisions.items()):
        removed = d['action'] == 'delete'
        selected = d['selected_position']
        require((i in replay['keep']) == (not removed), 'Replay keep and decision disagree')
        if removed:
            require(selected == replay['keep'][d['step']] and scores[selected] > scores[i] and d['path_iou'] > threshold,
                    'Invalid first-suppressor relationship')
        decisions_with_identity.append(dict(input_position=i, flat=flat[i], roi_row=int(roi_rows[i]),
                                            classification_score=float(scores[i]), action=d['action'], step=d['step'] + 1,
                                            suppressor_flat=flat[selected] if removed else None, pair_iou=d['path_iou']))
    save(out / 'candidate_decisions.json', decisions_with_identity)

    targets = []
    for target in job['targets']:
        g = target['gt_index']
        require(0 <= g < len(raw['gt_all']), 'Queued GT index is outside capture')
        gt_iou = box_iou(boxes, raw['gt_all'][g:g + 1])[:, 0]
        good = torch.where(gt_iou >= .75)[0].tolist()
        require(good and [flat[i] for i in good] == target['qualified_flats_before_nms'] and
                target['top_score_flats'] == [top_flat], 'Frozen GT/candidate list differs from reconstructed list')
        good_decisions = [decisions_with_identity[i] for i in good]
        require(all(d['action'] == 'delete' for d in good_decisions), 'Queued target still has a qualifying keep')
        top = sum(d['suppressor_flat'] == top_flat for d in good_decisions)
        targets.append(dict(gt_index=g, both_models_pre_nms_covered=target['both_models_pre_nms_covered'],
                            top_score_flat=top_flat, qualified_count=len(good), top_score_first_suppressed=top,
                            category='all' if top == len(good) else 'partial' if top else 'none',
                            first_suppressor_by_flat={d['flat']: d['suppressor_flat'] for d in good_decisions}))
    return dict(model=job['model'], stem=job['stem'], status='passed', capture_sha256=job['capture_sha256'],
                nms_input_count=n, nms_kept_count=len(original), full_output_match=True, targets=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    require(out.parent == HERE / 'outputs', 'Choose a new folder directly under week12/day02/outputs')
    queue = load_queue()
    cases = [next(job for job in queue['entries'] if job['model'] == m and job['action'] == 'new_replay') for m in MODELS]
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    counts = dict(candidate_capture_loads=0, new_replays=0, full_output_verifications=0, model_forwards=0, checkpoint_loads=0)
    results = []
    for job in cases:
        folder = out / f'{job["model"]}_{job["stem"]}'
        folder.mkdir()
        try:
            result = replay_job(job, folder, counts, replay_budget=len(cases))
        except Exception as exc:
            (folder / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            result = dict(model=job['model'], stem=job['stem'], status='unverified', error=str(exc), targets=[])
        save(folder / 'result.json', result)
        print(f'{result["model"]}/{result["stem"]}: {result["status"]}', flush=True)
        results.append(result)
    ok = all(r['status'] == 'passed' for r in results)
    save(out / 'pilot_index.json', results)
    save(out / 'verification.json', dict(status='passed' if ok else 'unverified', exit_code=int(not ok), counts=counts,
                                         queued_pilot_jobs=len(cases), verified_targets=sum(len(r['targets']) for r in results),
                                         queue_sha256=sha(FREEZE / 'replay_queue.json'), source_sha256=sha(Path(__file__)),
                                         greedy_replay_source_sha256=sha(BASELINE / 'replay_roi_nms.py'),
                                         command=subprocess.list2cmdline(sys.orig_argv)))
    save(out / 'artifact_manifest.json', {str(p.relative_to(out).as_posix()): sha(p)
                                            for p in sorted(out.rglob('*')) if p.is_file()})
    return int(not ok)


if __name__ == '__main__':
    sys.exit(main())
