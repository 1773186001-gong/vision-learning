"""Replay captured Fusion RoI NMS for one of the two verified Day01 cases."""

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
W10 = ROOT / 'week10/day05'
DAY01 = ROOT / 'week11/day01/outputs/fusion_roi_trace_run02'
CASES = {'030129': (8571, 'day02'), '030078': (2841, 'day03')}
sys.path.insert(0, str(W10))
from replay_roi_nms import csvout, exact, greedy_replay, load, read, require, save, sha


def replay_case(out, counts, stem, focus_flat, day):
    scope = f'Fusion/{stem}/G0'
    case = DAY01 / stem
    manifest = load(DAY01 / 'artifact_manifest.json')
    names = ('verification.json', 'run_metadata.json', f'{stem}/verification.json',
             f'{stem}/raw_capture.pth', f'{stem}/stage_candidates.csv', f'{stem}/final_sources.csv')
    fingerprints = {name: sha(DAY01 / name) for name in names}
    require(all(fingerprints[name] == manifest[name] for name in names), 'Day01 artifact hash differs')
    status = load(case / 'verification.json')
    require(status['status'] == 'passed' and status['identity_chain_verified'] and status['exact_reproduced'] and
            load(DAY01 / 'verification.json')['status'] == 'passed', 'Day01 capture is not verified')
    env = load(DAY01 / 'run_metadata.json')['environment']
    require(env['device'] == 'cpu' and env['torch'] == torch.__version__ and
            env['torchvision'] == torchvision.__version__, 'Day01 environment differs')
    source = Path(inspect.getfile(box_ops))
    saved_source = W10 / 'outputs/roi_nms_replay_run01/boxes_source.py.txt'
    implementation = inspect.getsource(box_ops.batched_nms)
    coordinate_path = inspect.getsource(box_ops._batched_nms_coordinate_trick)
    require('boxes.numel() > (4000' in implementation and 'boxes + offsets[:, None]' in coordinate_path and
            not torch.jit.is_tracing() and not torch.jit.is_scripting(), 'CPU coordinate-offset path changed')
    require(source.read_text(encoding='utf-8') == saved_source.read_text(encoding='utf-8'),
            'Installed NMS source differs from Week10 replay source')
    (out / 'boxes_source.py.txt').write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    replay_source = W10 / 'replay_roi_nms.py'
    save(out / 'input_provenance.json', dict(day01=str(DAY01), files=fingerprints,
                                            day01_manifest_sha256=sha(DAY01 / 'artifact_manifest.json'),
                                            entrypoint_sha256=sha(Path(__file__)),
                                            replay_source=str(replay_source), replay_source_sha256=sha(replay_source),
                                            nms_source=str(source), nms_source_sha256=sha(source)))

    counts['candidate_capture_loads'] += 1
    raw = torch.load(case / 'raw_capture.pth', map_location='cpu', weights_only=True)
    require((raw['model'], raw['stem'], raw['gt_index']) == ('Fusion', stem, 0), 'Wrong capture identity')
    nms = raw['nms']
    require(len(nms['args']) == 4 and not nms['kwargs'], 'Unexpected captured NMS signature')
    boxes, scores, labels, threshold = nms['args']
    n = len(boxes)
    require(boxes.shape == (n, 4) and scores.shape == labels.shape == (n,) and 0 < boxes.numel() <= 4000,
            'Unexpected input shape for the CPU coordinate-offset path')
    require(boxes.dtype == scores.dtype == torch.float32 and labels.dtype == torch.int64 and
            bool((labels == 1).all()) and threshold == .5, 'Unexpected original class, dtype or threshold')
    require(bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(scores).all()) and
            bool((boxes[:, 2:] > boxes[:, :2]).all()), 'Invalid input coordinates or scores')
    # The replay sees the original operator inputs only. Read saved output indices afterward.
    counts['independent_nms_replays'] += 1
    result = greedy_replay(boxes, scores, labels, threshold)
    original = nms['result']
    original_order = original.tolist()
    require(original.ndim == 1 and original.dtype == torch.int64 and len(set(original_order)) == len(original_order) and
            all(0 <= i < n for i in original_order), 'Invalid saved NMS output indices')

    problems = []
    generated = torch.tensor(result['keep'], dtype=torch.int64)
    cfg = raw['configuration']['roi']
    budget = cfg['detections']
    require(budget == 100 and cfg['score'] == .001, 'Day01 RoI configuration differs')
    final = generated[:budget]
    captured_post = raw['actual_post_output']
    prediction = raw['model_output']
    flags = dict(full_keep_order_exact=exact(generated, original),
                 replay_budget_and_post_exact=all(exact(tensor[final], post[0])
                                                  for tensor, post in zip((boxes, scores, labels), captured_post)),
                 replay_final_model_output_exact=all(exact(tensor[final], prediction[key])
                                                     for tensor, key in zip((boxes, scores, labels), ('boxes', 'scores', 'labels'))))
    counts['full_output_verifications'] += 1
    save(out / 'output_comparison.json', dict(checks=flags, original_indices=original_order, replay_indices=result['keep']))
    require(all(flags.values()), 'Replay failed full NMS or final output verification')

    flats = raw['rpn_stages'][0]['post_nms_topn']['ids']
    probs = raw['softmax']['result'][:, 1]
    clipped = raw['clip']['result'][:, 1]
    score_rows = torch.where(probs > cfg['score'])[0]
    roi_rows = score_rows[raw['small']['result']]
    require(exact(clipped[score_rows], raw['small']['args'][0]) and exact(clipped[roi_rows], boxes) and
            exact(probs[roi_rows], scores), 'Input identity chain does not match the captured tensors')
    flat_ids = flats[roi_rows].tolist()
    gt_iou = box_iou(boxes, raw['gt_all'][0:1])[:, 0].tolist()
    rows = [r for r in read(case / 'stage_candidates.csv') if r['stage'] == 'small_filtered']
    require(len(rows) == len(set(flat_ids)) == n and focus_flat in flat_ids, 'Day01 input table count or identity differs')
    for i, row in enumerate(rows):
        require((int(row['stage_index']), int(row['roi_row']), int(row['flat'])) == (i, int(roi_rows[i]), flat_ids[i]) and
                json.loads(row['xyxy']) == boxes[i].tolist() and float(row['person_score']) == float(scores[i]) and
                float(row['gt_iou']) == gt_iou[i], f'Day01 input table differs at position {i}')
    require([int(row['flat']) for row in read(case / 'final_sources.csv')] ==
            [flat_ids[i] for i in original_order[:budget]], 'Day01 final source table differs')

    shifted = boxes + labels.to(boxes)[:, None] * (boxes.max() + 1.)
    require(shifted.tolist() == result['shifted'], 'Float32 class offsets differ')
    pair_iou = box_iou(shifted, shifted)
    for pair in result['comparisons']:
        a, b = pair['selected_position'], pair['other_position']
        value = pair['path_iou']
        require(value == float(pair_iou[a, b]), f'Float32 overlap differs for positions {a}, {b}')
        if abs(value - threshold) <= 1e-6:
            problems.append(dict(kind='overlap_near_threshold', positions=[a, b], pair_iou=value))
    score_ties = [score for score, multiplicity in Counter(scores.tolist()).items() if multiplicity > 1]
    if score_ties:
        problems.append(dict(kind='score_ties', scores=score_ties))
    save(out / 'attribution_checks.json', dict(ambiguities=problems, tolerance_used_for_replay=False,
                                               independent_pair_checks=len(result['comparisons'])))
    require(not problems, 'First suppressor attribution needs further review')
    require(Counter(d['input_position'] for d in result['decisions']) == Counter(range(n)),
            'Replay decisions do not cover all input positions exactly once')

    decisions = []
    for d in sorted(result['decisions'], key=lambda d: d['input_position']):
        i, a = d['input_position'], d['selected_position']
        deleted = d['action'] == 'delete'
        decisions.append(dict(input_position=i, roi_row=int(roi_rows[i]), flat=flat_ids[i],
                              score=float(scores[i]), gt_iou=gt_iou[i], action=d['action'], replay_step=d['step'] + 1,
                              first_suppressor_flat=flat_ids[a] if deleted else None,
                              suppressor_score=float(scores[a]) if deleted else None,
                              suppressor_gt_iou=gt_iou[a] if deleted else None, pair_iou=d['path_iou'],
                              day01_kept=i in original_order))
    good = [d for d in decisions if d['gt_iou'] >= .75]
    focus = next(d for d in good if d['flat'] == focus_flat)
    require(focus['action'] == 'delete' and all(d['action'] == 'delete' for d in good), 'Unexpected saved coverage')
    focus_pos = focus['input_position']
    retained = [dict(flat=flat_ids[i], score=float(scores[i]), gt_iou=gt_iou[i],
                     **{f'pair_iou_to_{focus_flat}': float(pair_iou[i, focus_pos])},
                     earlier_replay_selection=(j + 1) < focus['replay_step'],
                     replay_first_suppressor=flat_ids[i] == focus['first_suppressor_flat'])
                for j, i in enumerate(original_order)]
    csvout(out / 'candidate_decisions.csv', decisions)
    csvout(out / 'good_box_attributions.csv', good)
    csvout(out / f'retained_vs_{focus_flat}.csv', retained)
    csvout(out / 'replay_steps.csv', [dict(step=step['step'] + 1, selected_flat=flat_ids[step['selected_position']],
                                          removed_flats=[flat_ids[i] for i in step['new_deleted_positions']])
                                     for step in result['steps']])
    groups = dict(sorted(Counter(d['first_suppressor_flat'] for d in good).items()))
    require(all(sha(DAY01 / name) == h for name, h in fingerprints.items()), 'Day01 input hash changed during replay')
    report = (f'# Week11 {day.capitalize()} · {scope} 类别 NMS 离线重放\n\n'
              f'读取 Day01 原始捕获的全部 {n} 个 NMS 输入。重放的 {len(original_order)} 个保留索引及顺序、'
              '预算切片与最终检测均与记录精确一致。\n\n'
              f'候选 `{focus_flat}` 在重放第 **{focus["replay_step"]}** 步被 `{focus["first_suppressor_flat"]}` 首次移除。'
              f'二者的预测框 IoU 为 **{focus["pair_iou"]:.6f}**，大于类别 NMS 阈值 0.5。'
              f'它们的 person 分数分别为 {focus["score"]:.6f} 和 {focus["suppressor_score"]:.6f}。'
              f'保留框对 G0 的 IoU 为 {focus["suppressor_gt_iou"]:.6f}。\n\n'
              f'{len(good)} 个达标框按重放首次抑制框计数如下。\n\n'
              '| 保留框 flat | 首次移除的达标框数 |\n|---:|---:|\n' +
              ''.join(f'| {flat} | {count} |\n' for flat, count in groups.items()) +
              '\n以上步骤来自与原输出一致的独立重放。原 CPU 捕获提供完整输入和保留索引。\n')
    (out / 'case_report_zh.md').write_text(report, encoding='utf-8')
    return dict(nms_input_count=n, nms_output_count=len(original_order), good_box_count=len(good),
                identity_chain_verified=True, keep_order_exact=True, focus=focus, first_suppressor_counts=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', choices=CASES, default='030129')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    focus_flat, day = CASES[args.stem]
    out = args.output.resolve()
    require(out.parent == HERE.parent / day / 'outputs', f'Choose a new folder directly under week11/{day}/outputs')
    out.mkdir(parents=True, exist_ok=False)
    counts = dict(candidate_capture_loads=0, independent_nms_replays=0, full_output_verifications=0,
                  checkpoint_loads=0, model_forwards=0, original_nms_operator_calls=0, training=0, server_connections=0)
    result, error = {}, None
    try:
        result = replay_case(out, counts, args.stem, focus_flat, day)
    except Exception as exc:
        error = str(exc)
        (out / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
    code = int(error is not None)
    save(out / 'verification.json', dict(status='failed' if code else 'passed', scope=f'Fusion/{args.stem}/G0', error=error,
                                         command=subprocess.list2cmdline(sys.orig_argv), counts=counts,
                                         exit_code=code, **result))
    save(out / 'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    if error:
        print(error, file=sys.stderr)
    print(f'{"FAILED" if code else "PASSED"}: {out}')
    return code


if __name__ == '__main__':
    sys.exit(main())
