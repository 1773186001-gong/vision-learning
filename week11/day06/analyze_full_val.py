"""Analyze the frozen Week11 RoI captures without new model forwards."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import torchvision
from torchvision.ops import box_iou

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOPE = ROOT / 'week11/day04/outputs/full_val_scope_run02/scope.json'
CAPTURE_RUN = ROOT / 'week11/day05/outputs/full_val_roi_capture_run01'
ANCHORS = ROOT / 'week11/day01/outputs/fusion_roi_trace_run02/ir_fusion_comparison.csv'
MODELS = ('IR', 'Fusion')
STAGES = ('rpn', 'decode', 'clip', 'remove_background', 'score_filter', 'before_nms', 'after_nms', 'budget', 'final')
LABELS = dict(zip(STAGES, ('RPN 输出', 'RoI 回归解码', '裁剪', '去背景展开', '分数筛选后', 'NMS 前',
                           'NMS 后', '100 框预算后', '最终输出')))
IOU = 0.75


def check(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) or
                             value is None else value for key, value in row.items()})


def reconstruct(raw, item, sample, configuration):
    check(raw['run_identity'] == item['run_identity'] and
          json.loads(json.dumps(raw['configuration'])) == configuration, 'Capture identity or configuration differs')
    gt = torch.as_tensor(sample['gt_boxes_xyxy_640'], dtype=torch.float32).reshape(-1, 4)
    check(torch.equal(raw['gt_all'], gt) and len(gt) == item['gt_count'], 'Frozen GT order or coordinates differ')
    if 'gt_labels' in raw:
        check(torch.equal(raw['gt_labels'], torch.full((len(gt),), sample['label'], dtype=torch.int64)), 'GT labels differ')
    else:
        check(item['gt_label_source'] == 'frozen_person_xml', 'Missing GT labels without frozen source')
    check(all([tuple(size) for size in raw[key]] == [(640, 640)] for key in ('transform_sizes', 'image_shapes')),
          'Coordinate canvas differs')

    rpn = raw['rpn_stages'][0]['post_nms_topn']
    flat = rpn['ids']
    rows = torch.arange(len(flat), dtype=torch.int64)
    proposals = raw['actual_rpn_return']
    check(len(flat.unique()) == len(proposals) == len(rows) and torch.equal(proposals, rpn['boxes']) and
          torch.equal(proposals, raw['post_proposals'][0]), 'RPN output row identity differs')
    decoded = raw['decode']['result'][:, 1]
    person = raw['clip']['result'][:, 1]
    scores = raw['softmax']['result'][:, 1]
    roi = configuration['roi']
    score_rows = torch.where(scores > roi['score'])[0]
    small_rows = score_rows[raw['small']['result']]
    boxes, nms_scores, labels, nms_threshold = raw['nms']['args']
    check(torch.equal(person[score_rows], raw['small']['args'][0]) and torch.equal(person[small_rows], boxes) and
          torch.equal(scores[small_rows], nms_scores) and nms_threshold == roi['nms'] and
          torch.equal(labels, torch.ones(len(small_rows), dtype=torch.int64)), 'Small-box or NMS identity differs')
    keep = raw['nms']['result']
    check(keep.dtype == torch.int64 and len(keep.unique()) == len(keep) and
          bool(((keep >= 0) & (keep < len(small_rows))).all()), 'Invalid actual NMS index')
    kept = small_rows[keep]
    budget = kept[:roi['detections']]
    prediction = raw['model_output']
    for expected, post, key in zip((person[budget], scores[budget], labels[keep][:roi['detections']]),
                                  raw['actual_post_output'], ('boxes', 'scores', 'labels')):
        check(torch.equal(expected, post[0]) and torch.equal(expected, prediction[key]), 'Final budget/output differs')
    check(len(small_rows) == item['nms_input_count'] and len(budget) == item['final_count'], 'Day05 index counts differ')
    stages = dict(zip(STAGES, ((proposals, rows), (decoded, rows), (person, rows), (person, rows),
                              (person[score_rows], score_rows), (boxes, small_rows), (boxes[keep], kept),
                              (person[budget], budget), (prediction['boxes'], budget))))
    return gt, flat, scores, stages


def describe_image(item, sample, raw, configuration):
    gt, flat, scores, stages = reconstruct(raw, item, sample, configuration)
    overlaps = {name: box_iou(boxes, gt) for name, (boxes, _) in stages.items()}
    before_rows = stages['before_nms'][1]
    retained_rows = set(stages['after_nms'][1].tolist())
    final_rows = set(stages['final'][1].tolist())
    top_score_positions = []
    if len(before_rows):
        before_scores = scores[before_rows]
        top_score_positions = torch.where(before_scores == before_scores.max())[0].tolist()

    stage_rows, gt_rows = [], []
    for g in range(len(gt)):
        key = dict(model=item['model'], stem=item['stem'], scene_id=sample['scene_id'], gt_index=g)
        coverage = {}
        maximum_iou_positions = []
        for name in STAGES:
            positions = stages[name][1]
            ious = overlaps[name][:, g]
            qualified = torch.where(ious >= IOU)[0]
            coverage[name] = bool(len(qualified))
            best = float(ious.max()) if len(ious) else None
            best_positions = torch.where(ious == best)[0] if best is not None else torch.empty(0, dtype=torch.int64)
            if name == 'before_nms':
                maximum_iou_positions = best_positions.tolist()
            stage_rows.append(dict(**key, stage=name, candidate_count=len(positions), qualified_count=len(qualified),
                                   covered=coverage[name], max_iou=best, qualified_flats=flat[positions[qualified]].tolist(),
                                   max_iou_flats=flat[positions[best_positions]].tolist()))
        gains = [after for before, after in zip(STAGES, STAGES[1:]) if not coverage[before] and coverage[after]]
        losses = [after for before, after in zip(STAGES, STAGES[1:]) if coverage[before] and not coverage[after]]
        check(not any(stage in gains for stage in STAGES[3:]) and coverage['budget'] == coverage['final'],
              'A filtering-only stage unexpectedly created coverage')

        def candidate(position):
            row = int(before_rows[position])
            return dict(flat=int(flat[row]), roi_row=row, score=float(scores[row]),
                        gt_iou=float(overlaps['before_nms'][position, g]), nms_kept=row in retained_rows,
                        final_kept=row in final_rows)

        gt_rows.append(dict(**key, coverage=coverage,
                            first_covered_stage=next((name for name in STAGES if coverage[name]), None),
                            first_lost_stage=losses[0] if losses else None, gain_stages=gains, loss_stages=losses,
                            top_score_candidates=[candidate(p) for p in top_score_positions],
                            best_iou_candidates=[candidate(p) for p in maximum_iou_positions]))
    return stage_rows, gt_rows


def summaries(gt_rows):
    model_rows, per_stage = {}, []
    for model in MODELS:
        selected = [r for r in gt_rows if r['model'] == model]
        check(len(selected) == 416, f'{model} GT count differs')
        for j, stage in enumerate(STAGES):
            covered = sum(r['coverage'][stage] for r in selected)
            previous = STAGES[j - 1] if j else None
            gains = sum(stage in r['gain_stages'] for r in selected) if j else None
            losses = sum(stage in r['loss_stages'] for r in selected) if j else None
            if j:
                check(per_stage[-1]['covered_gt'] + gains - losses == covered, 'Stage transition accounting differs')
            per_stage.append(dict(model=model, stage=stage, previous_stage=previous, total_gt=len(selected),
                                  covered_gt=covered, gained_gt=gains, lost_gt=losses))
        risk = [r for r in selected if r['coverage']['before_nms']]
        loss = [r for r in risk if not r['coverage']['after_nms']]
        model_rows[model] = dict(total_gt=len(selected), rpn_covered=sum(r['coverage']['rpn'] for r in selected),
                                 first_covered_at_decode=sum(r['first_covered_stage'] == 'decode' for r in selected),
                                 first_covered_at_clip=sum(r['first_covered_stage'] == 'clip' for r in selected),
                                 coverage_lost_at_decode=sum('decode' in r['loss_stages'] for r in selected),
                                 before_nms_covered=len(risk), nms_lost=len(loss), after_nms_covered=len(risk)-len(loss),
                                 nms_loss_rate=len(loss) / len(risk) if risk else None,
                                 budget_lost=sum('budget' in r['loss_stages'] for r in selected),
                                 final_covered=sum(r['coverage']['final'] for r in selected),
                                 never_covered=sum(r['first_covered_stage'] is None for r in selected),
                                 nms_loss_with_top_score_kept=sum(any(c['nms_kept'] for c in r['top_score_candidates']) for r in loss),
                                 nms_loss_with_top_score_tie=sum(len(r['top_score_candidates']) > 1 for r in loss))
    return model_rows, per_stage


def pair(gt_rows):
    lookup = {(r['model'], r['stem'], r['gt_index']): r for r in gt_rows}
    check(len(lookup) == len(gt_rows) == 832, 'Duplicate or missing model-GT identity')
    pairs = []
    for ir in (r for r in gt_rows if r['model'] == 'IR'):
        fusion = lookup['Fusion', ir['stem'], ir['gt_index']]
        check(ir['scene_id'] == fusion['scene_id'], 'Paired scene differs')
        a, b = ir['coverage'], fusion['coverage']
        pairs.append(dict(stem=ir['stem'], gt_index=ir['gt_index'], scene_id=ir['scene_id'], ir_coverage=a,
                          fusion_coverage=b, both_pre_nms=a['before_nms'] and b['before_nms'],
                          ir_nms_loss=a['before_nms'] and not a['after_nms'],
                          fusion_nms_loss=b['before_nms'] and not b['after_nms']))
    risk = [r for r in pairs if r['both_pre_nms']]
    bins = dict(both_preserved=0, only_ir_lost=0, only_fusion_lost=0, both_lost=0)
    for row in risk:
        name = {(False, False): 'both_preserved', (True, False): 'only_ir_lost',
                (False, True): 'only_fusion_lost', (True, True): 'both_lost'}[row['ir_nms_loss'], row['fusion_nms_loss']]
        bins[name] += 1
    check(sum(bins.values()) == len(risk), 'Shared denominator counts differ')
    per_stage = {}
    for stage in STAGES:
        ir = [r['ir_coverage'][stage] for r in pairs]
        fusion = [r['fusion_coverage'][stage] for r in pairs]
        per_stage[stage] = dict(both=sum(a and b for a, b in zip(ir, fusion)),
                                only_ir=sum(a and not b for a, b in zip(ir, fusion)),
                                only_fusion=sum(b and not a for a, b in zip(ir, fusion)),
                                neither=sum(not a and not b for a, b in zip(ir, fusion)))
        check(sum(per_stage[stage].values()) == len(pairs), 'Paired stage counts differ')
    return pairs, dict(shared_pre_nms_covered=len(risk), **bins, per_stage=per_stage)


def verify_anchors(rows):
    results = {(r['model'], r['stem'], r['gt_index'], r['stage']): r for r in rows if r['stem'] in ('030078', '030129')}
    with ANCHORS.open(encoding='utf-8-sig', newline='') as handle:
        old = list(csv.DictReader(handle))
    check(len(old) == 4, 'Four case anchors required')
    for r in old:
        for stage, count_field, iou_field in [('rpn', 'rpn_good_count', 'rpn_max_iou'),
                                              ('decode', 'decoded_good_count', 'decoded_max_iou'),
                                              ('before_nms', 'before_nms_good_count', None),
                                              ('after_nms', 'after_nms_good_count', None),
                                              ('final', 'final_good_count', 'final_max_iou')]:
            actual = results[r['model'], r['stem'], 0, stage]
            check(actual['qualified_count'] == int(r[count_field]) and
                  (iou_field is None or actual['max_iou'] == float(r[iou_field])), f'Day01 anchor differs for {r["stem"]}/{stage}')
    return len(old)


def write_report(out, per_model, stages, paired):
    ir, fusion = (per_model[m] for m in MODELS)
    by_stage = {(r['model'], r['stage']): r for r in stages}
    lines = ['# Week11 Day06 · 全验证集 RoI 阶段统计', '',
             '读取 300 份模型×图片捕获，IR 和 Fusion 共享同一批 416 个标注目标。'
             '以下是目标级几何覆盖，候选与该目标的 IoU ≥ 0.75 时计为达标。', '',
             '| 阶段 | IR 覆盖数 | Fusion 覆盖数 |', '|---|---:|---:|']
    for stage in STAGES:
        lines.append(f'| {LABELS[stage]} | {by_stage["IR", stage]["covered_gt"]} | {by_stage["Fusion", stage]["covered_gt"]} |')
    lines += ['', f'回归解码时首次达标，IR **{ir["first_covered_at_decode"]}** 个，'
               f'Fusion **{fusion["first_covered_at_decode"]}** 个。同一阶段也可能使原先达标的框掉出阈值，阶段表分别记录得失。', '',
               f'类别 NMS 在各自进入 NMS 前已覆盖的目标中，IR 丢失 **{ir["nms_lost"]}/{ir["before_nms_covered"]}**，'
               f'Fusion 丢失 **{fusion["nms_lost"]}/{fusion["before_nms_covered"]}**。', '',
               f'只比较 NMS 前两模型都覆盖的 **{paired["shared_pre_nms_covered"]}** 个目标，'
               f'两者都保住 {paired["both_preserved"]} 个，仅 IR 丢 {paired["only_ir_lost"]} 个，'
               f'仅 Fusion 丢 {paired["only_fusion_lost"]} 个，两者都丢 {paired["both_lost"]} 个。', '',
               '最高分类分数以整张图的 NMS 前候选为范围，逐目标记录该候选与标注的 IoU 及实际去留。'
               '在多人图里，最高分框可以属于另一人。保存高分框与丢失覆盖同时出现的情况，具体抑制关系仍需单独核验。', '',
               '本轮只加载原始候选捕获并计算 IoU，新增模型前向、权重加载和独立 NMS 重放均为 0。']
    (out / 'case_report_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def execute(out, counts):
    scope = load(SCOPE)
    capture_metadata = load(CAPTURE_RUN / 'run_metadata.json')
    check(sha(SCOPE) == capture_metadata['scope_sha256'] and load(CAPTURE_RUN / 'verification.json')['status'] == 'passed',
          'Frozen scope hash or Day05 capture status differs')
    check(scope['environment']['torch'] == torch.__version__ and scope['environment']['torchvision'] == torchvision.__version__,
          'CPU IoU runtime differs from frozen scope')
    entries = load(CAPTURE_RUN / 'capture_index.json')
    check(len(entries) == 300 and [(r['model'], r['stem'], r['action']) for r in entries] ==
          [(r['model'], r['stem'], r['action']) for r in scope['items']], 'Capture index differs from frozen scope')
    samples = {r['stem']: r for r in scope['samples']}
    stage_rows, gt_rows = [], []
    for i, entry in enumerate(entries, 1):
        path = ROOT / entry['path']
        check(entry['status'] == 'passed' and entry['identity_chain_verified'] and sha(path) == entry['sha256'],
              f'Capture integrity failed for {entry["model"]}/{entry["stem"]}')
        raw = torch.load(path, map_location='cpu', weights_only=True)
        counts['candidate_captures_read'] += 1
        try:
            stages, gts = describe_image(entry, samples[entry['stem']], raw,
                                         scope['model_weights'][entry['model']]['configuration'])
        except Exception as error:
            raise ValueError(f'{entry["model"]}/{entry["stem"]}: {error}') from error
        stage_rows.extend(stages)
        gt_rows.extend(gts)
        if i % 75 == 0:
            print(f'{i}/300 saved captures analyzed', flush=True)
    check(len(stage_rows) == 832 * len(STAGES), 'Unexpected stage row count')
    anchor_count = verify_anchors(stage_rows)
    per_model, per_stage = summaries(gt_rows)
    pairs, paired = pair(gt_rows)
    write_csv(out / 'per_gt_stage.csv', stage_rows)
    write_csv(out / 'per_gt_coverage.csv', gt_rows)
    write_csv(out / 'paired_per_gt.csv', pairs)
    write_csv(out / 'stage_summary.csv', per_stage)
    save(out / 'summary.json', dict(by_model=per_model, paired=paired))
    save(out / 'provenance.json', dict(scope=str(SCOPE), scope_sha256=sha(SCOPE),
                                      capture_index=str(CAPTURE_RUN / 'capture_index.json'),
                                      capture_index_sha256=sha(CAPTURE_RUN / 'capture_index.json'),
                                      anchors=str(ANCHORS), anchors_sha256=sha(ANCHORS), source_sha256=sha(Path(__file__))))
    write_report(out, per_model, per_stage, paired)
    return dict(stage_rows=len(stage_rows), model_gt_rows=len(gt_rows), paired_gt_rows=len(pairs),
                day01_case_anchors_verified=anchor_count, gt_iou_threshold=IOU)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    check(out.parent == HERE / 'outputs', 'Choose a new folder directly under week11/day06/outputs')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    counts = dict(candidate_captures_read=0, model_forwards=0, checkpoint_loads=0, independent_nms_replays=0)
    result, error = {}, None
    try:
        result = execute(out, counts)
    except Exception as exc:
        error = str(exc)
        detail = traceback.format_exc()
        (out / 'failure.log').write_text(detail, encoding='utf-8')
        print(detail, file=sys.stderr, flush=True)
    exit_code = int(error is not None)
    save(out / 'verification.json', dict(status='failed' if exit_code else 'passed', exit_code=exit_code, error=error,
                                         command=subprocess.list2cmdline(sys.orig_argv), counts=counts, **result))
    save(out / 'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(f'{"FAILED" if exit_code else "PASSED"}: {out}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
