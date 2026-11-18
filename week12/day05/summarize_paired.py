"""Summarize same-GT RoI NMS attribution on the frozen Week12 cohort."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = ROOT / 'week12/day01/outputs/frozen_roi_nms_queue_run01'
W11 = ROOT / 'week11/day06/outputs/full_val_roi_analysis_run01'
BATCH = {'IR': ROOT / 'week12/day03/outputs/ir_roi_nms_batch_run01',
         'Fusion': ROOT / 'week12/day04/outputs/fusion_roi_nms_batch_run01'}
LOSS_CATEGORIES = ('all', 'partial', 'none', 'unverified')
OUTCOMES = ('preserved', *LOSS_CATEGORIES)


def require(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tracked(base, name, sources):
    path = base / name
    found = sha(path)
    require(found == load(base / 'artifact_manifest.json')[name], f'Source hash differs: {path}')
    sources[path.relative_to(ROOT).as_posix()] = found
    return path


def validated_losses(model, expected, queue_hash, sources):
    base = BATCH[model]
    status = load(tracked(base, 'verification.json', sources))
    rows = read_csv(tracked(base, 'target_attributions.csv', sources))
    require(status['model'] == model and status['queue_sha256'] == queue_hash and
            status['status'] in ('passed', 'complete_with_unverified_jobs'), 'Batch scope or status differs')
    by_gt = {(r['stem'], int(r['gt_index'])): r for r in rows}
    require(len(rows) == len(by_gt) == len(expected) == status['queued_targets'] and by_gt.keys() == expected.keys(),
            f'{model} result does not cover all frozen loss targets')
    for key, row in by_gt.items():
        entry = expected[key]
        count = len(entry['qualified_flats_before_nms'])
        require(row['category'] in LOSS_CATEGORIES and row['model'] == model and
                row['scene_id'] == entry['scene_id'] and int(row['qualified_count']) == count > 0 and
                row['both_models_pre_nms_covered'] == str(entry['both_models_pre_nms_covered']),
                'Target attribution disagrees with frozen scope')
        if row['category'] == 'unverified':
            require(not row['top_score_first_suppressed'] and bool(row['reason']), 'Unverified row lacks explicit gap')
        else:
            top = int(row['top_score_first_suppressed'])
            require(0 <= top <= count and row['category'] == ('all' if top == count else 'partial' if top else 'none'),
                    'Target category disagrees with recorded direct suppression count')
    all_counts = Counter(r['category'] for r in rows)
    shared = Counter(r['category'] for r in rows if r['both_models_pre_nms_covered'] == 'True')
    require({key: all_counts[key] for key in LOSS_CATEGORIES} == status['categories'] and
            {key: shared[key] for key in LOSS_CATEGORIES} == status['paired_loss_categories'], 'Batch totals do not reconstruct')
    return by_gt, status


def summarize(out):
    sources = {}
    queue_path = tracked(FREEZE, 'replay_queue.json', sources)
    audit = load(tracked(FREEZE, 'verification.json', sources))
    paired_path = tracked(W11, 'paired_per_gt.csv', sources)
    w11_summary = load(tracked(W11, 'summary.json', sources))['paired']
    require(audit['input_artifacts'][paired_path.relative_to(ROOT).as_posix()] == sha(paired_path),
            'Paired GT file is not the source frozen on Day01')
    queue = load(queue_path)
    expected = {m: {(job['stem'], t['gt_index']): t for job in queue['entries'] if job['model'] == m
                    for t in job['targets']} for m in BATCH}
    both = read_csv(paired_path)
    for model in BATCH:
        require(set(expected[model]) == {(r['stem'], int(r['gt_index'])) for r in both
                                        if r[f'{model.lower()}_nms_loss'] == 'True'},
                'Frozen loss queue differs from Week11 paired loss flags')
    losses, statuses = {}, {}
    for model in BATCH:
        losses[model], statuses[model] = validated_losses(model, expected[model], sha(queue_path), sources)
    require(len({s['source_sha256'] for s in statuses.values()}) ==
            len({s['replay_source_sha256'] for s in statuses.values()}) == 1, 'IR/Fusion replay implementations differ')

    paired = []
    for row in both:
        if row['both_pre_nms'] != 'True':
            continue
        key = row['stem'], int(row['gt_index'])
        a, b = (losses[m].get(key) for m in ('IR', 'Fusion'))
        paired.append(dict(stem=key[0], gt_index=key[1], scene_id=row['scene_id'],
                           ir_outcome=a['category'] if a else 'preserved',
                           fusion_outcome=b['category'] if b else 'preserved',
                           ir_evidence=a['evidence'] if a else '', fusion_evidence=b['evidence'] if b else ''))
    require(len(paired) == len({(p['stem'], p['gt_index']) for p in paired}) == queue['shared_pre_nms_covered_gt'] == 297,
            'Shared pre-NMS denominator differs')
    matrix = Counter((p['ir_outcome'], p['fusion_outcome']) for p in paired)
    blocks = dict(both_preserved=matrix['preserved', 'preserved'],
                  only_ir_lost=sum(matrix[a, 'preserved'] for a in LOSS_CATEGORIES),
                  only_fusion_lost=sum(matrix['preserved', b] for b in LOSS_CATEGORIES),
                  both_lost=sum(matrix[a, b] for a in LOSS_CATEGORIES for b in LOSS_CATEGORIES))
    require(all(w11_summary[key] == value for key, value in blocks.items()), 'Paired loss blocks differ from Week11')
    table = []
    for category in OUTCOMES:
        ir = sum(matrix[category, b] for b in OUTCOMES)
        fusion = sum(matrix[a, category] for a in OUTCOMES)
        table.append(dict(outcome=category, ir_count=ir, fusion_count=fusion, fusion_minus_ir=fusion-ir))
    shared_counts = {model: {r['outcome']: r[f'{model.lower()}_count'] for r in table} for model in BATCH}
    for model in BATCH:
        require({key: shared_counts[model][key] for key in LOSS_CATEGORIES} == statuses[model]['paired_loss_categories'],
                'Paired category marginals differ from verified batch data')
    differences = {r['outcome']: r['fusion_minus_ir'] for r in table if r['outcome'] != 'preserved'}
    net = sum(differences.values())
    require(net == blocks['only_fusion_lost'] - blocks['only_ir_lost'] == 25, 'Net paired loss decomposition differs')

    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'paired_target_attributions.csv', paired)
    write_csv(out / 'paired_outcome_counts.csv', table)
    write_csv(out / 'paired_outcome_matrix.csv', [dict(ir_outcome=a, **{b: matrix[a, b] for b in OUTCOMES}) for a in OUTCOMES])
    save(out / 'summary.json', dict(total_frozen_loss_targets={m: statuses[m]['queued_targets'] for m in BATCH},
                                    full_loss_queue_counts={m: statuses[m]['categories'] for m in BATCH},
                                    shared_pre_nms_gt=len(paired), paired_loss_blocks=blocks, paired_outcome_counts=shared_counts,
                                    paired_loss_difference_by_category=differences, paired_net_extra_fusion_loss=net,
                                    only_ir_lost_categories={a: matrix[a, 'preserved'] for a in LOSS_CATEGORIES},
                                    only_fusion_lost_categories={b: matrix['preserved', b] for b in LOSS_CATEGORIES}))
    save(out / 'verification.json', dict(status='passed', shared_rows=len(paired), loss_rows_checked=sum(len(r) for r in losses.values()),
                                         attribution_verified_loss_rows=sum(len(r) for r in losses.values()) -
                                         sum(s['categories']['unverified'] for s in statuses.values()),
                                         full_output_matches_by_model={m: statuses[m]['full_output_matches'] for m in BATCH},
                                         new_replays=0, candidate_captures_read=0, model_forwards=0, checkpoint_loads=0,
                                         source_sha256=sha(Path(__file__)), input_artifacts=sources,
                                         command=subprocess.list2cmdline(sys.orig_argv)))
    save(out / 'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(f'PASS: {len(paired)} paired GTs, fusion minus IR loss {net}: {differences}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output.resolve()
    require(output.parent == HERE / 'outputs', 'Choose a new folder directly under week12/day05/outputs')
    summarize(output)
