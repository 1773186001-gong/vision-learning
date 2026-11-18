"""Freeze the RoI NMS loss cohort and check three saved replay trajectories."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
import torchvision
from torchvision.ops import box_iou

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ('IR', 'Fusion')
ANALYSIS = ROOT / 'week11/day06/outputs/full_val_roi_analysis_run01'
INDEX = ROOT / 'week11/day05/outputs/full_val_roi_capture_run01/capture_index.json'
SCOPE = ROOT / 'week11/day04/outputs/full_val_scope_run02/scope.json'
SAVED = {
    ('IR', '030129'): ROOT / 'week10/day05/outputs/roi_nms_replay_run01',
    ('Fusion', '030129'): ROOT / 'week11/day02/outputs/fusion_nms_030129_run01',
    ('Fusion', '030078'): ROOT / 'week11/day03/outputs/fusion_nms_030078_run01',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def tracked(base, name, manifest, fingerprints):
    path = base / name
    actual = sha(path)
    require(actual == manifest[name], f'Artifact SHA differs: {path}')
    fingerprints[rel(path)] = actual
    return path


def loss_queue(fingerprints):
    manifest = load(ANALYSIS / 'artifact_manifest.json')
    sources = {name: tracked(ANALYSIS, name, manifest, fingerprints) for name in (
        'verification.json', 'provenance.json', 'summary.json', 'per_gt_coverage.csv', 'per_gt_stage.csv', 'paired_per_gt.csv')}
    check = load(sources['verification.json'])
    require(check['status'] == 'passed' and check['gt_iou_threshold'] == .75 and check['paired_gt_rows'] == 416,
            'Unexpected Week11 analysis status or scope')
    provenance = load(sources['provenance.json'])
    for path, key in ((INDEX, 'capture_index_sha256'), (SCOPE, 'scope_sha256')):
        fingerprints[rel(path)] = sha(path)
        require(fingerprints[rel(path)] == provenance[key], f'Scope or capture index differs: {path}')
    env = load(SCOPE)['environment']
    require(env['device'] == 'cpu' and env['torch'] == torch.__version__ and env['torchvision'] == torchvision.__version__,
            'Runtime differs from the Week11 CPU capture')
    summary = load(sources['summary.json'])
    paired = {(r['stem'], int(r['gt_index'])): r for r in read_csv(sources['paired_per_gt.csv'])}
    require(len(paired) == 416 and sum(r['both_pre_nms'] == 'True' for r in paired.values()) == 297,
            'Paired cohort differs')
    stage = {(r['model'], r['stem'], int(r['gt_index']), r['stage']): r
             for r in read_csv(sources['per_gt_stage.csv']) if r['stage'] in ('before_nms', 'after_nms')}
    require(len(stage) == 2 * 832, 'Incomplete before/after NMS GT stages')
    jobs = {}
    for row in read_csv(sources['per_gt_coverage.csv']):
        model, stem, g = row['model'], row['stem'], int(row['gt_index'])
        cover = json.loads(row['coverage'])
        p = paired[stem, g]
        require(cover == json.loads(p[f'{model.lower()}_coverage']), 'Coverage and paired GT table differ')
        if not cover['before_nms'] or cover['after_nms']:
            continue
        before, after = [stage[model, stem, g, name] for name in ('before_nms', 'after_nms')]
        good = json.loads(before['qualified_flats'])
        require(len(good) == len(set(good)) == int(before['qualified_count']) > 0 and int(after['qualified_count']) == 0,
                'Loss target does not match candidate identities')
        top = json.loads(row['top_score_candidates'])
        require(top and all(c['nms_kept'] for c in top), 'Highest-score candidate record needs review')
        jobs.setdefault((model, stem), []).append(dict(gt_index=g, scene_id=row['scene_id'],
                   both_models_pre_nms_covered=p['both_pre_nms'] == 'True', qualified_flats_before_nms=good,
                   top_score_flats=[c['flat'] for c in top], nms_input_count=int(before['candidate_count'])))
    index = load(INDEX)
    unique_index = {(r['model'], r['stem']) for r in index}
    require(len(index) == len(unique_index) == 300 and set(jobs) <= unique_index, 'Capture index is incomplete')
    target_counts = {m: sum(len(targets) for (model, _), targets in jobs.items() if model == m) for m in MODELS}
    require(target_counts == {m: summary['by_model'][m]['nms_lost'] for m in MODELS} == {'IR': 97, 'Fusion': 124},
            'Loss target counts differ')
    paired_loss = {m: sum(t['both_models_pre_nms_covered'] for (model, _), targets in jobs.items() if model == m for t in targets)
                   for m in MODELS}
    require(paired_loss == {'IR': 80, 'Fusion': 105}, 'Shared-denominator loss counts differ')
    queue = []
    for r in index:
        key = (r['model'], r['stem'])
        if key not in jobs:
            continue
        require(r['status'] == 'passed' and r['identity_chain_verified'] and sha(ROOT / r['path']) == r['sha256'],
                f'Capture integrity fails for {key}')
        targets = sorted(jobs[key], key=lambda t: t['gt_index'])
        require(all(t['gt_index'] < r['gt_count'] and t['nms_input_count'] == r['nms_input_count'] for t in targets),
                f'GT or NMS input count differs for {key}')
        queue.append(dict(model=key[0], stem=key[1], action='reuse_replay' if key in SAVED else 'new_replay',
                          capture_path=r['path'], capture_sha256=r['sha256'], run_identity=r['run_identity'],
                          nms_input_count=r['nms_input_count'], targets=targets))
    counts = {m: dict(Counter(item['action'] for item in queue if item['model'] == m)) for m in MODELS}
    require(counts == {'IR': {'reuse_replay': 1, 'new_replay': 67},
                       'Fusion': {'reuse_replay': 2, 'new_replay': 81}}, 'Model-image replay inventory differs')
    return queue, target_counts, paired_loss, counts


def checked_history(job, fingerprints):
    model, stem = job['model'], job['stem']
    base = SAVED[model, stem]
    manifest = load(base / 'artifact_manifest.json')
    compare_name = 'output_verification.json' if model == 'IR' else 'output_comparison.json'
    def read_source(name):
        return tracked(base, name, manifest, fingerprints)

    status = load(read_source('verification.json'))
    compare = load(read_source(compare_name))
    prov = load(read_source('input_provenance.json'))
    decisions_path = read_source('candidate_decisions.csv')
    valid = status['attribution_validated'] if model == 'IR' else status['identity_chain_verified'] and status['keep_order_exact']
    require(status['status'] == 'passed' and valid, 'Saved replay attribution or verification was not accepted')
    if model == 'IR':
        raw_hashes = [r['sha256'] for r in prov if r['path'].replace('\\', '/').endswith(job['capture_path'])]
        require(raw_hashes == [job['capture_sha256']] and compare['full_index_sequence_equal'] and
                compare['replay_budget_and_model_exact'], 'IR archived source or output verification differs')
    else:
        require(prov['files'][f'{stem}/raw_capture.pth'] == job['capture_sha256'] and all(compare['checks'].values()),
                'Fusion archived source or output verification differs')
    raw = torch.load(ROOT / job['capture_path'], map_location='cpu', weights_only=True)
    require(raw['run_identity'] == job['run_identity'], 'Original raw capture identity differs')
    boxes, scores, labels, threshold = raw['nms']['args']
    keep = raw['nms']['result'].tolist()
    require(keep == compare['original_indices'] == compare['replay_indices'] and threshold == .5,
            'Saved complete keep sequence or threshold differs')
    probs = raw['softmax']['result'][:, 1]
    score_rows = torch.where(probs > raw['configuration']['roi']['score'])[0]
    roi_rows = score_rows[raw['small']['result']]
    require(torch.equal(raw['clip']['result'][roi_rows, 1], boxes) and torch.equal(probs[roi_rows], scores) and
            bool((labels == 1).all()), 'Original stage rows do not map to captured NMS input')
    flat = raw['rpn_stages'][0]['post_nms_topn']['ids'][roi_rows].tolist()
    decisions = read_csv(decisions_path)
    n = job['nms_input_count']
    require(len(decisions) == len(flat) == len(set(flat)) == n and
            sorted(int(r['input_position']) for r in decisions) == list(range(n)), 'Saved decisions do not enumerate all input positions')

    retained = {}
    deleted = {}
    for r in decisions:
        i = int(r['input_position'])
        identity_key, score_key = ('source_flat', 'classification_score') if model == 'IR' else ('flat', 'score')
        require(int(r[identity_key]) == flat[i] and float(r[score_key]) == float(scores[i]), 'Archived row identity or score differs')
        step = int(r['step']) + 1 if model == 'IR' else int(r['replay_step'])
        if r['action'] == 'keep':
            require(step not in retained, 'Duplicate keep step in saved decision table')
            retained[step] = i
        else:
            require(r['action'] == 'delete' and i not in keep and 1 <= step <= len(keep), 'Invalid saved deletion decision')
            suppressor = int(r['selected_flat'] if model == 'IR' else r['first_suppressor_flat'])
            require(suppressor == flat[keep[step - 1]] and scores[keep[step - 1]] > scores[i],
                    'Saved suppressor is not the higher-scoring retained candidate at that step')
            deleted[flat[i]] = suppressor
    require([retained[i] for i in sorted(retained)] == keep and len(deleted) + len(keep) == n,
            'Saved per-input decisions do not reproduce complete ordered keep indices')
    top = torch.where(scores == scores.max())[0].tolist()
    require(len(top) == 1 and top[0] == keep[0], 'Expected an unambiguous top-score candidate in the saved replay')
    top_flat = flat[top[0]]
    result = []
    for target in job['targets']:
        g = target['gt_index']
        good = [flat[i] for i in torch.where(box_iou(boxes, raw['gt_all'][g:g + 1])[:, 0] >= .75)[0].tolist()]
        require(good == target['qualified_flats_before_nms'] and target['top_score_flats'] == [top_flat] and
                all(f in deleted for f in good), 'Frozen target flats or their loss differs from the archived inputs')
        count = sum(deleted[f] == top_flat for f in good)
        category = 'all' if count == len(good) else 'none' if count == 0 else 'partial'
        result.append(dict(gt_index=g, top_score_flat=top_flat, qualified_count=len(good),
                           top_score_first_suppressed=count, category=category,
                           first_suppressor_by_flat={f: deleted[f] for f in good}))
    expected_g0 = {'IR': 'partial', 'Fusion': 'all'}[model]
    require(next(r['category'] for r in result if r['gt_index'] == 0) == expected_g0, 'Archived G0 learning anchor differs')
    return dict(model=model, stem=stem, replay_source=rel(base), original_keep_indices=keep,
                complete_decision_count=n, targets=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    out = parser.parse_args().output.resolve()
    require(out.parent == HERE / 'outputs', 'Choose a new folder directly under week12/day01/outputs')
    torch.set_num_threads(2)
    sources = {}
    queue, target_counts, paired_loss, per_model = loss_queue(sources)
    anchors = [checked_history(item, sources) for item in queue if item['action'] == 'reuse_replay']
    require(len(anchors) == 3 and sum(len(a['targets']) for a in anchors) == 5, 'Expected three accepted archived replays')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'replay_queue.json', dict(status='frozen', definition='covered before RoI class NMS and not covered afterward',
                                         original_nms_iou_threshold=.5, gt_iou_threshold=.75, total_paired_gt=416,
                                         shared_pre_nms_covered_gt=297, loss_targets=target_counts, shared_cohort_loss_targets=paired_loss,
                                         model_image_jobs=len(queue), by_model=per_model, entries=queue))
    save(out / 'reused_replay_checks.json', anchors)
    save(out / 'verification.json', dict(status='passed', checked_capture_files=len(queue), checked_saved_replays=len(anchors),
                                         checked_saved_loss_targets=sum(len(a['targets']) for a in anchors),
                                         complete_model_image_jobs=len(queue), queued_loss_targets=sum(target_counts.values()),
                                         planned_new_replays=sum(d['new_replay'] for d in per_model.values()),
                                         new_replays=0, model_forwards=0, checkpoint_loads=0, source_sha256=sha(Path(__file__)),
                                         input_artifacts=sources))
    save(out / 'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(f'PASS: {len(queue)} picture-model jobs, {sum(target_counts.values())} lost GTs, 3 saved replays, 0 new replays')
    return 0


if __name__ == '__main__':
    sys.exit(main())
