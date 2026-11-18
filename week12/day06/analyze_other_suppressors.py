"""Analyze retained suppressor positions using verified Week12 replay decisions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'week12/day05'))
from summarize_paired import BATCH, FREEZE, load, read_csv, require, save, sha, tracked, write_csv

PAIR = ROOT / 'week12/day05/outputs/roi_nms_paired_run01'
PILOT = ROOT / 'week12/day02/outputs/roi_nms_pilot_run01'
GROUPS = ('all_at_step_2', 'some_at_step_2', 'first_at_step_3_or_later')


def decisions_for(result, prior_hashes, sources):
    model = result['model']
    if result['origin'] == 'archived_replay':
        base = ROOT / result['replay_source']
        path = tracked(base, 'candidate_decisions.csv', sources)
        sequence_file = 'output_verification.json' if model == 'IR' else 'output_comparison.json'
        sequence = load(tracked(base, sequence_file, sources))
        for p in (path, base / sequence_file):
            require(prior_hashes[p.relative_to(ROOT).as_posix()] == sources[p.relative_to(ROOT).as_posix()],
                    'Archived file differs from Day01 preflight')
        keep = sequence['original_indices']
        require(keep == sequence['replay_indices'] == result['original_keep_indices'], 'Archived keep order differs')
        rows = []
        for r in read_csv(path):
            ir = model == 'IR'
            kept = r['action'] == 'keep'
            rows.append(dict(input_position=int(r['input_position']), flat=int(r['source_flat'] if ir else r['flat']),
                             classification_score=float(r['classification_score'] if ir else r['score']),
                             action=r['action'], step=int(r['step']) + 1 if ir else int(r['replay_step']),
                             suppressor_flat=None if kept else int(r['selected_flat'] if ir else r['first_suppressor_flat']),
                             pair_iou=None if kept else float(r['path_iou'] if ir else r['pair_iou'])))
        size = result['complete_decision_count']
    else:
        base = PILOT if result['origin'] == 'pilot_replay' else BATCH[model]
        directory = (ROOT / result['evidence']).parent.relative_to(base)
        path = tracked(base, (directory / 'candidate_decisions.json').as_posix(), sources)
        comp = load(tracked(base, (directory / 'output_comparison.json').as_posix(), sources))
        keep = comp['captured_keep']
        require(all(comp['checks'].values()) and keep == comp['replay_keep'], 'Original keep order does not match')
        rows = load(path)
        size = result['nms_input_count']
    require(sorted(r['input_position'] for r in rows) == list(range(size)) and
            len({r['flat'] for r in rows}) == size, 'Candidate decision coverage differs')
    kept = sorted((r for r in rows if r['action'] == 'keep'), key=lambda r: r['step'])
    require([r['input_position'] for r in kept] == keep and [r['step'] for r in kept] == list(range(1, len(keep) + 1)),
            'The retained step numbers do not match the ordered indices')
    by_flat = {r['flat']: r for r in rows}
    scores = [r['classification_score'] for r in rows]
    require(len(set(scores)) == len(scores), 'Original classification rank is ambiguous')
    ranked = {r['flat']: place for place, r in enumerate(sorted(rows, key=lambda r: -r['classification_score']), 1)}
    require(ranked[kept[0]['flat']] == 1, 'First retained box does not have the top original classification score')
    return by_flat, ranked, path.relative_to(ROOT).as_posix()


def trace_one(row, model, result, lookup):
    g = int(row['gt_index'])
    target = next(t for t in result['targets'] if t['gt_index'] == g)
    require(target['category'] == 'none' and target['top_score_first_suppressed'] == 0, 'Unexpected target attribution')
    decisions, ranks, source = lookup
    good_rows = target['first_suppressor_by_flat']
    require(len(good_rows) == target['qualified_count'] > 0, 'Incomplete qualified candidate identities')
    key = dict(model=model, stem=row['stem'], gt_index=g)
    attribution = []
    for flat, suppressor in good_rows.items():
        dropped, selected = decisions[int(flat)], decisions[suppressor]
        require(dropped['action'] == 'delete' and dropped['step'] == selected['step'] > 1 and
                dropped['suppressor_flat'] == suppressor and selected['action'] == 'keep' and
                ranks[suppressor] > 1 and dropped['pair_iou'] > .5, 'Saved first suppressor differs')
        attribution.append(dict(**key, qualified_flat=int(flat), suppressor_flat=suppressor,
                                nms_keep_step=selected['step'], suppressor_original_score_rank=ranks[suppressor],
                                suppressor_score=selected['classification_score'], pair_iou=dropped['pair_iou'], evidence=source))
    at_two = sum(a['nms_keep_step'] == 2 for a in attribution)
    unique = {a['suppressor_flat'] for a in attribution}
    group = 'all_at_step_2' if at_two == len(attribution) else 'some_at_step_2' if at_two else 'first_at_step_3_or_later'
    return dict(**key, qualified_count=len(attribution), distinct_suppressors=len(unique),
                first_suppressor_keep_step=min(a['nms_keep_step'] for a in attribution),
                last_suppressor_keep_step=max(a['nms_keep_step'] for a in attribution),
                lowest_suppressor_original_score_rank=min(a['suppressor_original_score_rank'] for a in attribution),
                highest_suppressor_original_score_rank=max(a['suppressor_original_score_rank'] for a in attribution),
                category=group), attribution


def execute(out):
    sources = {}
    preflight = load(tracked(FREEZE, 'verification.json', sources))
    paired = read_csv(tracked(PAIR, 'paired_target_attributions.csv', sources))
    summary = load(tracked(PAIR, 'summary.json', sources))
    pair_status = load(tracked(PAIR, 'verification.json', sources))
    require(pair_status['status'] == 'passed' and len(paired) == summary['shared_pre_nms_gt'] == 297, 'Paired scope differs')
    batches = {m: {r['stem']: r for r in load(tracked(path, 'batch_index.json', sources))} for m, path in BATCH.items()}
    anchors = {}
    targets, candidate_rows = [], []
    for row in paired:
        for model in BATCH:
            if row[f'{model.lower()}_outcome'] != 'none':
                continue
            result = batches[model][row['stem']]
            require(result['status'] == 'passed' and result['full_output_match'] and
                    result['evidence'] == row[f'{model.lower()}_evidence'], 'Paired attribution points to different evidence')
            key = model, row['stem']
            if key not in anchors:
                anchors[key] = decisions_for(result, preflight['input_artifacts'], sources)
            target, candidates = trace_one(row, model, result, anchors[key])
            targets.append(target)
            candidate_rows.extend(candidates)
    by_model = {}
    for model in BATCH:
        rows = [r for r in targets if r['model'] == model]
        require(len(rows) == summary['paired_outcome_counts'][model]['none'], 'Number of paired zero-by-top targets differs')
        counts = Counter(r['category'] for r in rows)
        by_model[model] = dict(targets=len(rows), by_retained_step_group={k: counts[k] for k in GROUPS},
                               one_suppressor=sum(r['distinct_suppressors'] == 1 for r in rows),
                               multiple_suppressors=sum(r['distinct_suppressors'] > 1 for r in rows))
    diff = {k: by_model['Fusion']['by_retained_step_group'][k] - by_model['IR']['by_retained_step_group'][k] for k in GROUPS}
    require(sum(diff.values()) == summary['paired_loss_difference_by_category']['none'], 'Day05 +18 does not reconcile')
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'other_suppressors_per_gt.csv', targets)
    write_csv(out / 'other_suppressors_per_candidate.csv', candidate_rows)
    save(out / 'summary.json', dict(shared_pre_nms_gt=len(paired), by_model=by_model, fusion_minus_ir_by_step_group=diff))
    save(out / 'verification.json', dict(status='passed', loss_targets=len(targets), qualified_candidates=len(candidate_rows),
                                         replay_decision_tables_read=len(anchors), new_replays=0, model_forwards=0,
                                         original_capture_tensor_reads=0, checkpoint_loads=0,
                                         source_sha256=sha(Path(__file__)), input_artifacts=sources,
                                         command=subprocess.list2cmdline(sys.orig_argv)))
    save(out / 'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(by_model=by_model, fusion_minus_ir=diff), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    require(out.parent == HERE / 'outputs', 'Choose a new folder directly under week12/day06/outputs')
    execute(out)
