"""Replay one model's pending jobs, preserving the Day01 and Day02 evidence."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import subprocess
import sys
import traceback

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAY02 = ROOT / 'week12/day02'
PILOT = DAY02 / 'outputs/roi_nms_pilot_run01'
sys.path.insert(0, str(DAY02))
from replay_pilot import BASELINE, FREEZE, MODELS, load, load_queue, replay_job, require, save, sha

CATEGORIES = ('all', 'partial', 'none', 'unverified')
BATCHES = {'IR': (66, 'day03'), 'Fusion': (80, 'day04')}


def checked(base, name):
    path = base / name
    require(sha(path) == load(base / 'artifact_manifest.json')[name], f'Prior artifact SHA differs: {path}')
    return path


def reused_results():
    archive = checked(FREEZE, 'reused_replay_checks.json')
    prior = checked(PILOT, 'pilot_index.json')
    pilot_verification = load(checked(PILOT, 'verification.json'))
    require(pilot_verification['status'] == 'passed' and pilot_verification['counts']['new_replays'] == 2 and
            pilot_verification['queue_sha256'] == sha(FREEZE / 'replay_queue.json') and
            pilot_verification['source_sha256'] == sha(DAY02 / 'replay_pilot.py') and
            pilot_verification['greedy_replay_source_sha256'] == sha(BASELINE / 'replay_roi_nms.py'),
            'Pilot queue or checked implementation differs')
    reused = {}
    for item in load(archive):
        key = item['model'], item['stem']
        reused[key] = dict(**item, status='passed', origin='archived_replay', evidence=archive.relative_to(ROOT).as_posix(),
                           full_output_match=True)
    for item in load(prior):
        key = item['model'], item['stem']
        evidence = checked(PILOT, f'{key[0]}_{key[1]}/result.json')
        require(key not in reused and item['status'] == 'passed' and item == load(evidence), 'Pilot result differs')
        reused[key] = dict(**item, origin='pilot_replay', evidence=evidence.relative_to(ROOT).as_posix())
    require(len(reused) == 5, 'Expected three archived and two pilot replays')
    return reused


def run_job(job, folder, counts, budget):
    folder.mkdir()
    before = counts['new_replays']
    try:
        result = replay_job(job, folder, counts, replay_budget=budget)
    except (ValueError, OSError) as error:
        (folder / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        comparison = folder / 'output_comparison.json'
        attribution = folder / 'attribution_checks.json'
        result = dict(model=job['model'], stem=job['stem'], status='unverified', error=str(error),
                      full_output_match=all(load(comparison)['checks'].values()) if comparison.is_file() else None,
                      ambiguity_details=load(attribution)['ambiguities'] if attribution.is_file() else [], targets=[])
    result.update(origin='batch_replay', replay_attempted=counts['new_replays'] - before == 1,
                  evidence=(folder / 'result.json').relative_to(ROOT).as_posix())
    save(folder / 'result.json', result)
    return result


def target_rows(job, result):
    require((job['model'], job['stem']) == (result['model'], result['stem']), 'Replay/capture identity differs')
    if 'capture_sha256' in result:
        require(job['capture_sha256'] == result['capture_sha256'], 'Replay uses a different raw capture')
    verified = result['status'] == 'passed'
    by_gt = {r['gt_index']: r for r in result['targets']}
    expected_gt = {t['gt_index'] for t in job['targets']}
    require((verified and len(by_gt) == len(result['targets']) and by_gt.keys() == expected_gt) or
            (not verified and not by_gt), 'Result GT and frozen GT queue differ')
    for target in job['targets']:
        g = target['gt_index']
        answer = by_gt.get(g)
        good_count = len(target['qualified_flats_before_nms'])
        if verified:
            require(answer['qualified_count'] == good_count and answer['category'] in CATEGORIES[:-1] and
                    set(map(int, answer['first_suppressor_by_flat'])) == set(target['qualified_flats_before_nms']) and
                    [answer['top_score_flat']] == target['top_score_flats'], 'Attribution differs from frozen candidate IDs')
        yield dict(model=job['model'], stem=job['stem'], gt_index=g, scene_id=target['scene_id'],
                   both_models_pre_nms_covered=target['both_models_pre_nms_covered'], origin=result['origin'],
                   category=answer['category'] if verified else 'unverified', qualified_count=good_count,
                   top_score_first_suppressed=answer['top_score_first_suppressed'] if verified else None,
                   reason=result.get('error'), evidence=result['evidence'])


def run_batch(model, budget, out, queued, reused):
    counts = dict(candidate_capture_loads=0, new_replays=0, full_output_verifications=0,
                  model_forwards=0, checkpoint_loads=0)
    (out / 'items').mkdir(parents=True, exist_ok=False)
    results, rows = [], []
    processed_new = 0
    for job in queued:
        key = job['model'], job['stem']
        if key in reused:
            result = reused[key]
        else:
            result = run_job(job, out / 'items' / job['stem'], counts, budget)
            processed_new += 1
            if processed_new == 1 or processed_new % 10 == 0 or processed_new == budget or result['status'] != 'passed':
                print(f'{model}: {processed_new}/{budget} pending jobs processed, latest {job["stem"]} {result["status"]}',
                      flush=True)
        results.append(result)
        rows.extend(target_rows(job, result))
    require(processed_new == budget and len(results) == len(queued), 'Batch coverage differs from frozen pending jobs')
    with (out / 'target_attributions.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(out / 'batch_index.json', results)
    all_counts = Counter(r['category'] for r in rows)
    paired_counts = Counter(r['category'] for r in rows if r['both_models_pre_nms_covered'])
    unverified = [dict(stem=r['stem'], reason=r.get('error'), ambiguity_details=r.get('ambiguity_details', []),
                       full_output_match=r['full_output_match'], evidence=r['evidence'])
                  for r in results if r['status'] != 'passed']
    save(out / 'verification.json', dict(status='passed' if not unverified else 'complete_with_unverified_jobs', model=model,
                                         queued_jobs=len(queued), queued_targets=len(rows),
                                         archived_jobs=sum(r['origin'] == 'archived_replay' for r in results),
                                         pilot_jobs=sum(r['origin'] == 'pilot_replay' for r in results),
                                         full_output_matches=sum(r['full_output_match'] is True for r in results),
                                         categories={c: all_counts[c] for c in CATEGORIES},
                                         paired_loss_categories={c: paired_counts[c] for c in CATEGORIES},
                                         unverified_jobs=unverified, counts=counts,
                                         queue_sha256=sha(FREEZE / 'replay_queue.json'), source_sha256=sha(Path(__file__)),
                                         replay_source_sha256=sha(DAY02 / 'replay_pilot.py'),
                                         command=subprocess.list2cmdline(sys.orig_argv)))
    save(out / 'artifact_manifest.json', {p.relative_to(out).as_posix(): sha(p)
                                            for p in sorted(out.rglob('*')) if p.is_file()})
    print(f'{model}: {len(queued)} jobs, {len(rows)} targets, {len(unverified)} unverified jobs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    budget, day = BATCHES[args.model]
    out = args.output.resolve()
    require(out.parent == HERE.parent / day / 'outputs', f'Choose a new folder directly under week12/{day}/outputs')
    queue = load_queue()
    reused = reused_results()
    selected = [job for job in queue['entries'] if job['model'] == args.model]
    require(sum((job['model'], job['stem']) not in reused for job in selected) == budget and
            sum(len(job['targets']) for job in selected) == queue['loss_targets'][args.model],
            'Model batch or reuse scope differs')
    torch.set_num_threads(2)
    run_batch(args.model, budget, out, selected, reused)
    return 0


if __name__ == '__main__':
    sys.exit(main())
