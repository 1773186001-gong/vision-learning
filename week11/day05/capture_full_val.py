"""Collect the missing full RoI captures in the frozen Week11 validation scope."""

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'week11/day01'))
from trace_fusion import capture, config, digest, eq, ir, load, normalized, require, restore_model, save, sha

SCOPE = ROOT / 'week11/day04/outputs/full_val_scope_run02/scope.json'
SCOPE_SHA = 'c5437e17b368d52803b44eab20c270ceb8d990e2245aa2804c6bcfab0d237c59'
DATA = ROOT / 'data/LLVIP_1000'
MODELS = ('IR', 'Fusion')
PER_MODEL_NEW = 148


def preflight():
    require(sha(SCOPE) == SCOPE_SHA, 'Frozen scope SHA differs')
    scope = load(SCOPE)
    frozen_env = scope['environment']
    require(frozen_env == dict(device='cpu', python=sys.version, torch=torch.__version__,
                               torchvision=torchvision.__version__), 'CPU environment differs from frozen scope')
    samples, items = scope['samples'], scope['items']
    stems = [sample['stem'] for sample in samples]
    require(scope['status'] == 'ready_for_review' and not scope['compatibility_gaps'] and
            len(stems) == len(set(stems)) == 150 and sum(len(s['gt_boxes_xyxy_640']) for s in samples) == 416,
            'Frozen data scope or reuse status differs')
    require([(i['model'], i['stem']) for i in items] == [(model, stem) for model in MODELS for stem in stems],
            'Frozen model-image scope or order differs')
    plan = scope['proposed_run']
    require((plan['new_forwards'], plan['reusable_captures'], plan['checkpoint_loads'],
             plan['fusion_structure_probes']) == (296, 4, 2, 1), 'Frozen execution budget differs')
    require(sha(ROOT / scope['source_manifest']['path']) == scope['source_manifest']['sha256'],
            'Dataset manifest SHA differs')
    for model in MODELS:
        actions = Counter(i['action'] for i in items if i['model'] == model)
        require(actions == {'new_forward': PER_MODEL_NEW, 'reuse_capture': 2}, f'{model} budget differs')
        weight = scope['model_weights'][model]
        require(sha(ROOT / weight['path']) == weight['sha256'], f'{model} checkpoint hash differs')
    for sample in samples:
        for entry in sample['files'].values():
            require(sha(ROOT / entry['path']) == entry['sha256'], f'Frozen input changed: {entry["path"]}')
    for item in items:
        if item['action'] == 'reuse_capture':
            entry = item['capture']
            require(sha(ROOT / entry['path']) == entry['sha256'], f'Reuse capture changed: {entry["path"]}')
    # Estimate space from both original models and leave a margin for per-image files and metadata.
    capacity = shutil.disk_usage(HERE).free
    require(capacity >= 2 * 1024**3, f'Only {capacity} bytes available for capture output')
    return scope


def verify_chain(raw, expected_configuration):
    require(raw['hooks_restored'] and raw['temporary_attributes_restored'], 'Temporary capture state was not restored')
    require(raw['configuration'] == expected_configuration, 'Live inference configuration changed')
    proposals = raw['actual_roi_input'][0][0]
    record = raw['rpn_records'][0]
    rpn_end = raw['rpn_stages'][0]['post_nms_topn']
    flat = rpn_end['ids']
    n = len(flat)
    require(len(raw['rpn_records']) == len(raw['rpn_stages']) == 1 and len(flat.unique()) == len(proposals) == n and
            all(eq(proposals, p) for p in (record['final_boxes'], rpn_end['boxes'], raw['actual_rpn_return'],
                                           raw['post_proposals'][0])) and eq(record['final_scores'], rpn_end['scores']),
            'RPN candidates do not match actual RoI input')
    require(raw['actual_roi_input'][1] == raw['transform_sizes'] == raw['image_shapes'] == [[640, 640]],
            'The captured coordinate canvas differs')
    logits, regression = raw['class_logits'], raw['box_regression']
    probs, decoded, clipped = (raw[key]['result'] for key in ('softmax', 'decode', 'clip'))
    require(logits.shape == probs.shape == (n, 2) and regression.shape == (n, 8) and
            decoded.shape == clipped.shape == (n, 2, 4), 'Unexpected predictor/branch shapes')
    require(eq(logits, raw['predictor_output'][0]) and eq(regression, raw['predictor_output'][1]) and
            eq(regression, raw['decode']['args'][0]) and eq(proposals, raw['decode']['args'][1][0]) and
            eq(logits, raw['softmax']['args'][0]) and eq(decoded, raw['clip']['args'][0]),
            'Predictor, regression or class branch identity differs')
    require(all(bool(torch.isfinite(t).all()) for t in (proposals, logits, regression, probs, decoded, clipped)),
            'Nonfinite proposal or RoI tensors')
    person = clipped[:, 1]
    roi = expected_configuration['roi']
    score_rows = torch.where(probs[:, 1] > roi['score'])[0]
    require(eq(person[score_rows], raw['small']['args'][0]) and raw['small']['kwargs'] == {'min_size': .01},
            'Score filter differs from actual small-box input')
    nms_rows = score_rows[raw['small']['result']]
    boxes, scores, labels, threshold = raw['nms']['args']
    require(eq(person[nms_rows], boxes) and eq(probs[nms_rows, 1], scores) and
            eq(labels, torch.ones(len(nms_rows), dtype=torch.int64)) and threshold == roi['nms'],
            'Reconstructed candidates differ from actual class NMS input')
    nms_keep = raw['nms']['result']
    require(nms_keep.dtype == torch.int64 and len(nms_keep.unique()) == len(nms_keep) and
            bool(((nms_keep >= 0) & (nms_keep < len(boxes))).all()), 'Invalid actual NMS indices')
    keep = nms_keep[:roi['detections']]
    for tensor, post, key in zip((boxes, scores, labels), raw['actual_post_output'], ('boxes', 'scores', 'labels')):
        require(eq(tensor[keep], post[0]) and eq(post[0], raw['model_output'][key]),
                f'Budget slice does not match actual final {key}')
    return dict(rpn_count=n, nms_input_count=len(boxes), nms_keep_count=len(nms_keep), final_count=len(keep))


def restored_state(model, detector, weights):
    state = digest(model)
    require(state == weights['model_state_sha256'] and normalized(config(detector)) == weights['configuration'] and
            all(not m.training for m in model.modules()), 'Restored model state, configuration or eval state differs')
    return state


def new_capture(model, model_name, sample, dataset, folder, run_name, counts, weights):
    expected_paths = sample['files']
    i = sample['manifest_val_position']
    record = dataset.get_record(i)
    require(record.stem == sample['stem'] and record.visible_path == ROOT / expected_paths['visible']['path'] and
            record.infrared_path == ROOT / expected_paths['infrared']['path'] and
            record.annotation_path == ROOT / expected_paths['annotation_voc']['path'], 'Dataset record differs from frozen order')
    image, target = dataset[i]
    gt = torch.tensor(sample['gt_boxes_xyxy_640'], dtype=torch.float32).reshape(-1, 4)
    labels = torch.full((len(gt),), sample['label'], dtype=torch.int64)
    require(eq(target['boxes'], gt) and eq(target['labels'], labels), 'Preprocessed GT differs from frozen annotations')
    folder.mkdir(parents=True, exist_ok=False)
    data = capture(model, image, target, dict(letterbox_wh=[640, 640], final_output_wh=[640, 640],
                                              basis='frozen validation Letterbox'), sample['stem'], folder, counts,
                   model_name=model_name, max_forwards=PER_MODEL_NEW, run_name=run_name, gt_index=None)
    details = verify_chain(data, weights['configuration'])
    path = folder / 'raw_capture.pth'
    return dict(**details, path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                run_identity=data['run_identity'], gt_count=len(gt), gt_label_source='capture_and_xml',
                identity_chain_verified=True)


def capture_model(out, scope, model_name, counts, ledger, registered, completed_models):
    torch.manual_seed(42)
    weights = scope['model_weights'][model_name]
    checkpoint = ROOT / weights['path']
    model = restore_model(counts, model_name=model_name, checkpoint=checkpoint)
    detector = model.detector if model_name == 'Fusion' else model
    state = restored_state(model, detector, weights)
    dataset = ir.LLVIPDataset(DATA, split='val', modality='paired' if model_name == 'Fusion' else 'infrared',
                              infrared_channels=3, transforms=ir.PairedLetterboxAndFlip('val'))
    require(len(dataset) == len(scope['samples']), 'Live validation split length changed')
    samples = {sample['stem']: sample for sample in scope['samples']}
    for item in (i for i in scope['items'] if i['model'] == model_name):
        stem = item['stem']
        sample = samples[stem]
        row = dict(model=model_name, stem=stem, scene_id=sample['scene_id'], action=item['action'])
        try:
            if item['action'] == 'reuse_capture':
                entry = item['capture']
                require(entry['checkpoint_sha256'] == weights['sha256'] and entry['model_state_sha256'] == state and
                        entry['gt_count'] == len(sample['gt_boxes_xyxy_640']), 'Reuse model/GT identity differs')
                row.update(entry, identity_chain_verified=True)
            else:
                row.update(new_capture(model, model_name, sample, dataset, out / 'items' / model_name / stem,
                                       out.name, counts, weights))
            row['status'] = 'passed'
            registered.append(row)
        except Exception as exc:
            row.update(status='failed', error=str(exc))
            raise
        finally:
            ledger.write(json.dumps(row, ensure_ascii=False) + '\n')
            ledger.flush()
        n = counts['validation_full_forwards_returned']
        if item['action'] == 'new_forward' and (n == 1 or n % 25 == 0 or n == PER_MODEL_NEW):
            print(f'{model_name}: {n}/{PER_MODEL_NEW} new captures verified, latest {stem}', flush=True)
    final_state = restored_state(model, detector, weights)
    require(sha(checkpoint) == weights['sha256'], f'{model_name} checkpoint bytes changed')
    require(all(counts[key] == PER_MODEL_NEW for key in ('validation_full_forward_attempts',
                'validation_full_forwards_returned', 'model_forward_calls', 'original_roi_nms_calls',
                'original_rpn_nms_records')), f'{model_name} actual execution counts differ')
    save(out / f'{model_name.lower()}_model_checks.json', dict(state_before=state, state_after=final_state,
                                                             configuration=weights['configuration'], counts=counts))
    completed_models.append(model_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument('--preflight-only', action='store_true')
    route.add_argument('--output', type=Path)
    args = parser.parse_args()
    scope = preflight()
    if args.preflight_only:
        print('PASS: 300 frozen items, 4 capture hashes, 450 input files, 2 checkpoint hashes; no models restored')
        return 0
    out = args.output.resolve()
    require(out.parent == HERE / 'outputs', 'Choose a new folder directly under week11/day05/outputs')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    save(out / 'run_metadata.json', dict(command=subprocess.list2cmdline(sys.orig_argv), cwd=str(Path.cwd()),
                                         scope= SCOPE.relative_to(ROOT).as_posix(), scope_sha256=SCOPE_SHA,
                                         source_sha256=sha(Path(__file__)),
                                         capture_source_sha256=sha(ROOT / 'week11/day01/trace_fusion.py'),
                                         started_utc=dt.datetime.now(dt.UTC).isoformat(), environment=scope['environment'],
                                         threads=2, seed=42, output_policy='save one raw artifact per new item, reference old captures'))
    counts_by_model = {model: dict(checkpoint_loads=0, strict_state_loads=0, model_restore_attempts=0,
                                  structure_probes=0, validation_full_forward_attempts=0, validation_full_forwards_returned=0,
                                  model_forward_calls=0, original_roi_nms_calls=0, original_rpn_nms_records=0)
                       for model in MODELS}
    torch.set_num_threads(2)
    completed_models, entries = [], []
    failure = None
    try:
        with (out / 'item_ledger.jsonl').open('x', encoding='utf-8') as ledger:
            for model_name in MODELS:
                capture_model(out, scope, model_name, counts_by_model[model_name], ledger, entries, completed_models)
        require([(row['model'], row['stem']) for row in entries] ==
                [(item['model'], item['stem']) for item in scope['items']], 'Completed order differs from frozen scope')
        require(all(sum(row['gt_count'] for row in entries if row['model'] == model) == 416 for model in MODELS),
                'Completed GT count differs from frozen scope')
    except Exception as exc:
        failure = str(exc)
        (out / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    end_code = int(failure is not None)
    save(out / 'capture_index.json', entries)
    save(out / 'verification.json', dict(status='failed' if end_code else 'passed', error=failure, exit_code=end_code,
                                         completed_models=completed_models, elapsed_seconds=round(time.monotonic() - start, 3),
                                         items_verified=len(entries), reuse_count=sum(r['action'] == 'reuse_capture' for r in entries),
                                         counts_by_model=counts_by_model, independent_nms_replays=0, training=0, server_connections=0))
    print(f'{"FAILED" if end_code else "PASSED"}: {len(entries)} items in {out}', flush=True)
    return end_code


if __name__ == '__main__':
    sys.exit(main())
