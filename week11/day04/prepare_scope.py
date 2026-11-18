"""Freeze Week11 full-validation scope and verify reusable captures without inference."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / p) for p in ('week08/day01', 'week08/day02')]
from llvip_dataset import _parse_voc_annotation
from paired_letterbox import PairedLetterbox

DATA = ROOT / 'data/LLVIP_1000'
W09 = ROOT / 'week09/day04/outputs/full_val_stage_run01'
W10 = ROOT / 'week10/day03/outputs/rpn_final_comparison_run01'
FUSION = ROOT / 'week11/day01/outputs/fusion_roi_trace_run02'
IR_078 = ROOT / 'week10/day06/outputs/success_case_trace_run01'
IR_129 = ROOT / 'week10/day04/outputs/roi_trace_run01'
CAPTURES = {('IR', '030078'): (IR_078, ''), ('IR', '030129'): (IR_129, ''),
            ('Fusion', '030078'): (FUSION, '030078'), ('Fusion', '030129'): (FUSION, '030129')}
WEIGHTS = {'IR': ROOT / 'week08/day03/outputs/ir_baseline_run01/best.pth',
           'Fusion': ROOT / 'week08/day05/outputs/fusion_concat_run01/best.pth'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def checked_file(path, expected):
    require(sha(path) == expected, f'SHA256 differs for {path}')
    return expected


def verified_artifact(base, name):
    path = base / name
    checked_file(path, read_json(base / 'artifact_manifest.json')[name])
    return path


def freeze_samples(historical):
    hashes = read_json(W09 / 'preflight/expected_inputs.json')['sha256']
    source_manifest = DATA / 'manifest.csv'
    manifest_sha = checked_file(source_manifest, hashes[relative(source_manifest)])
    letterbox = PairedLetterbox(640)
    samples = {}
    for index, row in enumerate(r for r in read_csv(source_manifest) if r['split'] == 'val'):
        stem = row['stem']
        require(stem not in samples, f'Duplicate validation stem {stem}')
        files = {}
        for key in ('infrared', 'visible', 'annotation_voc'):
            path = relative(DATA / row[key])
            files[key] = dict(path=path, sha256=checked_file(ROOT / path, hashes[path]))
        target, (width, height) = _parse_voc_annotation(ROOT / files['annotation_voc']['path'], index, stem)
        boxes = target['boxes']
        require(len(boxes) == int(row['person_boxes']), f'Manifest/XML GT count differs for {stem}')
        params = letterbox.parameters(height, width)
        left, top, _, _ = params.padding_ltrb
        # Same validation Letterbox coordinate operation, on annotations only.
        boxes *= params.scale
        boxes += boxes.new_tensor([left, top, left, top])
        letterbox.validate_boxes(boxes, 640, 640)
        samples[stem] = dict(stem=stem, scene_id=row['scene_id'], manifest_val_position=index,
                             files=files, gt_boxes_xyxy_640=boxes.tolist(), label=1)
    gt_keys = {(s['stem'], i, s['scene_id']) for s in samples.values() for i in range(len(s['gt_boxes_xyxy_640']))}
    old_gt = [r for r in read_csv(verified_artifact(W09, 'gt_coverage_sequences.csv'))
              if r['iou_convention'] == 'float32_cpu']
    for model in WEIGHTS:
        keys = [(r['stem'], int(r['gt_index']), r['scene_id']) for r in old_gt if r['model'] == model]
        require(set(keys) == gt_keys and len(keys) == len(gt_keys) == historical['actual'][model]['valid_gt'] and
                len(samples) == historical['actual'][model]['images'], f'Historical sample/GT scope differs for {model}')
    return samples, manifest_sha, len(gt_keys)


def check_capture(model, stem, sample, reference):
    base, subdir = CAPTURES[model, stem]
    status = read_json(verified_artifact(base, f'{subdir}/verification.json'.lstrip('/')))
    reproduced = status.get('exact_reproduced', status.get('day03_exact_reproduced'))
    require(status['status'] == 'passed' and reproduced and status['identity_chain_verified'], 'Prior capture failed')
    source = verified_artifact(base, f'{subdir}/raw_capture.pth'.lstrip('/'))
    if model == 'Fusion':
        report = read_json(verified_artifact(base, f'{subdir}/day03_reproduction.json'))
        require(all(report['checks'].values()), 'Fusion reproduction flags differ')
        state_before, state_after = report['state_before'], report['state_after']
    else:
        report = read_json(verified_artifact(base, 'model_checks.json'))
        state_before, state_after = report['state_before'], report['state_after']
        check_sha = report.get('checkpoint_sha', report.get('checkpoint_sha_after'))
        require(check_sha == reference['checkpoint_sha_after'], 'IR captured checkpoint differs')
    require(state_before == state_after == reference['state_before'] == reference['state_after'], 'Model state differs')
    raw = torch.load(source, map_location='cpu', weights_only=True)
    require(raw['run_identity'].endswith(f'/{model}/{stem}'), 'Capture identity differs')
    configuration = json.loads(json.dumps(raw['configuration']))
    require(configuration == reference['config'], 'Captured configuration differs')
    roi, rpn = configuration['roi'], configuration['rpn']
    require((roi['score'], roi['nms'], roi['detections']) == (.001, .5, 100) and
            (rpn['pre'], rpn['post'], rpn['score'], rpn['nms']) == (150, 150, .05, .7), 'Unexpected fixed thresholds')
    gt = torch.tensor(sample['gt_boxes_xyxy_640'], dtype=torch.float32).reshape(-1, 4)
    gt_labels = torch.full((len(gt),), sample['label'], dtype=torch.int64)
    require(torch.equal(raw['gt_all'], gt) and torch.equal(raw.get('gt_labels', gt_labels), gt_labels),
            'Captured full GT differs from frozen annotation')
    require(all([tuple(size) for size in raw[key]] == [(640, 640)] for key in ('transform_sizes', 'image_shapes')),
            'Capture coordinate canvas differs')

    roi_input = raw['actual_roi_input'][0][0] if model == 'Fusion' else raw['actual_roi_input'][0]
    flat = raw['rpn_stages'][0]['post_nms_topn']['ids']
    require(len(flat.unique()) == len(flat) == len(roi_input) and
            torch.equal(roi_input, raw['actual_rpn_return']) and torch.equal(roi_input, raw['post_proposals'][0]),
            'RPN to RoI identity differs')
    probs, decoded, clipped = (raw[key]['result'] for key in ('softmax', 'decode', 'clip'))
    require(probs.shape == (len(flat), 2) and decoded.shape == clipped.shape == (len(flat), 2, 4) and
            torch.equal(raw['class_logits'], raw['predictor_output'][0]) and
            torch.equal(raw['box_regression'], raw['predictor_output'][1]), 'Missing predictor or person branches')
    person = clipped[:, 1]
    score_rows = torch.where(probs[:, 1] > roi['score'])[0]
    nms_rows = score_rows[raw['small']['result']]
    boxes, scores, labels, threshold = raw['nms']['args']
    require(torch.equal(person[score_rows], raw['small']['args'][0]) and torch.equal(person[nms_rows], boxes) and
            torch.equal(probs[nms_rows, 1], scores) and bool((labels == 1).all()) and threshold == roi['nms'],
            'Score/small-box/NMS input identity differs')
    keep = raw['nms']['result'][:roi['detections']]
    for key, tensor, post in zip(('boxes', 'scores', 'labels'), (boxes, scores, labels), raw['actual_post_output']):
        require(torch.equal(tensor[keep], post[0]) and torch.equal(post[0], raw['model_output'][key]),
                'Saved NMS/budget/final output differs')
    return dict(path=relative(source), sha256=sha(source), run_identity=raw['run_identity'],
                original_forward_reused=True, gt_count=len(gt), nms_input_count=len(boxes), final_count=len(keep),
                gt_label_source='capture_and_xml' if 'gt_labels' in raw else 'frozen_person_xml',
                checkpoint_sha256=reference['checkpoint_sha_after'], model_state_sha256=state_before)


def prepare():
    historical = read_json(verified_artifact(W09, 'verification.json'))
    require(historical['status'] == 'passed', 'Week09 full validation did not pass')
    samples, manifest_sha, gt_count = freeze_samples(historical)
    environment = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__)
    day03_env = read_json(verified_artifact(W10, 'run_metadata.json'))['environment']
    require(all(day03_env[key] == value for key, value in environment.items()), 'Current CPU environment differs from Day03')
    references = {(r['model'], r['stem']): r for r in read_json(verified_artifact(W10, 'model_checks.json'))}
    weights = {}
    for model, path in WEIGHTS.items():
        a, b = (references[model, stem] for stem in ('030078', '030129'))
        require(all(a[key] == b[key] for key in ('config', 'checkpoint_sha_after', 'state_before')),
                f'Day03 model references disagree for {model}')
        weights[model] = dict(path=relative(path), sha256=checked_file(path, a['checkpoint_sha_after']),
                              configuration=a['config'], model_state_sha256=a['state_before'])
    reusable, gaps = {}, []
    for model, stem in CAPTURES:
        try:
            reusable[model, stem] = check_capture(model, stem, samples[stem], references[model, stem])
        except (ValueError, KeyError, IndexError, OSError, RuntimeError) as error:
            gaps.append(dict(model=model, stem=stem, reason=str(error)))
    gap_keys = {(g['model'], g['stem']) for g in gaps}
    items = []
    for model in WEIGHTS:
        for stem in samples:
            key = (model, stem)
            if key in reusable:
                items.append(dict(model=model, stem=stem, action='reuse_capture', capture=reusable[key]))
            else:
                items.append(dict(model=model, stem=stem, action='blocked_reuse' if key in gap_keys else 'new_forward'))
    per_model = {model: dict(Counter(i['action'] for i in items if i['model'] == model)) for model in WEIGHTS}
    return dict(status='ready_for_review' if not gaps else 'reuse_gaps', historical_scope_source=relative(W09 / 'verification.json'),
                source_manifest=dict(path=relative(DATA / 'manifest.csv'), sha256=manifest_sha),
                scope=dict(split='val', images=len(samples), gt_per_model=gt_count, model_image_items=len(items),
                           image_count_by_scene=dict(sorted(Counter(s['scene_id'] for s in samples.values()).items())),
                           gt_count_by_scene=dict(sorted(Counter(s['scene_id'] for s in samples.values()
                                                                for _ in s['gt_boxes_xyxy_640']).items()))),
                environment=dict(device='cpu', **environment),
                model_weights=weights, samples=list(samples.values()), items=items, compatibility_gaps=gaps,
                proposed_run=dict(by_model=per_model, new_forwards=sum(i['action'] == 'new_forward' for i in items),
                                  reusable_captures=len(reusable), checkpoint_loads=2, fusion_structure_probes=1),
                preparation_counts=dict(validation_forwards=0, checkpoint_loads=0, independent_nms_replays=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    require(out.parent == HERE / 'outputs', 'Choose a new folder directly under week11/day04/outputs')
    manifest = prepare()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / 'scope.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'scope.sha256').write_text(f'{sha(manifest_path)}  scope.json\n', encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ('status', 'scope', 'proposed_run', 'compatibility_gaps')},
                     ensure_ascii=False, indent=2))
    return int(bool(manifest['compatibility_gaps']))


if __name__ == '__main__':
    sys.exit(main())
