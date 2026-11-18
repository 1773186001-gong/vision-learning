"""Trace exactly two Fusion validation images and verify each against Day03."""

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

import torch
import torchvision
from PIL import Image
from torchvision.models.detection import roi_heads as roi_module
from torchvision.ops import FrozenBatchNorm2d, box_iou

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in ('week10/day06', 'week08/day03', 'week08/day05', 'week09/day04'):
    sys.path.insert(0, str(ROOT / directory))

from trace_success import config, cp, csvout, digest, eq, load, normalized, read, require, save, sha
from agnostic_capture import AgnosticCapture
import fusion_model
import ir_common as ir

DAY03 = ROOT / 'week10/day03/outputs/rpn_final_comparison_run01'
IR_TABLE = ROOT / 'week10/day06/outputs/success_case_trace_run01/two_case_comparison.csv'
IR_FAILURE = ROOT / 'week10/day04/outputs/roi_trace_run01'
CHECKPOINT = ROOT / 'week08/day05/outputs/fusion_concat_run01/best.pth'
WEIGHT_SHA256 = '9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a'
STEMS = ('030078', '030129')
IOU_THRESHOLD = 0.75


def restore_model(counts, *, model_name='Fusion', checkpoint=CHECKPOINT):
    original_load = torch.load
    original_state_load = torch.nn.Module.load_state_dict
    expected_probes = int(model_name == 'Fusion')
    restore = {'Fusion': fusion_model.restore, 'IR': ir.checkpoint_model}[model_name]

    def counted_load(path, *args, **kwargs):
        require(Path(path).resolve() == checkpoint and counts['checkpoint_loads'] == 0,
                'Unexpected checkpoint load')
        counts['checkpoint_loads'] += 1
        return original_load(path, *args, **kwargs)

    def strict_load(module, state, *args, **kwargs):
        require(kwargs.get('strict') is True and counts['strict_state_loads'] == 0,
                'Expected one strict model load')
        counts['strict_state_loads'] += 1
        return original_state_load(module, state, *args, **kwargs)

    def probe(module, args):
        if type(module).__name__ == 'BackboneWithFPN':
            counts['structure_probes'] += 1
            require(counts['structure_probes'] == 1 and args[0].shape == (1, 3, 640, 640)
                    and not bool(args[0].any()), 'Unexpected restoration probe')

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch, 'load', counted_load))
        stack.enter_context(patch.object(torch.nn.Module, 'load_state_dict', strict_load))
        stack.enter_context(patch.object(torch.hub, 'download_url_to_file',
                                         side_effect=RuntimeError('Download forbidden')))
        stack.callback(torch.nn.modules.module.register_module_forward_pre_hook(probe).remove)
        counts['model_restore_attempts'] += 1
        model, _ = restore(checkpoint, 'cpu')
    require(counts['checkpoint_loads'] == counts['strict_state_loads'] == 1 and
            counts['structure_probes'] == expected_probes,
            'Restore counts differ from plan')
    model.eval()
    require(any(isinstance(m, FrozenBatchNorm2d) for m in model.modules()) and
            not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()), 'FrozenBN required')
    return model


def capture(model, image, target, coordinates, stem, folder, counts, *, model_name='Fusion',
            max_forwards=len(STEMS), run_name=None, gt_index=0):
    det = model.detector if model_name == 'Fusion' else model
    roi = det.roi_heads
    original_post = roi.postprocess_detections
    operations = [(roi.box_coder, 'decode', 'decode'), (roi_module.F, 'softmax', 'softmax'),
                  (roi_module.box_ops, 'clip_boxes_to_image', 'clip'),
                  (roi_module.box_ops, 'remove_small_boxes', 'small'),
                  (roi_module.box_ops, 'batched_nms', 'nms')]
    tracked = [(det.rpn, 'filter_proposals'), (det.rpn, '_get_top_n_idx'),
               (roi, 'postprocess_detections')] + [(obj, attr) for obj, attr, _ in operations]
    missing = object()
    attrs = [(obj, attr, vars(obj).get(attr, missing)) for obj, attr in tracked]
    hooks = [(module, dict(module._forward_pre_hooks), dict(module._forward_hooks))
             for module in model.modules()]
    run_id = f'{run_name or folder.parent.name}/{model_name}/{stem}'
    data = dict(model=model_name, stem=stem, run_identity=run_id, gt_index=gt_index,
                gt_all=cp(target['boxes']), gt_labels=cp(target['labels']), coordinates=coordinates,
                configuration=normalized(config(det)))
    cap = None

    def capture_op(obj, attr, key):
        original = getattr(obj, attr)

        def wrapped(*args, **kwargs):
            require(key not in data, f'Repeated RoI call: {key}')
            result = original(*args, **kwargs)
            data[key] = dict(args=cp(args), kwargs=cp(kwargs), result=cp(result))
            if key == 'nms':
                counts['original_roi_nms_calls'] += 1
            return result

        return wrapped

    def post(logits, regression, proposals, shapes):
        require('class_logits' not in data, 'Repeated RoI postprocess')
        data.update(class_logits=cp(logits), box_regression=cp(regression),
                    post_proposals=cp(proposals), image_shapes=cp(shapes))
        with ExitStack() as narrow:
            for obj, attr, key in operations:
                narrow.enter_context(patch.object(obj, attr, capture_op(obj, attr, key)))
            output = original_post(logits, regression, proposals, shapes)
        data['actual_post_output'] = cp(output)
        return output

    def store(key, value):
        require(key not in data, f'Repeated hook: {key}')
        data[key] = cp(value)

    def model_pre(_, __):
        counts['model_forward_calls'] += 1
        require(counts['model_forward_calls'] == counts['validation_full_forward_attempts'] <= max_forwards,
                'Validation forward budget exceeded')

    def rpn_pre(_, args):
        nonlocal cap
        images, features, _ = args
        require(cap is None and images.image_sizes == [(640, 640)] and tuple(features) == ('0', '1', 'pool'),
                'Unexpected RPN geometry, feature keys or call count')
        cap = stack.enter_context(AgnosticCapture(det.rpn, tuple(features), target['boxes'], run_id))

    try:
        with ExitStack() as stack:
            for module, before in [(model, model_pre), (det.rpn, rpn_pre),
                                   (roi, lambda _, args: store('actual_roi_input', args[1:3]))]:
                stack.callback(module.register_forward_pre_hook(before).remove)
            for module, callback in [
                (det.rpn, lambda m, a, out: store('actual_rpn_return', out[0][0])),
                (det.transform, lambda m, a, out: store('transform_sizes', out[0].image_sizes)),
                (roi.box_predictor, lambda m, a, out: store('predictor_output', out)),
            ]:
                stack.callback(module.register_forward_hook(callback).remove)
            stack.enter_context(patch.object(roi, 'postprocess_detections', post))
            counts['validation_full_forward_attempts'] += 1
            with torch.inference_mode():
                data['model_output'] = cp(model([image])[0])
            counts['validation_full_forwards_returned'] += 1
    finally:
        if cap is not None:
            data.update(rpn_records=cp(cap.records), rpn_stages=cp(cap.states), catalogues=cp(cap.catalogues))
            counts['original_rpn_nms_records'] += len(cap.records)
        data['hooks_restored'] = all(dict(m._forward_pre_hooks) == pre and dict(m._forward_hooks) == post_hooks
                                     for m, pre, post_hooks in hooks)
        data['temporary_attributes_restored'] = all(vars(obj).get(attr, missing) is value
                                                    for obj, attr, value in attrs)
        with (folder / 'raw_capture.pth').open('xb') as handle:
            torch.save(data, handle)
    return data


def compare_reference(folder, data, old, model, state, reference_state, config_before):
    new_rpn, old_rpn = data['rpn_records'][0], old['records'][0]
    new_ids, old_ids = [item[0]['post_nms_topn']['ids'].tolist()
                        for item in (data['rpn_stages'], old['stages'])]
    new_rows = {flat: i for i, flat in enumerate(new_ids)}
    old_rows = {flat: i for i, flat in enumerate(old_ids)}
    rpn_diff = []
    for flat in sorted(new_rows.keys() | old_rows.keys()):
        i, j = new_rows.get(flat), old_rows.get(flat)
        b = new_rpn['final_boxes'][i] if i is not None else None
        c = old_rpn['final_boxes'][j] if j is not None else None
        s = new_rpn['final_scores'][i] if i is not None else None
        t = old_rpn['final_scores'][j] if j is not None else None
        rpn_diff.append(dict(flat=flat, current_row=i, day03_row=j,
                             boxes_exact=b is not None and c is not None and eq(b, c),
                             objectness_exact=s is not None and t is not None and eq(s, t),
                             current_box=b.tolist() if b is not None else None,
                             day03_box=c.tolist() if c is not None else None,
                             current_objectness=s.item() if s is not None else None,
                             day03_objectness=t.item() if t is not None else None))
    csvout(folder / 'day03_rpn_comparison.csv', rpn_diff)
    prediction, reference = data['model_output'], old['model_output']
    fields = ('boxes', 'scores', 'labels')
    final_diff = []
    for i in range(max(len(prediction['boxes']), len(reference['boxes']))):
        row = {'output_index': i}
        for key in fields:
            a = prediction[key][i] if i < len(prediction[key]) else None
            b = reference[key][i] if i < len(reference[key]) else None
            row.update({f'{key}_exact': a is not None and b is not None and eq(a, b),
                        f'current_{key}': a.tolist() if a is not None else None,
                        f'day03_{key}': b.tolist() if b is not None else None})
        final_diff.append(row)
    csvout(folder / 'day03_final_comparison.csv', final_diff)
    state_after = digest(model)
    flags = dict(rpn_flat_sequence_exact=new_ids == old_ids,
                 rpn_flats_unique=len(new_rows) == len(new_ids) and len(old_rows) == len(old_ids),
                 rpn_boxes_and_objectness_by_flat=all(r['boxes_exact'] and r['objectness_exact'] for r in rpn_diff),
                 actual_rpn_return_exact=eq(data['actual_rpn_return'], old['observed']['actual_rpn_return']),
                 ordered_final_output_exact=all(eq(prediction[k], reference[k]) for k in fields),
                 gt_exact=eq(data['gt_all'], old['gt_all']) and eq(data['gt_labels'], old['labels']),
                 coordinates_exact=data['coordinates'] == old['coordinates'] and
                 data['transform_sizes'] == old['observed']['transform_sizes'],
                 configuration_exact=data['configuration'] == normalized(old['configuration']) == config_before ==
                 normalized(config(model.detector)),
                 checkpoint_sha_exact=sha(CHECKPOINT) == WEIGHT_SHA256 == reference_state['checkpoint_sha_after'],
                 model_state_exact=state == state_after == reference_state['state_before'] == reference_state['state_after'],
                 all_modules_eval=all(not m.training for m in model.modules()),
                 hooks_restored=data['hooks_restored'], temporary_attributes_restored=data['temporary_attributes_restored'])
    result = dict(exact_reproduced=all(flags.values()), checks=flags, state_before=state, state_after=state_after,
                  day03_state=reference_state['state_before'], ordered_flat_ids=new_ids,
                  day03_ordered_flat_ids=old_ids, current_coordinates=data['coordinates'],
                  day03_coordinates=old['coordinates'], current_config=data['configuration'],
                  day03_config=normalized(old['configuration']), current_gt=data['gt_all'].tolist(),
                  day03_gt=old['gt_all'].tolist(), tolerance_used=False)
    save(folder / 'day03_reproduction.json', result)
    require(result['exact_reproduced'], f'Day03 reproduction failed on {data["stem"]}: '
            f'{[name for name, passed in flags.items() if not passed]}')


def analyze(data, folder):
    proposal = data['actual_roi_input'][0][0]
    rpn = data['rpn_records'][0]
    flats = data['rpn_stages'][0]['post_nms_topn']['ids'].tolist()
    n = len(flats)
    require(all(eq(proposal, source) for source in (rpn['final_boxes'], data['actual_rpn_return'],
                                                   data['post_proposals'][0])), 'RPN-to-RoI row identity differs')
    require(data['actual_roi_input'][1] == data['transform_sizes'] == data['image_shapes'] == [[640, 640]],
            'RoI coordinate canvas differs')
    logits, regression = data['class_logits'], data['box_regression']
    probs, decoded, clipped = (data[key]['result'] for key in ('softmax', 'decode', 'clip'))
    require(logits.shape == probs.shape == (n, 2) and regression.shape == (n, 8) and
            decoded.shape == clipped.shape == (n, 2, 4), 'Unexpected class branch shapes')
    require(eq(logits, data['predictor_output'][0]) and eq(regression, data['predictor_output'][1]) and
            eq(regression, data['decode']['args'][0]) and eq(proposal, data['decode']['args'][1][0]) and
            eq(logits, data['softmax']['args'][0]) and eq(decoded, data['clip']['args'][0]), 'Predictor identity differs')

    boxes, scores = clipped[:, 1:].reshape(-1, 4), probs[:, 1:].reshape(-1)
    labels = torch.ones(n, dtype=torch.int64)
    rows = torch.arange(n)
    roi = data['configuration']['roi']
    score_rows = torch.where(scores > roi['score'])[0]
    small = data['small']
    require(eq(boxes[score_rows], small['args'][0]) and small['kwargs'] == {'min_size': .01},
            'Score filter reconstruction differs from actual small-box input')
    nms_rows = score_rows[small['result']]
    nb, ns, nl, threshold = data['nms']['args']
    require(eq(boxes[nms_rows], nb) and eq(scores[nms_rows], ns) and eq(labels[nms_rows], nl) and
            threshold == roi['nms'], 'Small-box indices differ from actual NMS input')
    kept_rows = nms_rows[data['nms']['result']]
    final_rows = kept_rows[:roi['detections']]
    prediction = data['model_output']
    for tensor, post, key in zip((boxes, scores, labels), data['actual_post_output'], ('boxes', 'scores', 'labels')):
        require(eq(tensor[final_rows], post[0]) and eq(post[0], prediction[key]), f'Final {key} identity differs')

    gt = data['gt_all'][0:1]
    stages = [
        ('rpn_proposals', proposal, rows, 'actual RPN return and RoI input'),
        ('person_decoded', decoded[:, 1], rows, 'actual decode'),
        ('person_clipped', clipped[:, 1], rows, 'actual clip'),
        ('background_removed_expanded', boxes, rows, 'reconstructed and checked at small-box input'),
        ('score_filtered', boxes[score_rows], score_rows, 'strict score indices checked at actual small-box input'),
        ('small_filtered', nb, nms_rows, 'actual small-box indices and NMS input'),
        ('nms_all_output', boxes[kept_rows], kept_rows, 'actual full NMS indices'),
        ('detections_budget', data['actual_post_output'][0][0], final_rows, 'budget slice checked at actual postprocess'),
        ('model_output', prediction['boxes'], final_rows, 'actual model output'),
    ]
    candidates, coverage = {}, []
    for stage, stage_boxes, stage_rows, evidence in stages:
        overlaps = box_iou(stage_boxes, gt)[:, 0].tolist()
        stage_records = []
        for position, (row, iou) in enumerate(zip(stage_rows.tolist(), overlaps)):
            flat = flats[row]
            stage_records.append(dict(model='Fusion', stem=data['stem'], stage=stage, stage_index=position,
                                      roi_row=row, flat=flat, branch_id=f'{data["run_identity"]}/{flat}/class1',
                                      xyxy=stage_boxes[position].tolist(), gt_iou=iou, covered=iou >= IOU_THRESHOLD,
                                      person_score=float(scores[row]), rpn_objectness=float(rpn['final_scores'][row])))
        candidates[stage] = stage_records
        best = max(overlaps, default=None)
        good = [record['flat'] for record in stage_records if record['covered']]
        coverage.append(dict(model='Fusion', stem=data['stem'], stage=stage, count=len(stage_rows),
                             max_gt_iou=best, qualifying_count=len(good), qualifying_flats=good,
                             maximum_iou_flats=[r['flat'] for r in stage_records if r['gt_iou'] == best], evidence=evidence))

    focus = {flat: ['达标'] for entry in coverage for flat in entry['qualifying_flats']}
    for stage, key, label in [('rpn_proposals', 'gt_iou', 'RPN最高IoU'),
                               ('small_filtered', 'gt_iou', 'NMS前最高IoU'),
                               ('small_filtered', 'person_score', 'NMS前最高分类分数'),
                               ('model_output', 'gt_iou', '最终最高IoU'),
                               ('model_output', 'person_score', '最终最高分类分数')]:
        best = max((row[key] for row in candidates[stage]), default=None)
        for row in candidates[stage]:
            if row[key] == best:
                focus.setdefault(row['flat'], []).append(label)
    traces = []
    for flat, roles in sorted(focus.items()):
        for stage, records in candidates.items():
            row = next((r for r in records if r['flat'] == flat), None)
            traces.append(dict(model='Fusion', stem=data['stem'], flat=flat, roi_row=flats.index(flat),
                               branch_id=f'{data["run_identity"]}/{flat}/class1', roles=roles,
                               stage=stage, present=row is not None, stage_index=row['stage_index'] if row else None,
                               gt_iou=row['gt_iou'] if row else None, person_score=float(scores[flats.index(flat)])))
    csvout(folder / 'stage_candidates.csv', [row for records in candidates.values() for row in records])
    csvout(folder / 'stage_coverage.csv', coverage)
    csvout(folder / 'focus_traces.csv', traces)
    csvout(folder / 'final_sources.csv', candidates['model_output'])
    save(folder / 'verification.json', dict(status='passed', exact_reproduced=True, identity_chain_verified=True))
    return coverage, candidates


def prepare(out):
    manifest = load(DAY03 / 'artifact_manifest.json')
    source_hashes = {Path(entry['path']): entry['sha256'] for entry in load(DAY03 / 'input_provenance.json')}
    provenance = {}

    def check(path, expected):
        actual = sha(path)
        require(actual == expected, f'Input SHA mismatch: {path}')
        provenance[str(path)] = actual

    check(CHECKPOINT, WEIGHT_SHA256)
    for name in ('artifact_manifest.json',):
        path = DAY03 / name
        provenance[str(path)] = sha(path)
    for name in ('input_provenance.json', 'model_checks.json', 'run_metadata.json', 'verification.json',
                 *(f'Fusion_{stem}_raw_capture.pth' for stem in STEMS)):
        check(DAY03 / name, manifest[name])
    require(load(DAY03 / 'verification.json')['status'] == 'passed', 'Day03 verification failed')
    historical_env = load(DAY03 / 'run_metadata.json')['environment']
    actual_env = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__)
    require(all(historical_env[k] == v for k, v in actual_env.items()), 'Day03 Python/PyTorch/torchvision mismatch')

    for relative in ('week08/day01/llvip_dataset.py', 'week08/day02/paired_letterbox.py',
                     'week08/day02/paired_horizontal_flip.py', 'week08/day03/ir_common.py',
                     'week08/day05/fusion_model.py', 'week09/day04/agnostic_capture.py', 'data/LLVIP_1000/manifest.csv'):
        path = ROOT / relative
        check(path, source_hashes[path])
    roi_source = Path(inspect.getfile(roi_module))
    check(roi_source, load(IR_FAILURE / 'model_checks.json')['roi_source_sha'])
    check(IR_TABLE, load(IR_TABLE.parent / 'artifact_manifest.json')[IR_TABLE.name])
    ir_results = {row['stem']: row for row in read(IR_TABLE)}
    require(set(ir_results) == set(STEMS) and all(row['model'] == 'IR' for row in ir_results.values()),
            'Unexpected saved IR comparison scope')

    dataset = ir.LLVIPDataset(ROOT / 'data/LLVIP_1000', split='val', modality='paired',
                              infrared_channels=3, transforms=ir.PairedLetterboxAndFlip('val'))
    samples = {}
    for stem in STEMS:
        matching = [i for i, rec in enumerate(dataset.records) if rec.stem == stem]
        require(len(matching) == 1, f'Expected exactly one validation image for {stem}')
        index = matching[0]
        record = dataset.get_record(index)
        for path in (record.visible_path, record.infrared_path, record.annotation_path):
            check(path, source_hashes[path])
        with Image.open(record.infrared_path) as infrared, Image.open(record.visible_path) as visible:
            w, h = infrared.size
            require(infrared.size == visible.size, 'Paired original image geometry differs')
        params = dataset.transforms.letterbox.parameters(h, w)
        coords = dict(original_wh=[w, h], letterbox_wh=[640, 640], scale=params.scale,
                      padding_ltrb=list(params.padding_ltrb), internal_wh=[640, 640], final_output_wh=[640, 640],
                      final_output='model returns coordinates of supplied Letterbox input; no unpadding')
        image, target = dataset[index]
        require(len(target['boxes']) > 0 and bool((target['labels'] == 1).all()), f'{stem} missing person GT')
        samples[stem] = image, target, coords
    for path in (Path(__file__), ROOT / 'week10/day06/trace_success.py'):
        provenance[str(path)] = sha(path)
    save(out / 'input_provenance.json', provenance)
    return samples, ir_results


def ir_summary(row):
    return dict(model='IR', stem=row['stem'], rpn_max_iou=float(row['rpn_max_iou']),
                rpn_good_count=len(json.loads(row['rpn_good_identities'])),
                decoded_max_iou=float(row['decoded_max_iou']), decoded_good_count=int(row['decoded_good_count']),
                before_nms_good_count=int(row['nms_before_good_count']),
                after_nms_good_count=int(row['nms_after_good_count']),
                final_good_count=len(json.loads(row['final_good_identities'])),
                final_max_iou=float(row['final_best_iou']),
                final_best_identity=row['final_best_iou_identity'],
                nms_top_score_identity=row['nms_top_score_identity'],
                nms_top_score=float(row['nms_top_classification_score']),
                nms_top_score_iou=float(row['nms_top_gt_iou']))


def fusion_summary(coverage, candidates):
    by_stage = {row['stage']: row for row in coverage}
    rpn, decoded, pre, post, final = [by_stage[k] for k in ('rpn_proposals', 'person_decoded', 'small_filtered',
                                                          'nms_all_output', 'model_output')]
    best_final = max(candidates['model_output'], key=lambda row: row['gt_iou'], default={})
    top_score = max(candidates['small_filtered'], key=lambda row: row['person_score'], default={})
    return dict(model='Fusion', stem=rpn['stem'], rpn_max_iou=rpn['max_gt_iou'],
                rpn_good_count=rpn['qualifying_count'], decoded_max_iou=decoded['max_gt_iou'],
                decoded_good_count=decoded['qualifying_count'], before_nms_good_count=pre['qualifying_count'],
                after_nms_good_count=post['qualifying_count'], final_good_count=final['qualifying_count'],
                final_max_iou=final['max_gt_iou'], final_best_identity=best_final.get('branch_id'),
                nms_top_score_identity=top_score.get('branch_id'), nms_top_score=top_score.get('person_score'),
                nms_top_score_iou=top_score.get('gt_iou'))


def write_results(out, comparisons, stages):
    csvout(out / 'ir_fusion_comparison.csv', comparisons)
    csvout(out / 'stage_coverage.csv', stages)
    report = ['# Week11 Day01 · Fusion RoI 两例追踪', '',
              '两例的 RPN 有序候选与完整最终输出均精确复现 Week10 Day03 CPU 结果，RoI 身份链逐段核验。', '',
              '| 图片 | 模型 | RPN 达标 | 解码达标 | NMS 前达标 | NMS 后达标 | 最终达标 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in comparisons:
        report.append('| {stem} | {model} | {rpn_good_count} | {decoded_good_count} | {before_nms_good_count} | '
                      '{after_nms_good_count} | {final_good_count} |'.format(**row))
    report += ['', '达标指对 G0 的 IoU ≥ 0.75。']
    stage_names = dict(rpn_proposals='RPN', person_decoded='回归解码', person_clipped='裁剪',
                       background_removed_expanded='去背景展开', score_filtered='分类分数筛选后',
                       small_filtered='小框筛选后（NMS 前）', nms_all_output='类别 NMS 后',
                       detections_budget='100 框预算后', model_output='最终输出')
    for stem in STEMS:
        records = [row for row in stages if row['stem'] == stem]
        reached = [row for row in records if row['qualifying_count']]
        decoded, clipped = records[1:3]
        last = (f'最后一次达标出现在**{stage_names[reached[-1]["stage"]]}**，'
                f'候选 flat 为 {reached[-1]["qualifying_flats"]}。' if reached else '整个 RoI 流程均没有达标框。')
        report.extend(['', f'`{stem}/G0` 的 Fusion 回归解码后达标 {decoded["qualifying_count"]} 个，'
                       f'最大 IoU {decoded["max_gt_iou"]:.6f}。裁剪后达标 {clipped["qualifying_count"]} 个。{last}'])
    (out / 'case_report_zh.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output.resolve()
    require(output.is_relative_to(HERE / 'outputs') and output.parent == HERE / 'outputs',
            'Use a new run folder directly under week11/day01/outputs')
    output.mkdir(parents=True, exist_ok=False)
    counts = dict(checkpoint_loads=0, strict_state_loads=0, model_restore_attempts=0, structure_probes=0,
                  validation_full_forward_attempts=0, validation_full_forwards_returned=0, model_forward_calls=0,
                  original_roi_nms_calls=0, original_rpn_nms_records=0, reference_capture_loads=0,
                  ir_forwards=0, independent_nms_replays=0, training=0, server_connections=0, test_inference=0)
    completed = []
    phase = 'preflight'
    failure = None
    with (output / 'run.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            samples, ir_results = prepare(output)
            torch.set_num_threads(2)
            torch.manual_seed(42)
            phase = 'Fusion restore'
            model = restore_model(counts)
            configuration = normalized(config(model.detector))
            rpn, roi = configuration['rpn'], configuration['roi']
            require((rpn['pre'], rpn['post'], rpn['score'], rpn['nms']) == (150, 150, .05, .7) and
                    (roi['score'], roi['nms'], roi['detections']) == (.001, .5, 100), 'Model thresholds differ')
            states = {row['stem']: row for row in load(DAY03 / 'model_checks.json') if row['model'] == 'Fusion'}
            comparisons, all_stages = [], []
            initial_state = digest(model)
            for stem in STEMS:
                phase = f'{stem} reference checks'
                folder = output / stem
                folder.mkdir()
                counts['reference_capture_loads'] += 1
                reference = torch.load(DAY03 / f'Fusion_{stem}_raw_capture.pth', map_location='cpu', weights_only=True)
                reference_state = states[stem]
                image, target, coords = samples[stem]
                state = digest(model)
                require(reference['model'] == 'Fusion' and reference['stem'] == stem and reference['gt_index'] == 0,
                        'Reference capture identity differs')
                checks = dict(state_exact=state == initial_state == reference_state['state_before'],
                              configuration_exact=configuration == normalized(config(model.detector)) ==
                              normalized(reference['configuration']) == reference_state['config'],
                              gt_exact=eq(target['boxes'], reference['gt_all']) and
                              eq(target['labels'], reference['labels']), coordinates_exact=coords == reference['coordinates'])
                save(folder / 'pre_forward_checks.json', dict(checks=checks, model_state=state,
                                                                day03_model_state=reference_state['state_before'],
                                                                configuration=configuration))
                require(all(checks.values()), f'Pre-forward checks failed: {[k for k, v in checks.items() if not v]}')
                phase = f'{stem} capture'
                print(f'Capture {stem}', flush=True)
                data = capture(model, image, target, coords, stem, folder, counts)
                phase = f'{stem} Day03 exact comparison'
                compare_reference(folder, data, reference, model, state, reference_state, configuration)
                phase = f'{stem} RoI identity verification'
                coverage, candidates = analyze(data, folder)
                comparisons.extend((ir_summary(ir_results[stem]), fusion_summary(coverage, candidates)))
                all_stages.extend(coverage)
                completed.append(stem)
                print(f'Passed {stem}: {len(candidates["rpn_proposals"])} RPN and '
                      f'{len(candidates["model_output"])} final boxes', flush=True)
            phase = 'aggregate results'
            require(all(counts[key] == 2 for key in ('validation_full_forward_attempts', 'model_forward_calls',
                                                     'validation_full_forwards_returned', 'original_roi_nms_calls',
                                                     'original_rpn_nms_records')), 'Unexpected forward or NMS count')
            write_results(output, comparisons, all_stages)
        except Exception as error:
            failure = str(error)
            traceback.print_exc()
    exit_code = int(failure is not None)
    save(output / 'verification.json', dict(status='passed' if not exit_code else 'failed', completed=completed,
                                            failed_phase=phase if exit_code else None, error=failure,
                                            counts=counts, iou_threshold=IOU_THRESHOLD, exit_code=exit_code))
    save(output / 'run_metadata.json', dict(command=subprocess.list2cmdline(sys.orig_argv), cwd=str(Path.cwd()),
                                            environment=dict(python=sys.version, torch=torch.__version__,
                                                             torchvision=torchvision.__version__, device='cpu',
                                                             threads=torch.get_num_threads(), seed=42),
                                            source_sha256=sha(Path(__file__)), counts=counts, exit_code=exit_code))
    save(output / 'artifact_manifest.json', {p.relative_to(output).as_posix(): sha(p)
                                            for p in sorted(output.rglob('*')) if p.is_file()})
    print(f'{"FAILED" if exit_code else "PASSED"}: {output}')
    if failure is not None:
        print(f'{phase}: {failure}', file=sys.stderr)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
