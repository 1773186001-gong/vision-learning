"""Validation-only RPN proposal recall diagnostic; never trains or changes RPN budgets."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import inspect
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'week08/day03'))
sys.path.insert(0, str(ROOT / 'week08/day05'))

import torch
import torchvision
from torchvision.ops import box_iou
import ir_common as ir
import fusion_common as fusion

TOP_K = (100, 300, 1000)
THRESHOLDS = (0.50, 0.75)


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def tensor_digest(module):
    """Hash model parameters and persistent buffers without storing a second model."""
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        h.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def check_boxes(boxes, image_size, label):
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f'{label}: expected Nx4 boxes')
    if not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError(f'{label}: nonfinite boxes')
    if len(boxes):
        h, w = image_size
        if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError(f'{label}: degenerate boxes')
        if bool((boxes < -1e-4).any()) or bool((boxes[:, [0, 2]] > w + 1e-4).any()) or bool((boxes[:, [1, 3]] > h + 1e-4).any()):
            raise ValueError(f'{label}: coordinates outside transformed image {image_size}')


def rpn_configuration(detector):
    r = detector.rpn
    t = detector.transform
    return {
        'pre_nms_top_n': dict(r._pre_nms_top_n),
        'post_nms_top_n': dict(r._post_nms_top_n),
        'active_pre_nms_top_n': r.pre_nms_top_n(),
        'active_post_nms_top_n': r.post_nms_top_n(),
        'nms_thresh': r.nms_thresh, 'score_thresh': r.score_thresh,
        'minimum_box_size': r.min_size,
        'anchor_sizes': r.anchor_generator.sizes,
        'anchor_aspect_ratios': r.anchor_generator.aspect_ratios,
        'transform_min_size': t.min_size, 'transform_max_size': t.max_size,
        'transform_mean': t.image_mean, 'transform_std': t.image_std,
        'roi_feature_keys': detector.roi_heads.box_roi_pool.featmap_names,
        'proposal_order': 'Original filter_proposals output: batched_nms returns decreasing objectness score; keep is sliced by original post_nms_top_n, then [:K].',
        'coordinate_system': 'xyxy pixels in transformed, unpadded ImageList.image_sizes; GT captured from the same original forward call',
    }


class CaptureRPN:
    """Read-only forward hooks; all handles removed even after exceptions."""
    def __init__(self, detector):
        self.detector = detector
        self.handles = []
        self.reset()

    def reset(self):
        self.transform_calls = 0
        self.rpn_calls = 0
        self.transformed = None
        self.proposals = None

    def __enter__(self):
        self.handles = [self.detector.transform.register_forward_hook(self.transform_hook),
                        self.detector.rpn.register_forward_hook(self.rpn_hook)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def transform_hook(self, module, args, output):
        self.transform_calls += 1
        images, targets = output
        if targets is None:
            raise ValueError('Pass GT through the original eval forward to capture its transform')
        self.transformed = (list(images.image_sizes), [t['boxes'].detach().cpu().clone() for t in targets])

    def rpn_hook(self, module, args, output):
        self.rpn_calls += 1
        images, _, targets = args
        if self.transformed is None or targets is None:
            raise ValueError('Missing original transformed targets')
        sizes, boxes = self.transformed
        if list(images.image_sizes) != sizes:
            raise ValueError('Transform/RPN image sizes differ')
        for before, target in zip(boxes, targets):
            torch.testing.assert_close(before, target['boxes'].detach().cpu(), rtol=0, atol=0)
        proposals, losses = output
        if losses:
            raise ValueError('RPN eval unexpectedly returned training losses')
        self.proposals = [p.detach().cpu().clone() for p in proposals]


def statistics(values):
    if not values:
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None,
                'quantiles': {}, 'histogram': {'edges': [i / 10 for i in range(11)], 'counts': [0] * 10}}
    t = torch.tensor(values, dtype=torch.float64)
    return {'count': len(values), 'mean': t.mean().item(), 'std': t.std(unbiased=False).item(),
            'min': t.min().item(), 'max': t.max().item(),
            'quantiles': {str(q): torch.quantile(t, q).item() for q in (0.10, 0.25, 0.50, 0.75, 0.90)},
            'histogram': {'edges': [i / 10 for i in range(11)], 'counts': torch.histc(t, bins=10, min=0, max=1).to(torch.int64).tolist()}}


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_inputs(args, data):
    files = [Path(__file__), ROOT / 'week08/day01/llvip_dataset.py',
             ROOT / 'week08/day02/paired_letterbox.py', ROOT / 'week08/day02/paired_horizontal_flip.py',
             ROOT / 'week08/day03/ir_common.py', ROOT / 'week08/day05/fusion_common.py',
             ROOT / 'week08/day05/fusion_model.py', args.data_root / 'manifest.csv']
    from torchvision.models.detection import faster_rcnn, rpn, generalized_rcnn, transform
    from torchvision.ops import boxes as box_ops
    files += [Path(inspect.getfile(m)) for m in (faster_rcnn, rpn, generalized_rcnn, transform, box_ops)]
    hashes = {str(p.resolve()): ir.sha(p) for p in files}
    # The original RPN output order is the semantic basis of TopK; fail closed
    # if this installed implementation no longer has the audited NMS/slice path.
    source = inspect.getsource(rpn.RegionProposalNetwork.filter_proposals)
    fragments = ('box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)',
                 'keep = keep[: self.post_nms_top_n()]', 'boxes, scores = boxes[keep], scores[keep]')
    if not all(fragment in source for fragment in fragments):
        raise RuntimeError('Installed RPN filtering implementation differs; audit TopK score ordering before proceeding')
    records = data.dataset.records
    all_splits = {}
    with (args.data_root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            info = all_splits.setdefault(row['split'], {'images': 0, 'gt_boxes': 0, 'scenes': set(), 'empty_gt_images': 0})
            info['images'] += 1
            info['gt_boxes'] += int(row['person_boxes'])
            info['empty_gt_images'] += int(int(row['person_boxes']) == 0)
            info['scenes'].add(row['scene_id'])
    for info in all_splits.values():
        info['scenes'] = sorted(info['scenes'])
    scene_sets = [set(v['scenes']) for v in all_splits.values()]
    if any(a & b for i, a in enumerate(scene_sets) for b in scene_sets[i + 1:]):
        raise ValueError('Manifest scene splits overlap')
    if len(records) != 150 or sum(r.person_boxes for r in records) != 416:
        raise ValueError('Expected original validation split of 150 images / 416 GT')
    selected = records[:args.limit or len(records)]
    stems = [r.stem for r in selected]
    return {'file_sha256': hashes, 'manifest_splits': all_splits,
            'validation_order': 'manifest order; shuffle=False; batch_size=1; num_workers=0',
            'selected_stems': stems, 'selected_stems_sha256': hashlib.sha256('\n'.join(stems).encode()).hexdigest(),
            'selected_images': len(selected), 'expected_selected_gt': sum(r.person_boxes for r in selected),
            'selected_scenes': sorted({r.scene_id for r in selected}),
            'selected_data_paths': [{'stem': r.stem, 'visible': str(r.visible_path), 'infrared': str(r.infrared_path), 'annotation': str(r.annotation_path)} for r in selected],
            'test_images_read': False, 'scene_isolation_checked': True}


@torch.inference_mode()
def diagnose(name, checkpoint, args, data, expected_manifest_sha):
    started = time.perf_counter()
    digest_before = ir.sha(checkpoint)
    print(f'LOAD model={name} checkpoint={checkpoint} sha256={digest_before}', flush=True)
    restore = ir.checkpoint_model if name == 'IR' else fusion.restore
    model, ck = restore(checkpoint, args.device)
    model.eval()
    detector = model if name == 'IR' else model.detector
    metadata = {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                'sha256_before': digest_before, 'epoch': ck['epoch'], 'training_config': ck['config']}
    if ck['config']['manifest_sha256'] != expected_manifest_sha:
        raise ValueError(f'{name}: checkpoint manifest checksum differs')
    del ck
    configuration = rpn_configuration(detector)
    print(f'RPN_CONFIG model={name} {json.dumps(configuration, ensure_ascii=False)}', flush=True)
    state_before = tensor_digest(model)
    ir_data = ir.loader(args.data_root, 'val').dataset if name == 'IR' else None
    image_rows, gt_rows, maximums = [], [], {k: [] for k in TOP_K}
    with CaptureRPN(detector) as capture:
        for index, (images, targets) in enumerate(data):
            if args.limit and index >= args.limit:
                break
            record = data.dataset.get_record(index)
            if len(images) != 1 or len(targets) != 1:
                raise ValueError('Original batch size must remain one')
            if ir_data is not None:
                original_ir, original_target = ir_data[index]
                if ir_data.get_record(index).stem != record.stem:
                    raise ValueError('IR/Fusion dataset order differs')
                torch.testing.assert_close(images[0]['infrared'], original_ir, rtol=0, atol=0)
                for key in targets[0]:
                    torch.testing.assert_close(targets[0][key], original_target[key], rtol=0, atol=0)
            for modality, image in images[0].items():
                if image.shape != (3, 640, 640) or image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
                    raise ValueError(f'{record.stem}: invalid {modality} input')
            moved_targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            model_images = ([im['infrared'].to(args.device) for im in images] if name == 'IR' else
                            [{k: v.to(args.device) for k, v in im.items()} for im in images])
            capture.reset()
            # Complete original forward; final RoI detections are deliberately unused.
            model(model_images, moved_targets)
            if capture.transform_calls != 1 or capture.rpn_calls != 1 or capture.proposals is None:
                raise ValueError('Expected one transform and one RPN call per image')
            sizes, gt_batches = capture.transformed
            if len(sizes) != 1 or sizes[0] != (640, 640):
                raise ValueError('Original 640x640 geometry unexpectedly changed')
            gt, proposals = gt_batches[0], capture.proposals[0]
            torch.testing.assert_close(gt, targets[0]['boxes'], rtol=0, atol=0)
            check_boxes(gt, sizes[0], f'{name}/{record.stem}/GT')
            check_boxes(proposals, sizes[0], f'{name}/{record.stem}/RPN')
            if len(proposals) > configuration['active_post_nms_top_n']:
                raise ValueError('Proposals exceed the unchanged original RPN budget')
            if len(gt) != record.person_boxes:
                raise ValueError('GT count changed during transforms')
            row = {'model': name, 'image_index': index, 'stem': record.stem, 'scene_id': record.scene_id,
                   'gt_count': len(gt), 'proposal_count': len(proposals), 'empty_gt': int(len(gt) == 0),
                   'transformed_height': sizes[0][0], 'transformed_width': sizes[0][1]}
            per_k = {}
            for k in TOP_K:
                used = proposals[:k]
                best = box_iou(gt, used).max(dim=1).values if len(gt) and len(used) else torch.zeros(len(gt))
                if not bool(torch.isfinite(best).all()):
                    raise FloatingPointError('Nonfinite max IoU')
                maximums[k].extend(best.tolist())
                per_k[k] = best.tolist()
                row[f'used_top{k}'] = len(used)
                row[f'fewer_than_top{k}'] = int(len(proposals) < k)
                row[f'mean_max_iou_top{k}'] = best.mean().item() if len(gt) else None
                for threshold in THRESHOLDS:
                    hits = int((best >= threshold).sum())
                    row[f'hit_top{k}_iou{threshold:.2f}'] = hits
                    row[f'recall_top{k}_iou{threshold:.2f}'] = hits / len(gt) if len(gt) else None
            image_rows.append(row)
            for gt_index, box in enumerate(gt.tolist()):
                gt_row = {'model': name, 'image_index': index, 'stem': record.stem, 'scene_id': record.scene_id,
                          'gt_index': gt_index, 'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3],
                          'proposal_count': len(proposals)}
                for k in TOP_K:
                    gt_row[f'max_iou_top{k}'] = per_k[k][gt_index]
                    for threshold in THRESHOLDS:
                        gt_row[f'covered_top{k}_iou{threshold:.2f}'] = int(per_k[k][gt_index] >= threshold)
                gt_rows.append(gt_row)
            print(f'IMAGE model={name} index={index + 1} stem={record.stem} gt={len(gt)} proposals={len(proposals)} '
                  f'recall100@.75={row["recall_top100_iou0.75"]}', flush=True)
    if not image_rows:
        raise ValueError('No validation images processed')
    state_after = tensor_digest(model)
    if state_before != state_after:
        raise RuntimeError(f'{name}: persistent model state changed during inference')
    if configuration != rpn_configuration(detector):
        raise RuntimeError(f'{name}: original RPN/transform configuration changed')
    metadata['sha256_after'] = ir.sha(checkpoint)
    metadata['unchanged'] = metadata['sha256_after'] == digest_before
    if not metadata['unchanged']:
        raise RuntimeError(f'{name}: checkpoint file changed')
    total_gt = sum(row['gt_count'] for row in image_rows)
    summary = {'checkpoint': metadata, 'rpn_config': configuration,
               'processed_images': len(image_rows), 'total_valid_gt': total_gt,
               'empty_gt_images': sum(row['empty_gt'] for row in image_rows),
               'zero_proposal_images': sum(row['proposal_count'] == 0 for row in image_rows),
               'proposal_count_statistics': statistics([row['proposal_count'] for row in image_rows]),
               'top_k': {}, 'dataset_ir_pair_equivalence_verified': name == 'IR',
               'state_sha256_before': state_before, 'state_sha256_after': state_after,
               'persistent_state_unchanged': True, 'rpn_configuration_unchanged': True,
               'elapsed_seconds': time.perf_counter() - started}
    # Proposal counts are unbounded counts, so omit the IoU-specific histogram.
    summary['proposal_count_statistics'].pop('histogram')
    for k in TOP_K:
        s = statistics(maximums[k])
        hits = {f'{t:.2f}': sum(row[f'hit_top{k}_iou{t:.2f}'] for row in image_rows) for t in THRESHOLDS}
        summary['top_k'][str(k)] = {'recall': {t: n / total_gt if total_gt else None for t, n in hits.items()},
                                  'hits': hits, 'denominator_valid_gt': total_gt, 'mean_max_iou': s['mean'],
                                  'max_iou_statistics': s,
                                  'images_with_fewer_proposals_than_k': sum(row[f'fewer_than_top{k}'] for row in image_rows),
                                  'effective_proposals_min': min(row[f'used_top{k}'] for row in image_rows),
                                  'effective_proposals_max': max(row[f'used_top{k}'] for row in image_rows)}
    print(f'MODEL_DONE model={name} images={len(image_rows)} gt={total_gt} state_unchanged=True checkpoint_unchanged=True', flush=True)
    del model, detector
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return summary, image_rows, gt_rows


def compare(ir_rows, fusion_rows):
    if len(ir_rows) != len(fusion_rows):
        raise ValueError('IR/Fusion processed image counts differ')
    differences = []
    for a, b in zip(ir_rows, fusion_rows):
        for field in ('image_index', 'stem', 'scene_id', 'gt_count'):
            if a[field] != b[field]:
                raise ValueError(f'IR/Fusion per-image mismatch: {field}')
        row = {'image_index': a['image_index'], 'stem': a['stem'], 'scene_id': a['scene_id'], 'gt_count': a['gt_count'],
               'IR_proposal_count': a['proposal_count'], 'Fusion_proposal_count': b['proposal_count'],
               'proposal_count_difference_fusion_minus_ir': b['proposal_count'] - a['proposal_count']}
        for k in TOP_K:
            for t in THRESHOLDS:
                field = f'recall_top{k}_iou{t:.2f}'
                row[f'IR_{field}'] = a[field]
                row[f'Fusion_{field}'] = b[field]
                row[f'difference_fusion_minus_ir_{field}'] = b[field] - a[field] if a[field] is not None else None
            field = f'mean_max_iou_top{k}'
            row[f'IR_{field}'], row[f'Fusion_{field}'] = a[field], b[field]
            row[f'difference_fusion_minus_ir_{field}'] = b[field] - a[field] if a[field] is not None else None
            row[f'IR_used_top{k}'], row[f'Fusion_used_top{k}'] = a[f'used_top{k}'], b[f'used_top{k}']
            row[f'IR_fewer_than_top{k}'], row[f'Fusion_fewer_than_top{k}'] = a[f'fewer_than_top{k}'], b[f'fewer_than_top{k}']
            for t in THRESHOLDS:
                hit = f'hit_top{k}_iou{t:.2f}'
                row[f'IR_{hit}'], row[f'Fusion_{hit}'] = a[hit], b[hit]
        differences.append(row)
    return differences


def plots(output, models, gt_rows, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label = ' (SMOKE / partial validation)' if smoke else ' (validation)'
    caps = {name: models[name]['rpn_config']['active_post_nms_top_n'] for name in ('IR', 'Fusion')}
    budget_note = (f'Original post-NMS caps: IR={caps["IR"]}, Fusion={caps["Fusion"]}. '
                   'K >= cap uses all actual retained proposals (counts may be lower).')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, threshold in zip(axes, THRESHOLDS):
        for name, marker in (('IR', 'o'), ('Fusion', 's')):
            axis.plot(TOP_K, [models[name]['top_k'][str(k)]['recall'][f'{threshold:.2f}'] for k in TOP_K], marker=marker, label=name)
        axis.set(title=f'Proposal Recall at IoU {threshold:.2f}', xlabel='Requested TopK (original RPN cap unchanged)', ylim=(0, 1.02))
        axis.set_xticks(TOP_K)
        axis.grid(alpha=.3)
        axis.legend()
    axes[0].set_ylabel('Covered GT / all valid GT')
    fig.suptitle('IR vs Fusion RPN proposals' + label)
    fig.text(.5, .01, budget_note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / 'proposal_recall_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    bins = [i / 20 for i in range(21)]
    for axis, k in zip(axes, TOP_K):
        for name in ('IR', 'Fusion'):
            values = [row[f'max_iou_top{k}'] for row in gt_rows if row['model'] == name]
            axis.hist(values, bins=bins, histtype='step', linewidth=1.8, label=name)
        axis.set(title=f'Top{k}', xlabel='Maximum proposal IoU per GT', xlim=(0, 1))
        axis.axvline(.5, color='gray', linestyle=':', alpha=.6)
        axis.axvline(.75, color='gray', linestyle='--', alpha=.6)
        axis.legend()
    axes[0].set_ylabel('Number of GT boxes')
    fig.suptitle('GT maximum IoU distribution' + label)
    fig.text(.5, .01, budget_note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / 'max_iou_distribution.png', dpi=160)
    plt.close(fig)


def report(output, result):
    models, run = result['models'], result['run']
    smoke = run['scope'] == 'smoke'
    lines = ['# Week09 Day01：RPN Proposal Recall 诊断', '',
             f'状态：{"小样本冒烟完成，不能代替全验证集结论" if smoke else "全验证集诊断完成"}。', '',
             '## 实验边界', '',
             '仅使用原 validation split；未读取 test 图像，未训练、未创建优化器、未更新模型权重。完整原模型 eval forward 中，通过临时只读 hook 捕获 transform 后 GT 与 RPN 后处理候选框；最终 RoI 检测结果未用于本次指标。', '',
             '候选框和 GT 均为同次内部 transform 后的 xyxy 像素坐标，并实际检查640×640尺寸及GT坐标不变。原 RPN 后处理已按 objectness 降序返回，TopK直接切片；保留原 pre/post NMS、score threshold、anchor、特征键与RoI配置。', '',
             '全局 Recall = 达到阈值的 GT 总数 / 全部有效 GT 总数；不是逐图 Recall 的平均。空GT图记入图片/候选统计，其逐图Recall为空，不进入GT分母。无候选时每个GT最大IoU为0。', '',
             '## 真实运行', '', f'- 范围：{run["scope"]}；每模型 {models["IR"]["processed_images"]} 图、{models["IR"]["total_valid_gt"]} 个GT。',
             f'- 开始时间：{run["started_at"]}；固定种子：{run["seed"]}。',
             f'- 设备：{run["environment"]["device"]}；GPU：{run["environment"]["gpu"]}。',
             f'- Python：{run["environment"]["python"]}；Torch：{run["environment"]["torch"]}；TorchVision：{run["environment"]["torchvision"]}。',
             '- 真实命令：', '', '```text', run['command'], '```', '',
             '## 观测结果', '', '|模型|TopK|有效候选范围|Recall@0.50|Recall@0.75|平均最大IoU|候选不足K的图数|', '|---|---:|---|---:|---:|---:|---:|']
    for name in ('IR', 'Fusion'):
        for k in TOP_K:
            s = models[name]['top_k'][str(k)]
            fmt = lambda v: f'{v:.6f}' if v is not None else 'NA（无GT）'
            lines.append(f'|{name}|{k}|{s["effective_proposals_min"]}–{s["effective_proposals_max"]}|{fmt(s["recall"]["0.50"])}|{fmt(s["recall"]["0.75"])}|{fmt(s["mean_max_iou"])}|{s["images_with_fewer_proposals_than_k"]}|')
    lines.extend(['', '差值定义为 Fusion − IR；完整命中数、分母、分位数、直方图及逐图差异见JSON/CSV。', '',
                  '|TopK|Recall@0.50差值|Recall@0.75差值|平均最大IoU差值|', '|---:|---:|---:|---:|'])
    for k in TOP_K:
        d = result['comparison'][str(k)]
        fmt = lambda v: f'{v:+.6f}' if v is not None else 'NA'
        lines.append(f'|{k}|{fmt(d["recall_difference"]["0.50"])}|{fmt(d["recall_difference"]["0.75"])}|{fmt(d["mean_max_iou_difference"])}|')
    lines.extend(['', '![Proposal Recall](proposal_recall_comparison.png)', '', '![GT最大IoU分布](max_iou_distribution.png)', '',
                  '## 原配置与完整性', ''])
    for name in ('IR', 'Fusion'):
        m = models[name]
        c = m['rpn_config']
        lines.extend([f'- {name} checkpoint：`{m["checkpoint"]["path"]}`；epoch={m["checkpoint"]["epoch"]}；SHA256=`{m["checkpoint"]["sha256_before"]}`，前后文件校验一致。',
                      f'- {name} eval pre/post NMS top-N={c["active_pre_nms_top_n"]}/{c["active_post_nms_top_n"]}，score={c["score_thresh"]}，NMS IoU={c["nms_thresh"]}；空GT图片={m["empty_gt_images"]}，零候选图片={m["zero_proposal_images"]}。',
                      f'- {name} 持久state前后SHA256一致；原RPN/transform配置未改变。RoI实际特征键：`{c["roi_feature_keys"]}`。'])
    caps = {name: models[name]['rpn_config']['active_post_nms_top_n'] for name in ('IR', 'Fusion')}
    lines.extend(['', f'本次实际post-NMS上限：IR={caps["IR"]}，Fusion={caps["Fusion"]}。当请求TopK达到或超过对应模型上限时，使用全部实际保留候选（实际数还可能更少），不代表曾生成1000个后处理候选。相同指标可能来自同一候选集合。', '',
                  '所有检查的项目源码、TorchVision内部实现及manifest的SHA256、选择样本顺序和环境信息均保存在 `rpn_metrics.json`。未读取服务器以外的历史测试缓存作为本次推断输入。', '',
                  '## 解释与 Week09 Day02 建议', ''])
    if smoke:
        lines.append('本次只证明该样本范围内程序能够获取真实RPN候选并计算指标。全验证集指标、机制归因与Day02分支选择尚未执行，应先完成150图全量诊断。')
    else:
        diffs = [result['comparison'][str(k)]['recall_difference']['0.75'] for k in TOP_K]
        if all(d is not None and d < 0 for d in diffs):
            lines.append('真实观测：Fusion在三个请求TopK下的IoU=0.75候选覆盖率均低于IR。合理假设：差异可能在RPN阶段已经出现；本次未检验统计显著性，也不能据此证明其造成最终AP75下降。')
            lines.append('Day02具体建议：固定本次best权重与validation，先从逐GT表筛选IR覆盖但Fusion未覆盖（IoU=0.75）的目标，按场景、目标面积和最佳IoU差值排序，画出相同图上的GT及两模型原RPN候选，检查分支特征、RPN回归与候选排序的具体失败模式。')
        elif all(d is not None and d >= 0 for d in diffs):
            lines.append('真实观测：Fusion在三个请求TopK下的IoU=0.75候选覆盖率均未低于IR。这削弱“验证集RPN覆盖不足已经解释最终高IoU差异”的假设，但不能排除单图/子集差异或RPN排序影响。')
            lines.append('Day02具体建议：固定本次best权重与validation，在同一次前向中追踪原RPN候选到RoI类别/框回归，再到最终score排序和NMS，比较GT最大IoU在RoI回归前后的变化，并审查高IoU真阳性是否被低分或最终NMS丢弃。')
        else:
            lines.append('真实观测：不同TopK下IoU=0.75差异方向不一致。本次结果混合，不能仅凭候选Recall确定根因。')
            lines.append('Day02具体建议：先区分前100候选排序问题与全部原候选覆盖问题，再查看逐图/逐GT差值最大的场景；固定权重，关联RoI回归前后IoU与最终score/NMS，不根据test调参。')
    lines.extend(['', '仍需验证的机制：RPN候选排序、RoI分类/回归、最终score排序和NMS各自对定位质量的影响。本次是两个best checkpoint的描述性配对诊断，无重训练、干预或显著性检验，不能作因果证明。', '',
                  'RPN Proposal Recall不等于最终检测Recall，也不是AP75。既有最终AP75来自test，而本次RPN来自validation；两者集合不同，不能直接把数值差异作一一对应归因。', '',
                  '运行是否全部完成，以 `rpn_metrics.json` 中 `run.status=completed` 和日志 `COMPLETED` 为准。'])
    with (output / 'rpn_diagnostic_report.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def run(args):
    ir.seed_all(args.seed)
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    data = fusion.loader(args.data_root, 'val', shuffle=False)
    inputs = inspect_inputs(args, data)
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'hostname': platform.node(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
           'numpy': np.__version__, 'device': args.device, 'cuda_build': torch.version.cuda,
           'cudnn': torch.backends.cudnn.version(), 'cudnn_deterministic': torch.backends.cudnn.deterministic,
           'cudnn_benchmark': torch.backends.cudnn.benchmark, 'threads': torch.get_num_threads(),
           'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
           'gpu_memory_bytes': torch.cuda.get_device_properties(args.device).total_memory if args.device.startswith('cuda') else None,
           'deterministic_policy': 'Fixed Python/NumPy/Torch seeds; manifest order; cudnn deterministic=True and benchmark=False. GPU/CPU tied-score NMS can differ; exact cross-device equality is not claimed.'}
    command = subprocess.list2cmdline(sys.orig_argv)
    print('COMMAND', command, flush=True)
    print('ENVIRONMENT', json.dumps(env, ensure_ascii=False), flush=True)
    print(f'INPUTS selected_images={inputs["selected_images"]} expected_gt={inputs["expected_selected_gt"]} order_sha256={inputs["selected_stems_sha256"]}', flush=True)
    result = {'run': {'status': 'running', 'scope': 'smoke' if args.limit and args.limit < 150 else 'full_validation',
                      'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'seed': args.seed, 'command': command,
                      'environment': env, 'output': str(args.output), 'split': 'val', 'requested_limit': args.limit,
                      'training_performed': False, 'optimizer_created': False, 'test_inference_performed': False},
              'inputs': inputs, 'models': {}, 'comparison': {}}
    all_images, all_gts = {}, []
    manifest_sha = ir.sha(args.data_root / 'manifest.csv')
    for name, checkpoint in (('IR', args.ir_checkpoint), ('Fusion', args.fusion_checkpoint)):
        summary, images, gts = diagnose(name, checkpoint, args, data, manifest_sha)
        if summary['processed_images'] != inputs['selected_images'] or summary['total_valid_gt'] != inputs['expected_selected_gt']:
            raise ValueError('Observed sample/GT count differs from selected validation manifest')
        result['models'][name] = summary
        all_images[name] = images
        all_gts.extend(gts)
    for a, b in zip([r for r in all_gts if r['model'] == 'IR'], [r for r in all_gts if r['model'] == 'Fusion']):
        for field in ('stem', 'gt_index', 'xmin', 'ymin', 'xmax', 'ymax'):
            if a[field] != b[field]:
                raise ValueError('IR/Fusion per-GT geometry/order differs')
    differences = compare(all_images['IR'], all_images['Fusion'])
    for k in TOP_K:
        a, b = (result['models'][name]['top_k'][str(k)] for name in ('IR', 'Fusion'))
        result['comparison'][str(k)] = {
            'recall_difference': {t: b['recall'][t] - a['recall'][t] if a['recall'][t] is not None else None for t in ('0.50', '0.75')},
            'mean_max_iou_difference': b['mean_max_iou'] - a['mean_max_iou'] if a['mean_max_iou'] is not None else None}
    # These seven primary artifacts only contain completed computations; failures remain in run.log.
    write_csv(args.output / 'rpn_per_image.csv', differences)
    if all_gts:
        write_csv(args.output / 'rpn_per_gt.csv', all_gts)
    else:
        with (args.output / 'rpn_per_gt.csv').open('x', encoding='utf-8-sig', newline='') as f:
            f.write('model,image_index,stem,scene_id,gt_index,xmin,ymin,xmax,ymax,proposal_count,max_iou_top100,max_iou_top300,max_iou_top1000\n')
    plots(args.output, result['models'], all_gts, result['run']['scope'] == 'smoke')
    result['run']['status'] = 'completed'
    result['run']['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    report(args.output, result)
    save_json(args.output / 'rpn_metrics.json', result)
    print('METRICS', json.dumps({n: result['models'][n]['top_k'] for n in ('IR', 'Fusion')}, ensure_ascii=False), flush=True)
    print('COMPLETED', args.output, flush=True)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/LLVIP_1000')
    parser.add_argument('--output', type=Path, required=True, help='Must be a new, nonexistent directory')
    parser.add_argument('--ir-checkpoint', type=Path, default=ROOT / 'week08/day03/outputs/ir_baseline_run01/best.pth')
    parser.add_argument('--fusion-checkpoint', type=Path, default=ROOT / 'week08/day05/outputs/fusion_concat_run01/best.pth')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--limit', type=int, default=0, help='0: all 150 validation images; N: first N in manifest order')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0 or args.limit > 150 or args.threads < 1:
        parser.error('limit must be 0..150 and threads must be positive')
    for name in ('data_root', 'output', 'ir_checkpoint', 'fusion_checkpoint'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'run.log').open('x', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except BaseException:
                print('FAILED: diagnostic incomplete; no final result may be inferred from partial files.', flush=True)
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
