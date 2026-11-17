"""Day03 公用组件：已有 IR Dataset/同步预处理、严格权重缓存、单类别检测指标。"""
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'week08/day01'))
sys.path.insert(0, str(ROOT / 'week08/day02'))
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, FrozenBatchNorm2d
from llvip_dataset import LLVIPDataset, detection_collate_fn
from paired_horizontal_flip import PairedLetterboxAndFlip

LOSS_KEYS = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def weights_info():
    url = Weights.DEFAULT.url
    path = Path(torch.hub.get_dir()) / 'checkpoints' / url.rsplit('/', 1)[-1]
    return {'source': url, 'path': str(path), 'exists': path.is_file()}


def frozen_norms(module):
    # weights=None 构造器默认普通BN；恢复COCO结构时显式换成FrozenBN。
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=1e-5))
        else:
            frozen_norms(child)


def build_model(pretrained):
    kwargs = dict(min_size=640, max_size=640, box_score_thresh=0.001,
                  box_detections_per_img=100)
    if pretrained:
        info = weights_info()
        path = Path(info['path'])
        if not path.is_file():
            raise FileNotFoundError(f"预训练权重未缓存，禁止自动下载或随机退化：{info}")
        checksum = sha(path)
        if not checksum.startswith('907ea3f9'):
            raise ValueError('预训练权重SHA256与官方文件名哈希前缀不符')
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=Weights.DEFAULT, progress=False, **kwargs)
    else:
        # 仅用于恢复完整checkpoint；调用者随即 strict=True 加载全部模型状态。
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, **kwargs)
        frozen_norms(model)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def loader(root, split, seed=42, shuffle=False):
    # Dataset内部仍检查配对完整性，但只把IR Tensor返回并输入模型。
    ds = LLVIPDataset(root, split, modality='infrared', infrared_channels=3,
                      transforms=PairedLetterboxAndFlip(split))
    expected = {'train': 700, 'val': 150, 'test': 150}[split]
    if len(ds) != expected:
        raise ValueError(f'{split} 数量不是 {expected}')
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0,
                      collate_fn=detection_collate_fn,
                      generator=torch.Generator().manual_seed(seed))


def move(images, targets, device):
    return ([x.to(device) for x in images],
            [{k: v.to(device) for k, v in t.items()} for t in targets])


def match(predictions, truths, threshold, score_threshold=0.0):
    """每张图按分数贪心一对一匹配，再做数据集级分数排序；无crowd/ignore。"""
    events = []
    total = sum(len(t) for t in truths)
    for p, gt in zip(predictions, truths):
        order = torch.argsort(p['scores'], descending=True, stable=True)
        used = set()
        for idx in order.tolist():
            score = float(p['scores'][idx])
            if score < score_threshold:
                continue
            hit = False
            if len(gt):
                overlaps = box_iou(p['boxes'][idx:idx+1], gt)[0]
                for j in used:
                    overlaps[j] = -1
                value, j = overlaps.max(0)
                if float(value) >= threshold:
                    used.add(int(j))
                    hit = True
            events.append((score, int(hit)))
    events.sort(key=lambda row: -row[0])
    return events, total


def metrics(predictions, truths, score_threshold=0.5):
    aps = []
    for i in range(10):
        events, total = match(predictions, truths, 0.5 + 0.05*i)
        tp = 0
        points = []
        for n, (_, correct) in enumerate(events, 1):
            tp += correct
            points.append((tp / max(total, 1), tp / n))
        ap = sum(max((p for r, p in points if r >= k/100), default=0)
                 for k in range(101)) / 101 if total else 0.0
        aps.append(ap)
    events, total = match(predictions, truths, 0.5, score_threshold)
    tp = sum(x[1] for x in events)
    return {'precision': tp / max(len(events), 1), 'recall': tp / max(total, 1),
            'AP50': aps[0], 'mAP50_95': sum(aps)/10,
            'TP': tp, 'FP': len(events)-tp, 'FN': total-tp,
            'score_threshold': score_threshold,
            'protocol': 'single_person_101pt_all_area_max100_no_ignore; not official COCO API'}


@torch.inference_mode()
def infer(model, data, device):
    model.eval()
    predictions, truths, stems = [], [], []
    for index, (images, targets) in enumerate(data):
        outputs = model([x.to(device) for x in images])
        for out, t in zip(outputs, targets):
            p = {k: out[k].detach().cpu() for k in ('boxes', 'scores', 'labels')}
            if not all(torch.isfinite(v).all() for v in p.values()):
                raise FloatingPointError('预测含非有限数')
            predictions.append(p)
            truths.append(t['boxes'].cpu())
        stems.append(data.dataset.get_record(index).stem)
    return predictions, truths, stems


def checkpoint_model(path, device):
    # 只加载本项目自行生成的可信checkpoint；包含optimizer和RNG状态。
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(False)
    model.load_state_dict(ckpt['model'], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in ckpt['trainable_names'])
    return model.to(device), ckpt


def device_check(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available() or 'RTX 3060' not in torch.cuda.get_device_name(device):
            raise RuntimeError('需要实际 RTX 3060 CUDA；禁止自动回退')
    return device


def plot_history(history, output):
    if importlib.util.find_spec('matplotlib') is None:
        print('CURVE DEFERRED: matplotlib未安装；训练CSV已保存，取回后用本地plot_training.py生成曲线。', flush=True)
        return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in ('total_loss',) + LOSS_KEYS:
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    axes[0].legend(fontsize=7)
    for key in ('AP50', 'mAP50_95'):
        axes[1].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(output)/'loss_curve.png')
    plt.close(fig)
    return True


def write_history(history, output):
    with (Path(output)/'training_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output)
