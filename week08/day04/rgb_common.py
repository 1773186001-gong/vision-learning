"""Day04只增加RGB数据入口；模型、指标、种子及checkpoint实现直接复用Day03。"""
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'week08/day03'))
import ir_common as ir
from train_ir import atomic_save, cuda_sync, smoke, update

MANIFEST_SHA = '8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb'
WEIGHTS_SHA = '907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5'
REFERENCE_RUN = ROOT / 'week08/day03/outputs/ir_baseline_run01'
OUTPUT_ROOT = ROOT / 'week08/day04/outputs'
SETTINGS = dict(model='fasterrcnn_mobilenet_v3_large_320_fpn', epochs=10,
                batch_size=1, num_workers=0, seed=42, lr=0.001,
                momentum=0.9, weight_decay=0.0005, scheduler=None, amp=False,
                letterbox=640, min_size=640, max_size=640, selection='val mAP50_95')


def rgb_loader(root, split, seed=42, shuffle=False):
    # 原Dataset同步处理两模态，但此模式仅返回visible，模型不接收IR。
    ds = ir.LLVIPDataset(root, split, modality='visible', infrared_channels=3,
                         transforms=ir.PairedLetterboxAndFlip(split))
    if len(ds) != {'train':700, 'val':150, 'test':150}[split]:
        raise ValueError(f'{split}样本数量与IR实验不符')
    return ir.DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0,
                        collate_fn=ir.detection_collate_fn,
                        generator=ir.torch.Generator().manual_seed(seed))


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()) or path == OUTPUT_ROOT.resolve():
        raise ValueError('输出只能位于week08/day04/outputs的独立run目录，禁止写入IR结果或原数据')
    return path


def write_new_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def protected_hashes():
    expected = read_json(Path(__file__).with_name('protected_inputs.json'))
    actual = {}
    for relative, digest in expected['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('保护清单路径超出项目')
        actual[relative] = ir.sha(path)
        if actual[relative] != digest:
            raise ValueError(f'已有Day01–Day03文件与批准的基线不同：{relative}；停止，不覆盖')
    return actual


def runtime_hashes():
    folder=Path(__file__).parent
    expected=read_json(folder/'upload_manifest.json')['runtime_files']
    actual={}
    for name,digest in expected.items():
        path=(folder/name).resolve()
        if path.parent!=folder.resolve(): raise ValueError('运行包路径非法')
        actual[name]=ir.sha(path)
        if actual[name]!=digest: raise ValueError(f'Day04运行包哈希不同：{name}')
    return actual


def read_reference(reference=REFERENCE_RUN):
    reference = Path(reference).resolve()
    if reference != REFERENCE_RUN.resolve():
        raise ValueError('本次对照必须使用既有IR run01')
    config = read_json(reference / 'config.json')
    if any(config.get(k) != v for k,v in SETTINGS.items()):
        raise ValueError('IR参考配置与本次公平对照设定不一致')
    if config['manifest_sha256'] != MANIFEST_SHA or config['weights']['sha256'] != WEIGHTS_SHA:
        raise ValueError('IR数据清单或预训练权重哈希不符')
    with (reference / 'training_log.csv').open(encoding='utf-8-sig', newline='') as stream:
        history = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    if [r['epoch'] for r in history] != list(range(1,11)):
        raise ValueError('IR参考历史不是完整10轮')
    completed = read_json(reference / 'completed.json')
    scores = read_json(reference / 'test_metrics.json')
    if scores != completed['test']:
        raise ValueError('IR已保存test指标与完成标记不一致')
    return {'path': str(reference), 'config': config, 'history':history,
            'best_epoch':completed['best_epoch'], 'test':scores}


def comparison(config, history, best_epoch, scores, reference):
    for key,value in SETTINGS.items():
        if config[key] != value or reference['config'][key] != value:
            raise ValueError(f'对照配置改变：{key}')
    for key in ('manifest_sha256','torch','torchvision'):
        if config[key] != reference['config'][key]:
            raise ValueError(f'对照环境或数据改变：{key}')
    if config['weights']['sha256'] != reference['config']['weights']['sha256']:
        raise ValueError('预训练初始化文件不一致')
    if scores['protocol'] != reference['test']['protocol'] or scores['score_threshold'] != reference['test']['score_threshold']:
        raise ValueError('两次评估协议不一致')
    fields = ('precision','recall','AP50','mAP50_95','TP','FP','FN')
    return {'fair_settings_match':True, 'changed_model_input':'infrared -> visible',
            'protocol':scores['protocol'], 'IR':reference['test'], 'RGB':scores,
            'RGB_minus_IR':{k:scores[k]-reference['test'][k] for k in fields},
            'IR_best_epoch':reference['best_epoch'],'RGB_best_epoch':best_epoch,
            'IR_epoch_seconds_sum':sum(r['epoch_seconds'] for r in reference['history']),
            'RGB_epoch_seconds_sum':sum(r['epoch_seconds'] for r in history),
            'IR_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in reference['history']),
            'RGB_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in history),
            'IR_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in reference['history']),
            'RGB_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in history),
            'limitation':'单次、单seed、场景隔离子集；耗时非稳定基准；不推断融合成绩'}
