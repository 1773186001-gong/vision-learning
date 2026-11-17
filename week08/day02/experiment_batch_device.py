"""Step3：真实 LLVIP 批次和 CPU/CUDA 传输验证，不创建或训练模型。

明确 --device cpu 或 cuda:0；不自动回退、不建立远程连接。
"""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
DAY02 = Path(__file__).resolve().parent
DAY01 = DAY02.parent / "day01"
sys.path.insert(0, str(DAY01))

import torch
from torch.utils.data import DataLoader, Subset
from llvip_dataset import LLVIPDataset, detection_collate_fn
from paired_horizontal_flip import PairedLetterboxAndFlip


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory(device):
    if device.type != "cuda":
        return None
    sync(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free, "device_total_bytes": total,
    }


def walk_batch(images, targets):
    for index, image in enumerate(images):
        for key, value in image.items():
            yield f"images[{index}].{key}", value
    for index, target in enumerate(targets):
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                yield f"targets[{index}].{key}", value


def describe(images, targets):
    result = {}
    for name, tensor in walk_batch(images, targets):
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "device": str(tensor.device)}
        print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    return result


def validate_batch(images, targets, device):
    require(isinstance(images, list) and isinstance(targets, list), "collate 后应为 list")
    require(len(images) == len(targets) == 2, "batch size 必须为2")
    for image, target in zip(images, targets):
        require(set(image) == {"visible", "infrared"}, "必须包含 RGB/IR")
        for tensor in image.values():
            require(tensor.shape == (3, 640, 640) and tensor.dtype == torch.float32,
                    "图像应为 float32 [3,640,640]")
            require(bool(torch.isfinite(tensor).all()), "图像含非有限数")
        boxes = target["boxes"]
        require(boxes.dtype == torch.float32 and boxes.ndim == 2 and boxes.shape[1] == 4,
                "boxes 应为 float32 [N,4]")
        count = boxes.shape[0]
        require(target["area"].shape == (count,) and target["area"].dtype == torch.float32,
                "area 形状/类型错误")
        for key in ("labels", "iscrowd"):
            require(target[key].shape == (count,) and target[key].dtype == torch.int64,
                    f"{key} 形状/类型错误")
        require(target["image_id"].shape == (1,) and target["image_id"].dtype == torch.int64,
                "image_id 错误")
        require(bool(torch.isfinite(boxes).all()), "boxes 含非有限数")
        require(bool((boxes[:, 2:] > boxes[:, :2]).all()), "非法框")
        require(bool(((boxes >= 0) & (boxes <= 640)).all()), "框越界")
        require(bool((target["labels"] == 1).all()), "person 类别应为1")
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        torch.testing.assert_close(area, target["area"], rtol=1e-6, atol=1e-6)
    for name, tensor in walk_batch(images, targets):
        require(tensor.device == device, f"{name} 未到达 {device}")


def move_batch(images, targets, device):
    # 保留 list 和每张图不同长度的 target；不指定 dtype，不使用半精度。
    moved_images = [{k: v.to(device, non_blocking=False, copy=True)
                     for k, v in image.items()} for image in images]
    moved_targets = [{k: (v.to(device, non_blocking=False, copy=True)
                         if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                      for k, v in target.items()} for target in targets]
    return moved_images, moved_targets


def equal_batch(source_images, source_targets, images, targets):
    source = dict(walk_batch(source_images, source_targets))
    moved = dict(walk_batch(images, targets))
    require(source.keys() == moved.keys(), "传输前后字段变化")
    for name, original in source.items():
        actual = moved[name].detach().cpu()
        require(original.shape == actual.shape and original.dtype == actual.dtype,
                f"{name} shape/dtype 变化")
        # float32→float32 的纯拷贝没有计算舍入，精确比较是适当的。
        require(torch.equal(original, actual), f"{name} 数值变化")


def basic_ops(images, targets):
    blends, areas, means = [], [], []
    for image, target in zip(images, targets):
        blended = image["visible"] * 0.5 + image["infrared"] * 0.5
        blends.append(blended)
        means.append(blended.mean())
        boxes = target["boxes"]
        areas.append((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    return blends, areas, means


def fingerprint(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def run(args, output):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    report = {"requested_device": str(device), "seed": args.seed,
              "torch": torch.__version__, "python": sys.executable,
              "cuda_build": torch.version.cuda,
              "gpu_status": "pending_not_run_cpu" if not is_cuda else "running",
              "gpu_name": None, "gpu_memory": None, "batches": []}
    print(f"Week08 Day02 Step3 | device={device} | seed={args.seed} | batch_size=2")
    print(f"Python={sys.executable}; torch={torch.__version__}; CUDA build={torch.version.cuda}")
    if is_cuda:
        expected_python = Path(r"D:\cv-server-env\Scripts\python.exe")
        require(Path(sys.executable).resolve() == expected_python,
                f"GPU 实验请使用服务器解释器 {expected_python}")
        require(torch.cuda.is_available(), "CUDA 不可用；停止，禁止回退 CPU")
        torch.cuda.set_device(device)
        report["gpu_name"] = torch.cuda.get_device_name(device)
        require("RTX 3060" in report["gpu_name"], "本实验目标为 RTX 3060，请核对设备")
        warmup = torch.ones(1, device=device) + 1
        sync(device)
        del warmup
        torch.cuda.reset_peak_memory_stats(device)
        report["memory_baseline"] = memory(device)
        print(f"实际 GPU={report['gpu_name']}; baseline={report['memory_baseline']}")
    else:
        print("CPU 逻辑验证；GPU 名称、显存、CUDA 传输/计算均待服务器实测")

    # 原数据路径先做存在性检查；不创建或同步数据目录。
    root = args.data_root.resolve()
    require(root.is_dir() and (root / "manifest.csv").is_file(), f"数据子集不存在：{root}")
    protected = list(DAY01.glob("*.py")) + list(DAY01.glob("*.md"))
    protected += [DAY02 / name for name in (
        "paired_letterbox.py", "paired_horizontal_flip.py", "experiment_letterbox.py",
        "experiment_horizontal_flip.py", "DAY02_STEP01_GPT_HANDOFF.md", "DAY02_STEP02_GPT_HANDOFF.md")]
    for folder in ("step01_letterbox", "step02_horizontal_flip"):
        protected += list((DAY02 / "outputs" / folder).glob("*"))
    protected += [root / "manifest.csv", root / "summary.json"]

    # 每个 split 一个真实 batch；train 有一个正样本及一个合法空标注。
    selections = {}
    datasets = {}
    for split in ("train", "val", "test"):
        pipeline = PairedLetterboxAndFlip(split, generator=torch.Generator().manual_seed(args.seed))
        expected_p = 0.5 if split == "train" else 0.0
        require(pipeline.horizontal_flip.p == expected_p, "预处理概率配置错误")
        dataset = LLVIPDataset(root, split, modality="paired", transforms=pipeline)
        indices = ([next(i for i, r in enumerate(dataset.records) if not r.is_negative),
                    next(i for i, r in enumerate(dataset.records) if r.is_negative)]
                   if split == "train" else [0, 1])
        datasets[split], selections[split] = dataset, indices
        for index in indices:
            rec = dataset.get_record(index)
            protected += [rec.visible_path, rec.infrared_path, rec.annotation_path]
    hashes = fingerprint(protected)
    for split, ds in datasets.items():
        indices = selections[split]
        loader = DataLoader(Subset(ds, indices), batch_size=2, shuffle=False, num_workers=0,
                            pin_memory=False, collate_fn=detection_collate_fn,
                            generator=torch.Generator().manual_seed(args.seed))
        cpu_images, cpu_targets = next(iter(loader))
        validate_batch(cpu_images, cpu_targets, torch.device("cpu"))
        stems = [ds.get_record(i).stem for i in indices]
        print(f"\n[{split}] stems={stems}; flip_p={ds.transforms.horizontal_flip.p}; CPU before:")
        before_structure = describe(cpu_images, cpu_targets)
        # 同种子独立 pipeline 做复现比对，不重写任何图像变换。
        ref = LLVIPDataset(root, split, modality="paired", transforms=PairedLetterboxAndFlip(
            split, generator=torch.Generator().manual_seed(args.seed)))
        reference = [ref[i] for i in indices]
        ref_images, ref_targets = detection_collate_fn(reference)
        equal_batch(ref_images, ref_targets, cpu_images, cpu_targets)
        if split != "train":
            repeated = [ds[i] for i in indices]
            repeat_images, repeat_targets = detection_collate_fn(repeated)
            equal_batch(cpu_images, cpu_targets, repeat_images, repeat_targets)

        sync(device)
        start = time.perf_counter()
        images, targets = move_batch(cpu_images, cpu_targets, device)
        sync(device)
        transfer_ms = (time.perf_counter() - start) * 1000
        after_transfer_memory = memory(device)
        print(f"{device} after:")
        after_structure = describe(images, targets)
        validate_batch(images, targets, device)
        equal_batch(cpu_images, cpu_targets, images, targets)
        cpu_expected = basic_ops(cpu_images, cpu_targets)
        sync(device)
        start = time.perf_counter()
        computed = basic_ops(images, targets)
        sync(device)
        compute_ms = (time.perf_counter() - start) * 1000
        after_compute_memory = memory(device)
        for actual_group, expected_group in zip(computed, cpu_expected):
            for actual, expected in zip(actual_group, expected_group):
                require(actual.device == device, "基础运算未在目标设备执行")
                # 包含归约 mean；CPU/CUDA 归约顺序可不同。
                torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
        means = [v.item() for v in computed[2]]
        print(f"PASS: 全部 Tensor 保持 shape/dtype/数值；基础运算结果与 CPU 一致")
        print(f"blend_means={means}; copy_ms={transfer_ms:.3f}; basic_ops_ms={compute_ms:.3f}")
        if is_cuda:
            print(f"显存 after_transfer={after_transfer_memory}; after_ops={after_compute_memory}")
        record = {
            "split": split, "stems": stems, "flip_p": ds.transforms.horizontal_flip.p,
            "before": before_structure, "after": after_structure,
            "targets_cpu": [{k: v.tolist() for k, v in t.items()} for t in cpu_targets],
            "targets_after": [{k: v.cpu().tolist() for k, v in t.items()} for t in targets],
            "transfer_exact": True, "basic_ops_close": True,
            "blend_means": means, "transfer_ms": transfer_ms, "basic_ops_ms": compute_ms,
            "memory_after_transfer": after_transfer_memory,
            "memory_after_compute": after_compute_memory,
        }
        report["batches"].append(record)
        del images, targets, computed
    require(hashes == fingerprint(protected), "受保护代码/产物或源文件发生变化")
    report["protected_sha256"] = hashes
    report["gpu_memory"] = memory(device)
    report["gpu_status"] = "passed" if is_cuda else "pending_not_run_cpu"
    report["current_device_checks_passed"] = True
    report["comparison_policy"] = {"copy": "same dtype, exact torch.equal",
                                    "operations": {"rtol": 1e-5, "atol": 1e-6}}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPASS: 原代码/Step1/Step2产物和所用源数据哈希未变")
    print(f"当前设备验证通过；GPU status={report['gpu_status']}")
    print(f"结果：{output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists() or not any(output.iterdir()), "输出目录非空，禁止覆盖")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "terminal_output.txt").open("w", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                with torch.no_grad():
                    run(args, output)
            except Exception as error:
                traceback.print_exc()
                (output / "failure.json").write_text(json.dumps(
                    {"status": "failed", "device": args.device, "error": str(error)},
                    ensure_ascii=False, indent=2), encoding="utf-8")
                raise SystemExit(1)


if __name__ == "__main__":
    main()
