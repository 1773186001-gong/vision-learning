"""真实 LLVIP 同步翻转验证；日志、JSON、对比图独立保存。"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
DAY02 = Path(__file__).resolve().parent
DAY01 = DAY02.parent / "day01"
sys.path.insert(0, str(DAY01))

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as F
from llvip_dataset import LLVIPDataset, detection_collate_fn
from paired_letterbox import PairedLetterbox
from paired_horizontal_flip import PairedRandomHorizontalFlip, PairedLetterboxAndFlip


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


def exact(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def check_result(before, after, target, result, flip):
    width = before[0].shape[-1]
    expected = target["boxes"].clone()
    if flip:
        expected[:, 0] = width - target["boxes"][:, 2]
        expected[:, 2] = width - target["boxes"][:, 0]
    exact(result["boxes"], expected)
    exact(result["boxes"][:, [1, 3]], target["boxes"][:, [1, 3]])
    assert result["boxes"].dtype == torch.float32
    assert result["boxes"].shape == target["boxes"].shape
    for key in ("labels", "image_id", "iscrowd", "area"):
        exact(result[key], target[key])
        if target[key].numel():
            assert result[key].data_ptr() != target[key].data_ptr()
    PairedLetterbox.validate_boxes(result["boxes"], width, before[0].shape[-2])
    for old, new in zip(before, after):
        exact(new, torch.flip(old, [-1]) if flip else old)
        assert old.data_ptr() != new.data_ptr()


def draw(tensor, boxes):
    image = F.to_pil_image(tensor)
    drawing = ImageDraw.Draw(image)
    for i, box in enumerate(boxes.tolist(), 1):
        drawing.rectangle(box, outline=(255, 35, 35), width=2)
        drawing.text((box[0] + 2, box[1] + 2), str(i), fill=(255, 220, 0))
    return image


def make_panel(before, after, target, result, stem, flip):
    panel = Image.new("RGB", (1280, 1340), "white")
    drawing = ImageDraw.Draw(panel)
    for row, (images, labels, title) in enumerate(
        ((before, target, "BEFORE (letterboxed)"),
         (after, result, f"AFTER shared_flip={flip}"))
    ):
        for col, mode in enumerate(("RGB", "IR")):
            x, y = col * 640, row * 670
            drawing.text((x + 8, y + 8), f"{stem} {mode} {title} boxes={len(labels['boxes'])}", fill="black")
            panel.paste(draw(images[col], labels["boxes"]), (x, y + 30))
    return panel


def digest(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def run(args, output):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    raw = LLVIPDataset(args.data_root, "train", modality="paired")
    positive = next(i for i, r in enumerate(raw.records) if not r.is_negative)
    negative = next(i for i, r in enumerate(raw.records) if r.is_negative)
    protected = list(DAY01.glob("*.py")) + list(DAY01.glob("*.md"))
    protected += [DAY02 / "paired_letterbox.py", DAY02 / "experiment_letterbox.py",
                  DAY02 / "DAY02_STEP01_GPT_HANDOFF.md", args.data_root / "manifest.csv"]
    protected += list((DAY02 / "outputs" / "step01_letterbox").glob("*"))
    for index in (positive, negative):
        rec = raw.get_record(index)
        protected += [rec.visible_path, rec.infrared_path, rec.annotation_path]
    before_hashes = digest(protected)
    records, panels = [], []
    print(f"Week08 Day02 Step 2 | seed={args.seed} | CPU | num_workers=0")
    print(f"环境 torch={torch.__version__}; dataset train={len(raw)}")
    letterbox = PairedLetterbox(640)
    forced = PairedRandomHorizontalFlip(p=1)
    skipped = PairedRandomHorizontalFlip(p=0)
    for index in (positive, negative):
        images, target_raw = raw[index]
        rgb, ir, target = letterbox(images["visible"], images["infrared"], target_raw)
        images_snapshot = (rgb.clone(), ir.clone())
        target_snapshot = {k: v.clone() for k, v in target.items()}
        rec = raw.get_record(index)
        print(f"\n[train/{rec.stem}] negative={rec.is_negative}; W=640; boxes={tuple(target['boxes'].shape)}")
        print(f"Letterbox 后/翻转前 xyxy={target['boxes'].tolist()}")
        case = {"stem": rec.stem, "negative": rec.is_negative,
                "boxes_raw": target_raw["boxes"].tolist(),
                "boxes_before_flip": target["boxes"].tolist(), "cases": []}
        for flip, operation in ((True, forced), (False, skipped)):
            out_rgb, out_ir, result = operation(rgb, ir, target)
            check_result((rgb, ir), (out_rgb, out_ir), target, result, flip)
            exact(rgb, images_snapshot[0])
            exact(ir, images_snapshot[1])
            for key in target:
                exact(target[key], target_snapshot[key])
            print(f"p={operation.p}; shared_flip={flip}; xyxy={result['boxes'].tolist()}")
            print(f"area before={target['area'].tolist()}; after={result['area'].tolist()}")
            filename = f"train_{rec.stem}_{'flip' if flip else 'no_flip'}_comparison.jpg"
            panel = make_panel((rgb, ir), (out_rgb, out_ir), target, result, rec.stem, flip)
            panel.save(output / filename, quality=95)
            panels.append(panel)
            case["cases"].append({"flip": flip, "p": operation.p,
                                  "boxes_after": result["boxes"].tolist(),
                                  "boxes_shape": list(result["boxes"].shape),
                                  "area_before": target["area"].tolist(),
                                  "area_after": result["area"].tolist(), "image": filename})
        once_rgb, once_ir, once_t = forced(rgb, ir, target)
        twice_rgb, twice_ir, twice_t = forced(once_rgb, once_ir, once_t)
        exact(twice_rgb, rgb)
        exact(twice_ir, ir)
        for key in target:
            exact(twice_t[key], target[key])
        print("PASS: 强制两分支同步、框/面积/字段、输入不变且无共享存储、双翻转精确恢复")
        case["checks_passed"] = True
        records.append(case)

    # 真实正样本上验证 p=0.5；同种子重建 RNG，比较决策和输出。
    images, t = raw[positive]
    rgb, ir, t = letterbox(images["visible"], images["infrared"], t)
    rng_a = torch.Generator().manual_seed(args.seed)
    rng_b = torch.Generator().manual_seed(args.seed)
    random_a = PairedRandomHorizontalFlip(0.5, rng_a)
    random_b = PairedRandomHorizontalFlip(0.5, rng_b)
    decisions = []
    for _ in range(12):
        decision_a, decision_b = random_a.sample_decision(), random_b.sample_decision()
        assert decision_a == decision_b
        out = random_a.apply(rgb, ir, t, flip=decision_a)
        repeated = random_b.apply(rgb, ir, t, flip=decision_b)
        check_result((rgb, ir), out[:2], t, out[2], decision_a)
        for a, b in zip(out[:2], repeated[:2]):
            exact(a, b)
        exact(out[2]["boxes"], repeated[2]["boxes"])
        decisions.append(decision_a)
    print(f"\np=0.5，固定种子随机序列（True=翻转）：{decisions}")
    print("PASS: 两次独立同种子运行的决策、RGB/IR 和 boxes 相同")

    batches = []
    for split in ("train", "val", "test"):
        pipeline = PairedLetterboxAndFlip(split, generator=torch.Generator().manual_seed(args.seed))
        ds = LLVIPDataset(args.data_root, split, modality="paired", transforms=pipeline)
        indices = [positive, negative] if split == "train" else [0, 1]
        loader = DataLoader(Subset(ds, indices), batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=detection_collate_fn,
                            generator=torch.Generator().manual_seed(args.seed))
        batch_images, targets = next(iter(loader))
        reference = LLVIPDataset(args.data_root, split, modality="paired", transforms=letterbox)
        ref_rng = PairedRandomHorizontalFlip(pipeline.horizontal_flip.p,
                            generator=torch.Generator().manual_seed(args.seed))
        batch_decisions = []
        for source_index, pair, tgt in zip(indices, batch_images, targets):
            ref_images, ref_tgt = reference[source_index]
            decision = ref_rng.sample_decision()
            batch_decisions.append(decision)
            check_result((ref_images["visible"], ref_images["infrared"]),
                         (pair["visible"], pair["infrared"]), ref_tgt, tgt, decision)
            assert pair["visible"].shape == pair["infrared"].shape == (3, 640, 640)
        box_shapes = [list(t["boxes"].shape) for t in targets]
        print(f"DataLoader {split}: p={pipeline.horizontal_flip.p}; decisions={batch_decisions}; "
              f"RGB/IR=(3,640,640); boxes={box_shapes}; PASS")
        batches.append({"split": split, "p": pipeline.horizontal_flip.p,
                        "decisions": batch_decisions, "boxes_shapes": box_shapes})

    try:
        forced(rgb, ir[:, :-1, :], t)
    except ValueError:
        print("PASS: 拒绝 RGB/IR 尺寸不一致（人为构造的反例）")
    else:
        raise AssertionError("没有拒绝尺寸不一致")
    assert before_hashes == digest(protected)
    print("PASS: Day01、Step1 代码/产物、manifest 与所用源文件 SHA256 前后一致")
    overview = Image.new("RGB", (1280, 1340), "white")
    for i, panel in enumerate(panels):
        overview.paste(panel.resize((640, 670), Image.Resampling.LANCZOS),
                       ((i % 2) * 640, (i // 2) * 670))
    overview.save(output / "overview.jpg", quality=95)
    results = {"seed": args.seed, "samples": records, "random_decisions": decisions,
               "dataloader": batches, "protected_sha256": before_hashes,
               "all_checks_passed": True}
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nALL CHECKS PASSED；结果目录：{output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，停止以避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "terminal_output.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, output)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
