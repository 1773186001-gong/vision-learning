"""Day02 Step 1：用真实 LLVIP 样本验证、记录和可视化 Letterbox。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

# 保持 Day01 原文件不变，也不在 Day01 写入 __pycache__。
sys.dont_write_bytecode = True
DAY01 = Path(__file__).resolve().parents[1] / "day01"
sys.path.insert(0, str(DAY01))

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from llvip_dataset import LLVIPDataset, detection_collate_fn
from torch.utils.data import DataLoader
from paired_letterbox import PairedLetterbox


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_image(tensor, boxes):
    image = F.to_pil_image(tensor)
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(boxes.tolist(), 1):
        draw.rectangle(box, outline=(255, 45, 45), width=2)
        draw.text((box[0] + 2, box[1] + 2), str(index), fill=(255, 220, 0))
    return image


def comparison(before, target, after, target_after, stem, split):
    # 上排保留原图的原生尺寸；下排 640x640 也按原生尺寸展示，不拉伸。
    width, height = before["visible"].shape[-1], before["visible"].shape[-2]
    cell_w = max(width, 640)
    canvas = Image.new("RGB", (2 * cell_w, height + 640 + 80), "white")
    draw = ImageDraw.Draw(canvas)
    for col, modality in enumerate(("visible", "infrared")):
        x = col * cell_w
        draw.text((x + 8, 8), f"{split}/{stem} {modality} BEFORE {width}x{height}", fill="black")
        canvas.paste(draw_image(before[modality], target["boxes"]), (x, 30))
        draw.text((x + 8, height + 40), "AFTER 640x640 | same scale/padding | gray=padding", fill="black")
        canvas.paste(draw_image(after[modality], target_after["boxes"]), (x, height + 70))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，避免覆盖：{output}")
    datasets = {s: LLVIPDataset(args.data_root, s, modality="paired")
                for s in ("train", "val", "test")}
    selection = [(s, next(i for i, r in enumerate(ds.records) if not r.is_negative))
                 for s, ds in datasets.items()]
    selection += [(s, i) for s, ds in datasets.items()
                  for i, r in enumerate(ds.records) if r.is_negative]
    protected = list(DAY01.glob("*.py")) + list(DAY01.glob("*.md"))
    protected.append(args.data_root / "manifest.csv")
    for split, index in selection:
        r = datasets[split].get_record(index)
        protected.extend([r.visible_path, r.infrared_path, r.annotation_path])
    hashes_before = {str(p.resolve()): sha256(p) for p in protected}
    output.mkdir(parents=True, exist_ok=True)
    lines, results, panels = [], [], []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    letterbox = PairedLetterbox(640)
    log("Week08 Day02 Step 1 | 真实 LLVIP 同步 Letterbox")
    log("尺寸均按 W x H；padding 顺序为 left, top, right, bottom")
    for split, index in selection:
        ds = datasets[split]
        record = ds.get_record(index)
        if len({record.visible_path.stem, record.infrared_path.stem,
                record.annotation_path.stem, record.stem}) != 1:
            raise ValueError("RGB/IR/XML 文件名不一致，停止检查")
        images, target = ds[index]
        originals = {k: v.clone() for k, v in target.items()}
        rgb, ir, transformed_target = letterbox(images["visible"], images["infrared"], target)
        after = {"visible": rgb, "infrared": ir}
        height, width = images["visible"].shape[-2:]
        params = letterbox.parameters(height, width)
        left, top, right, bottom = params.padding_ltrb
        expected = target["boxes"] * params.scale + target["boxes"].new_tensor([left, top, left, top])
        torch.testing.assert_close(transformed_target["boxes"], expected)
        restored = (transformed_target["boxes"] - expected.new_tensor([left, top, left, top])) / params.scale
        torch.testing.assert_close(restored, target["boxes"])
        torch.testing.assert_close(transformed_target["area"], target["area"] * params.scale**2)
        for key in target:
            torch.testing.assert_close(target[key], originals[key])
        for key in ("labels", "image_id", "iscrowd"):
            torch.testing.assert_close(transformed_target[key], target[key])
        assert rgb.shape[-2:] == ir.shape[-2:] == (640, 640)
        if top:
            assert torch.all(rgb[:, :top, :] == letterbox.fill)
            assert torch.all(ir[:, :top, :] == letterbox.fill)
        # 同一图像传入两分支时，输出必须逐元素相同，验证两分支使用同一几何操作。
        same_a, same_b, _ = letterbox(images["visible"], images["visible"], target)
        torch.testing.assert_close(same_a, same_b, rtol=0, atol=0)
        panel = comparison(images, target, after, transformed_target, record.stem, split)
        image_name = f"{split}_{record.stem}_comparison.jpg"
        panel.save(output / image_name, quality=95)
        panels.append(panel)
        log(f"\n[{split}/{record.stem}] negative={record.is_negative}")
        log(f"RGB 原始={width}x{height}; IR 原始={images['infrared'].shape[-1]}x{images['infrared'].shape[-2]}")
        log(f"scale={params.scale:.6f}; resized={params.resized_wh}; padding LTRB={params.padding_ltrb}")
        log(f"RGB/IR 输出={rgb.shape[-1]}x{rgb.shape[-2]}; boxes shape={tuple(transformed_target['boxes'].shape)}")
        log(f"原始 xyxy={target['boxes'].tolist()}")
        log(f"变换 xyxy={transformed_target['boxes'].tolist()}")
        log("检查通过：坐标公式、逆变换、area、类别、空框形状、同步几何及输入 target 不变")
        results.append({"split": split, "stem": record.stem,
                        "parameters": asdict(params), "negative": record.is_negative,
                        "boxes_before": target["boxes"].tolist(),
                        "boxes_after": transformed_target["boxes"].tolist(),
                        "boxes_shape_after": list(transformed_target["boxes"].shape),
                        "comparison": image_name})

    # 实际通过 Day01 transforms 接口与 DataLoader 集成，输出固定尺寸。
    integrated = LLVIPDataset(args.data_root, "train", modality="paired", transforms=letterbox)
    batch_images, batch_targets = next(iter(DataLoader(integrated, batch_size=2,
                                    num_workers=0, collate_fn=detection_collate_fn)))
    assert all(img[mode].shape == (3, 640, 640)
               for img in batch_images for mode in ("visible", "infrared"))
    log("\nDataset transforms + DataLoader 集成通过：batch=2，RGB/IR 均为 (3,640,640)")
    # 反例仅用于验证拒绝不一致输入，不作为真实数据检查结果。
    try:
        letterbox(images["visible"], images["infrared"][:, :-1, :], target)
    except ValueError as error:
        log(f"尺寸不一致拒绝测试通过：{error}")
    else:
        raise AssertionError("未拒绝 RGB/IR 不同尺寸")
    hashes_after = {path: sha256(Path(path)) for path in hashes_before}
    assert hashes_before == hashes_after
    log("Day01 代码/文档、manifest 及所用源图/XML 的 SHA256 前后相同")
    log("空间对应关系：同名与尺寸检查通过；像素级配准仍须结合对比图人工核验")
    log(f"完成：{len(results)} 个真实样本（含空标注），结果保存在 {output}")
    # 总览保留每张前后对照的结构；详细坐标以 JSON/终端为准。
    thumbs = [p.resize((960, round(p.height * 960 / p.width)), Image.Resampling.LANCZOS)
              for p in panels]
    overview = Image.new("RGB", (960, sum(p.height for p in thumbs)), "white")
    y = 0
    for thumb in thumbs:
        overview.paste(thumb, (0, y))
        y += thumb.height
    overview.save(output / "overview.jpg", quality=95)
    (output / "results.json").write_text(json.dumps(
        {"samples": results, "protected_file_sha256": hashes_before,
         "checks_passed": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "terminal_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
