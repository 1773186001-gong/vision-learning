"""小规模读取 LLVIP Dataset/DataLoader，验证张量和检测标注。

只读取指定的少量 batch，不训练模型、不使用 GPU、不修改数据。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from llvip_dataset import LLVIPDataset, detection_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 LLVIP PyTorch DataLoader。")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="LLVIP_1000 子集根目录，例如 .\\data\\LLVIP_1000",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )
    parser.add_argument(
        "--modality",
        choices=("visible", "infrared", "paired"),
        default="paired",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--infrared-channels", type=int, choices=(1, 3), default=3)
    return parser.parse_args()


def validate_target(target: Dict[str, Tensor], image_size: tuple[int, int]) -> None:
    boxes = target["boxes"]
    labels = target["labels"]
    width, height = image_size

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes 应为 [N, 4]，实际为 {tuple(boxes.shape)}")
    if labels.shape != (boxes.shape[0],):
        raise ValueError("labels 数量与 boxes 不一致")
    if labels.numel() and not torch.all(labels == 1):
        raise ValueError("person 标签应全部为 1")
    if boxes.numel():
        if not torch.all(boxes[:, 0] < boxes[:, 2]):
            raise ValueError("存在 xmin >= xmax")
        if not torch.all(boxes[:, 1] < boxes[:, 3]):
            raise ValueError("存在 ymin >= ymax")
        if torch.any(boxes < 0):
            raise ValueError("存在负数边界框坐标")
        if torch.any(boxes[:, [0, 2]] > width):
            raise ValueError("存在超出图像宽度的边界框")
        if torch.any(boxes[:, [1, 3]] > height):
            raise ValueError("存在超出图像高度的边界框")


def tensor_summary(tensor: Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]"
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_batches <= 0 or args.num_workers < 0:
        raise ValueError("batch-size/num-batches 必须为正数，num-workers 不能为负数")

    datasets = {
        split: LLVIPDataset(
            root=args.data_root,
            split=split,
            modality=args.modality,
            infrared_channels=args.infrared_channels,
        )
        for split in ("train", "val", "test")
    }
    dataset = datasets[args.split]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    print("========== LLVIP DataLoader Check ==========")
    print(f"root: {dataset.root}")
    print(
        f"dataset sizes: train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    print(
        f"checking: split={args.split}, modality={args.modality}, "
        f"batch_size={args.batch_size}, infrared_channels={args.infrared_channels}"
    )

    checked_samples = 0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if batch_index > args.num_batches:
            break
        print(f"\n[batch {batch_index}] samples={len(images)}")
        for sample_index, (image, target) in enumerate(zip(images, targets)):
            dataset_index = checked_samples + sample_index
            record = dataset.get_record(dataset_index)

            if args.modality == "paired":
                if not isinstance(image, dict):
                    raise TypeError("paired 模式应返回 visible/infrared 字典")
                visible = image["visible"]
                infrared = image["infrared"]
                if visible.shape[-2:] != infrared.shape[-2:]:
                    raise ValueError("RGB/IR 张量空间尺寸不一致")
                image_size = (visible.shape[-1], visible.shape[-2])
                image_text = (
                    f"RGB({tensor_summary(visible)}), "
                    f"IR({tensor_summary(infrared)})"
                )
            else:
                if not isinstance(image, Tensor):
                    raise TypeError("单模态模式应返回 Tensor")
                image_size = (image.shape[-1], image.shape[-2])
                image_text = tensor_summary(image)

            validate_target(target, image_size)
            print(
                f"  {record.stem} scene={record.scene_id}: {image_text}, "
                f"boxes={tuple(target['boxes'].shape)}"
            )
        checked_samples += len(images)

    negative_indices = [
        index for index, record in enumerate(dataset.records) if record.is_negative
    ]
    if negative_indices:
        negative_index = negative_indices[0]
        negative_image, negative_target = dataset[negative_index]
        if args.modality == "paired":
            assert isinstance(negative_image, dict)
            negative_size = (
                negative_image["visible"].shape[-1],
                negative_image["visible"].shape[-2],
            )
        else:
            assert isinstance(negative_image, Tensor)
            negative_size = (negative_image.shape[-1], negative_image.shape[-2])
        validate_target(negative_target, negative_size)
        if negative_target["boxes"].shape != (0, 4):
            raise ValueError("负样本 boxes 应为形状 (0, 4) 的空张量")
        negative_record = dataset.get_record(negative_index)
        print(
            f"负样本检查通过：{negative_record.stem}，"
            f"boxes={tuple(negative_target['boxes'].shape)}"
        )

    print(f"\n检查通过：成功读取并验证 {checked_samples} 个样本。")
    print("本脚本未训练模型，也未修改数据。")


if __name__ == "__main__":
    main()
