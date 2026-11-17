"""Step 2：复用 Letterbox，在其后同步随机水平翻转 RGB/IR/xyxy 框。"""
import math

import torch

from paired_letterbox import PairedLetterbox


class PairedRandomHorizontalFlip:
    """兼容 LLVIPDataset.transforms 的三参数接口。

    默认使用 PyTorch RNG；实验可传 CPU Generator 固定序列。
    当前实验 num_workers=0；多 worker 时应为各 worker 配置独立种子。
    """

    def __init__(self, p=0.5, generator=None):
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("翻转概率 p 必须在 [0,1]")
        self.p = float(p)
        self.generator = generator

    def sample_decision(self):
        if self.p == 0:
            return False
        if self.p == 1:
            return True
        # 每个样本对只抽一次，绝不能为 RGB 和 IR 各抽一次。
        return bool(torch.rand((), generator=self.generator).item() < self.p)

    def apply(self, visible, infrared, target, *, flip):
        """给定一个共享决策；用于显式复现及强制翻转/不翻转验证。"""
        if not isinstance(flip, bool):
            raise ValueError("flip 必须为 bool")
        for name, image in (("RGB", visible), ("IR", infrared)):
            if image.ndim != 3 or image.dtype != torch.float32:
                raise ValueError(f"{name} 必须是 float32 CHW")
            if not torch.isfinite(image).all():
                raise ValueError(f"{name} 含非有限像素")
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError("RGB/IR 尺寸不一致，不能进行同步翻转")
        height, width = visible.shape[-2:]
        if min(height, width) <= 0:
            raise ValueError("图像尺寸必须为正")
        boxes = target["boxes"]
        if boxes.dtype != torch.float32:
            raise ValueError("boxes 必须为 float32")
        PairedLetterbox.validate_boxes(boxes, width, height)
        # 即使不翻转也返回独立副本，避免后续操作通过共享存储改写输入。
        result = {key: value.clone() for key, value in target.items()}
        if flip:
            visible_out = torch.flip(visible, dims=[-1])
            infrared_out = torch.flip(infrared, dims=[-1])
            # 两列都从原 boxes 取值，不能先覆盖 xmin 再用它计算 xmax。
            result["boxes"][:, 0] = width - boxes[:, 2]
            result["boxes"][:, 2] = width - boxes[:, 0]
        else:
            visible_out, infrared_out = visible.clone(), infrared.clone()
        new_boxes = result["boxes"]
        result["area"] = ((new_boxes[:, 2] - new_boxes[:, 0])
                          * (new_boxes[:, 3] - new_boxes[:, 1]))
        PairedLetterbox.validate_boxes(new_boxes, width, height)
        return visible_out, infrared_out, result

    def __call__(self, visible, infrared, target):
        flip = self.sample_decision()
        return self.apply(visible, infrared, target, flip=flip)


class PairedLetterboxAndFlip:
    """训练 p=0.5；验证/测试 p=0，始终先执行既有 Letterbox。"""

    def __init__(self, split, train_flip_p=0.5, generator=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split 必须为 train/val/test")
        self.letterbox = PairedLetterbox(size=640)
        self.horizontal_flip = PairedRandomHorizontalFlip(
            p=train_flip_p if split == "train" else 0.0,
            generator=generator,
        )

    def __call__(self, visible, infrared, target):
        visible, infrared, target = self.letterbox(visible, infrared, target)
        return self.horizontal_flip(visible, infrared, target)
