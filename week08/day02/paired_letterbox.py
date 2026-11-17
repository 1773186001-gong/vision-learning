"""RGB/IR 同步 Letterbox，可作为 Day01 LLVIPDataset 的 transforms。

输入/输出均为 float32 CHW 图像和像素 xyxy target；不修改输入对象。
"""
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class LetterboxParameters:
    original_wh: tuple[int, int]
    resized_wh: tuple[int, int]
    output_wh: tuple[int, int]
    scale: float
    padding_ltrb: tuple[int, int, int, int]


class PairedLetterbox:
    def __init__(self, size: int = 640, fill: float = 114 / 255):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("size 必须为正整数")
        if not 0 <= fill <= 1:
            raise ValueError("fill 应在 [0,1]")
        self.size = size
        self.fill = fill

    def parameters(self, height: int, width: int) -> LetterboxParameters:
        if min(height, width) <= 0:
            raise ValueError("原始尺寸必须为正数")
        ratio = min(self.size / width, self.size / height)
        new_w = max(1, round(width * ratio))
        new_h = max(1, round(height * ratio))
        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        return LetterboxParameters(
            (width, height), (new_w, new_h), (self.size, self.size), ratio,
            (left, top, self.size - new_w - left, self.size - new_h - top),
        )

    @staticmethod
    def validate_boxes(boxes: Tensor, width: int, height: int) -> None:
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.is_floating_point():
            raise ValueError("boxes 必须为浮点 [N,4]，空标注应为 [0,4]")
        if not torch.isfinite(boxes).all():
            raise ValueError("boxes 包含 NaN 或 Inf")
        if boxes.numel():
            if (boxes[:, 2:] <= boxes[:, :2]).any():
                raise ValueError("边界框宽高非正")
            limits = boxes.new_tensor([width, height, width, height])
            if (boxes < 0).any() or (boxes > limits).any():
                raise ValueError("边界框越界；请检查源标注，不自动裁剪")

    def __call__(self, visible: Tensor, infrared: Tensor, target: dict):
        for name, image in (("RGB", visible), ("IR", infrared)):
            if image.ndim != 3 or image.dtype != torch.float32:
                raise ValueError(f"{name} 应为 float32 CHW")
            if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
                raise ValueError(f"{name} 像素应有限且位于 [0,1]")
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError(
                f"RGB/IR 原始尺寸不一致：{tuple(visible.shape)} / "
                f"{tuple(infrared.shape)}；停止，不能静默强制配准"
            )
        height, width = visible.shape[-2:]
        self.validate_boxes(target["boxes"], width, height)
        params = self.parameters(height, width)
        new_w, new_h = params.resized_wh

        def apply_image(image):
            resized = F.resize(image, [new_h, new_w],
                               interpolation=InterpolationMode.BILINEAR, antialias=True)
            return F.pad(resized, list(params.padding_ltrb), fill=self.fill)

        # 对两种模态复用同一个 params；不分别计算、不随机采样。
        visible_out, infrared_out = apply_image(visible), apply_image(infrared)
        target_out = {key: value.clone() for key, value in target.items()}
        boxes = target_out["boxes"]
        left, top, _, _ = params.padding_ltrb
        boxes *= params.scale
        boxes += boxes.new_tensor([left, top, left, top])
        target_out["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        self.validate_boxes(boxes, self.size, self.size)
        return visible_out, infrared_out, target_out
