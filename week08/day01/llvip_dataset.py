"""LLVIP RGB/IR/VOC XML PyTorch Dataset。

这个 Dataset 直接读取 Week08 Day01 生成的 LLVIP_1000 子集：

- 不需要转换为 YOLO 格式；
- person 的检测类别编号为 1，0 保留给 Faster R-CNN 的背景类；
- 支持 visible、infrared、paired 三种读取模式；
- paired 模式保持 RGB/IR 同名配对，方便后续实现多模态融合；
- 当前版本不做随机数据增强，避免 RGB/IR 几何变换不同步。
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional


Modality = Literal["visible", "infrared", "paired"]
PairedTransform = Callable[
    [Tensor, Tensor, Dict[str, Tensor]],
    Tuple[Tensor, Tensor, Dict[str, Tensor]],
]


@dataclass(frozen=True)
class LLVIPRecord:
    """manifest.csv 中一对 RGB/IR 样本的路径与基本信息。"""

    split: str
    source_split: str
    scene_id: str
    stem: str
    infrared_path: Path
    visible_path: Path
    annotation_path: Path
    person_boxes: int
    is_negative: bool


def _read_manifest(root: Path, split: str) -> List[LLVIPRecord]:
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到子集清单：{manifest_path}")

    records: List[LLVIPRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "split",
            "source_split",
            "scene_id",
            "stem",
            "infrared",
            "visible",
            "annotation_voc",
            "person_boxes",
            "is_negative",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"manifest.csv 缺少字段：{sorted(missing_columns)}")

        for row in reader:
            if row["split"] != split:
                continue
            records.append(
                LLVIPRecord(
                    split=row["split"],
                    source_split=row["source_split"],
                    scene_id=row["scene_id"],
                    stem=row["stem"],
                    infrared_path=root / row["infrared"],
                    visible_path=root / row["visible"],
                    annotation_path=root / row["annotation_voc"],
                    person_boxes=int(row["person_boxes"]),
                    is_negative=bool(int(row["is_negative"])),
                )
            )

    if not records:
        raise ValueError(f"manifest.csv 中没有 split={split!r} 的样本")
    return records


def _parse_voc_annotation(
    annotation_path: Path,
    image_id: int,
    expected_stem: str,
) -> Tuple[Dict[str, Tensor], Tuple[int, int]]:
    """把一个 Pascal VOC XML 转成 TorchVision 检测 target。"""
    try:
        root = ET.parse(annotation_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"XML 无法解析：{annotation_path} ({error})") from error

    xml_filename = root.findtext("filename")
    if xml_filename and Path(xml_filename).stem != expected_stem:
        raise ValueError(
            f"XML filename 与图像不一致：{annotation_path} -> {xml_filename}"
        )

    try:
        width = int(root.findtext("size/width", ""))
        height = int(root.findtext("size/height", ""))
    except ValueError as error:
        raise ValueError(f"XML 图像尺寸非法：{annotation_path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"XML 图像尺寸必须为正数：{annotation_path}")

    boxes: List[List[float]] = []
    for object_index, object_node in enumerate(root.findall("object"), start=1):
        class_name = (object_node.findtext("name") or "").strip()
        if class_name != "person":
            raise ValueError(
                f"{annotation_path} 的目标 {object_index} 类别不是 person："
                f"{class_name!r}"
            )

        box_node = object_node.find("bndbox")
        if box_node is None:
            raise ValueError(f"{annotation_path} 的目标 {object_index} 缺少 bndbox")
        try:
            xmin = float(box_node.findtext("xmin", ""))
            ymin = float(box_node.findtext("ymin", ""))
            xmax = float(box_node.findtext("xmax", ""))
            ymax = float(box_node.findtext("ymax", ""))
        except ValueError as error:
            raise ValueError(
                f"{annotation_path} 的目标 {object_index} 坐标无法解析"
            ) from error

        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            raise ValueError(
                f"{annotation_path} 的目标 {object_index} 越界："
                f"{(xmin, ymin, xmax, ymax)}"
            )
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                f"{annotation_path} 的目标 {object_index} 宽高非正："
                f"{(xmin, ymin, xmax, ymax)}"
            )
        boxes.append([xmin, ymin, xmax, ymax])

    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        )
    else:
        boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        area = torch.empty((0,), dtype=torch.float32)

    object_count = boxes_tensor.shape[0]
    target = {
        "boxes": boxes_tensor,
        # TorchVision 检测器约定：0 是背景，person 从 1 开始。
        "labels": torch.ones((object_count,), dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": area,
        "iscrowd": torch.zeros((object_count,), dtype=torch.int64),
    }
    return target, (width, height)


def _load_visible(path: Path) -> Tuple[Tensor, Tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        size = image.size
        tensor = transform_functional.pil_to_tensor(image).to(torch.float32) / 255.0
    return tensor, size


def _load_infrared(path: Path, channels: int) -> Tuple[Tensor, Tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("L")
        size = image.size
        tensor = transform_functional.pil_to_tensor(image).to(torch.float32) / 255.0
    if channels == 3:
        tensor = tensor.repeat(3, 1, 1)
    return tensor, size


class LLVIPDataset(Dataset):
    """直接读取 RGB、IR 和 VOC XML 的 LLVIP 检测数据集。"""

    VALID_SPLITS = {"train", "val", "test"}
    VALID_MODALITIES = {"visible", "infrared", "paired"}

    def __init__(
        self,
        root: str | Path,
        split: str,
        modality: Modality = "paired",
        infrared_channels: int = 3,
        transforms: Optional[PairedTransform] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split 必须是 {sorted(self.VALID_SPLITS)}，收到 {split!r}")
        if modality not in self.VALID_MODALITIES:
            raise ValueError(
                f"modality 必须是 {sorted(self.VALID_MODALITIES)}，收到 {modality!r}"
            )
        if infrared_channels not in {1, 3}:
            raise ValueError("infrared_channels 只能是 1 或 3")

        self.split = split
        self.modality = modality
        self.infrared_channels = infrared_channels
        self.transforms = transforms
        self.records = _read_manifest(self.root, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor | Dict[str, Tensor], Dict[str, Tensor]]:
        record = self.records[index]
        for path in (
            record.visible_path,
            record.infrared_path,
            record.annotation_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"样本文件不存在：{path}")

        visible, visible_size = _load_visible(record.visible_path)
        infrared, infrared_size = _load_infrared(
            record.infrared_path, self.infrared_channels
        )
        target, xml_size = _parse_voc_annotation(
            record.annotation_path,
            image_id=index,
            expected_stem=record.stem,
        )

        if visible_size != infrared_size or visible_size != xml_size:
            raise ValueError(
                f"{record.stem} 尺寸不一致：RGB={visible_size}，"
                f"IR={infrared_size}，XML={xml_size}"
            )
        if target["boxes"].shape[0] != record.person_boxes:
            raise ValueError(
                f"{record.stem} 的 XML 目标数与 manifest 不一致："
                f"{target['boxes'].shape[0]} != {record.person_boxes}"
            )

        if self.transforms is not None:
            visible, infrared, target = self.transforms(visible, infrared, target)

        if self.modality == "visible":
            image: Tensor | Dict[str, Tensor] = visible
        elif self.modality == "infrared":
            image = infrared
        else:
            image = {"visible": visible, "infrared": infrared}
        return image, target

    def get_record(self, index: int) -> LLVIPRecord:
        """返回样本元数据，不读取图像。"""
        return self.records[index]


def detection_collate_fn(
    batch: List[Tuple[Tensor | Dict[str, Tensor], Dict[str, Tensor]]],
) -> Tuple[List[Tensor | Dict[str, Tensor]], List[Dict[str, Tensor]]]:
    """检测任务不能直接 stack 不同目标数的 boxes，因此保留为 list。"""
    images, targets = zip(*batch)
    return list(images), list(targets)
