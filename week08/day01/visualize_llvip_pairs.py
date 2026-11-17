"""生成 LLVIP RGB/IR 并排边界框可视化。

从 train、val、test 中按场景选取代表样本，并额外显示合法负样本。
脚本只读取 LLVIP_1000，不修改图像、XML 或 manifest。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms import functional as transform_functional

from llvip_dataset import LLVIPDataset


DISPLAY_SIZE = (640, 512)
HEADER_HEIGHT = 40
BOX_COLOR = (255, 48, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 LLVIP RGB/IR 配对与边界框。")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="LLVIP_1000 子集根目录，例如 .\\data\\LLVIP_1000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="可视化输出目录。",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=3,
        help="每个 split 选择多少个不同场景（默认：3）。",
    )
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> List[int]:
    if length <= 0 or count <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def choose_scene_diverse_indices(
    dataset: LLVIPDataset, samples_per_split: int
) -> List[int]:
    scene_to_indices: Dict[str, List[int]] = {}
    for index, record in enumerate(dataset.records):
        scene_to_indices.setdefault(record.scene_id, []).append(index)

    scenes = sorted(scene_to_indices)
    scene_positions = evenly_spaced_indices(len(scenes), samples_per_split)
    selected: List[int] = []
    for position in scene_positions:
        indices = scene_to_indices[scenes[position]]
        selected.append(indices[len(indices) // 2])
    return selected


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
) -> Image.Image:
    original_width, original_height = image.size
    resized = image.resize(DISPLAY_SIZE, Image.Resampling.BILINEAR)
    scale_x = DISPLAY_SIZE[0] / original_width
    scale_y = DISPLAY_SIZE[1] / original_height
    draw = ImageDraw.Draw(resized)

    for box_index, (xmin, ymin, xmax, ymax) in enumerate(boxes, start=1):
        scaled_box = (
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y,
        )
        draw.rectangle(scaled_box, outline=BOX_COLOR, width=3)
        label_position = (scaled_box[0] + 2, max(2, scaled_box[1] - 12))
        draw.text(label_position, str(box_index), fill=BOX_COLOR)
    return resized


def make_pair_panel(
    visible_tensor: Tensor,
    infrared_tensor: Tensor,
    boxes: Sequence[Sequence[float]],
    split: str,
    scene_id: str,
    stem: str,
) -> Image.Image:
    visible = transform_functional.to_pil_image(visible_tensor)
    infrared = transform_functional.to_pil_image(infrared_tensor)
    visible = draw_boxes(visible, boxes)
    infrared = draw_boxes(infrared, boxes)

    panel_width = DISPLAY_SIZE[0] * 2
    panel_height = DISPLAY_SIZE[1] + HEADER_HEIGHT
    panel = Image.new("RGB", (panel_width, panel_height), "white")
    panel.paste(visible, (0, HEADER_HEIGHT))
    panel.paste(infrared, (DISPLAY_SIZE[0], HEADER_HEIGHT))

    draw = ImageDraw.Draw(panel)
    box_count = len(boxes)
    draw.text(
        (10, 5),
        f"RGB | split={split} scene={scene_id} id={stem} boxes={box_count}",
        fill="black",
    )
    draw.text(
        (DISPLAY_SIZE[0] + 10, 5),
        f"IR | split={split} scene={scene_id} id={stem} boxes={box_count}",
        fill="black",
    )
    if box_count == 0:
        draw.text((10, 22), "NEGATIVE SAMPLE: no person boxes", fill=(0, 90, 200))
    return panel


def make_overview(panels: Sequence[Image.Image], columns: int = 2) -> Image.Image:
    thumbnail_size = (640, 276)
    rows = (len(panels) + columns - 1) // columns
    overview = Image.new(
        "RGB",
        (thumbnail_size[0] * columns, thumbnail_size[1] * rows),
        (230, 230, 230),
    )
    for index, panel in enumerate(panels):
        thumbnail = panel.resize(thumbnail_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumbnail_size[0]
        y = (index // columns) * thumbnail_size[1]
        overview.paste(thumbnail, (x, y))
    return overview


def main() -> None:
    args = parse_args()
    if args.samples_per_split <= 0:
        raise ValueError("samples-per-split 必须为正数")

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        split: LLVIPDataset(data_root, split=split, modality="paired")
        for split in ("train", "val", "test")
    }
    selections: List[Tuple[str, int]] = []
    for split, dataset in datasets.items():
        selections.extend(
            (split, index)
            for index in choose_scene_diverse_indices(dataset, args.samples_per_split)
        )

    # 子集中只有一个合法负样本；若尚未选到，则额外加入可视化。
    selected_keys = set(selections)
    for split, dataset in datasets.items():
        for index, record in enumerate(dataset.records):
            if record.is_negative and (split, index) not in selected_keys:
                selections.append((split, index))

    panels: List[Image.Image] = []
    manifest_rows: List[Dict[str, object]] = []
    for split, index in selections:
        dataset = datasets[split]
        image, target = dataset[index]
        if not isinstance(image, dict):
            raise TypeError("paired 模式应返回 visible/infrared 字典")
        record = dataset.get_record(index)
        boxes = target["boxes"].tolist()
        panel = make_pair_panel(
            visible_tensor=image["visible"],
            infrared_tensor=image["infrared"],
            boxes=boxes,
            split=split,
            scene_id=record.scene_id,
            stem=record.stem,
        )
        filename = f"{split}_scene{record.scene_id}_{record.stem}.jpg"
        output_path = output_dir / filename
        panel.save(output_path, quality=92, subsampling=0)
        panels.append(panel)
        manifest_rows.append(
            {
                "split": split,
                "scene_id": record.scene_id,
                "stem": record.stem,
                "person_boxes": len(boxes),
                "is_negative": int(record.is_negative),
                "visualization": filename,
            }
        )
        print(
            f"已生成：{filename} | scene={record.scene_id} | boxes={len(boxes)}"
        )

    overview_path = output_dir / "overview.jpg"
    make_overview(panels).save(overview_path, quality=92, subsampling=0)

    with (output_dir / "visualization_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n共生成 {len(panels)} 个 RGB/IR 配对可视化。")
    print(f"总览图：{overview_path}")
    print("请人工检查两种模态中的同一编号框是否覆盖同一名行人。")


if __name__ == "__main__":
    main()
