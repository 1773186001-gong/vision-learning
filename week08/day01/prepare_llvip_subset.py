"""从完整 LLVIP 中生成场景隔离的 1,000 对实验子集。

设计原则：
1. 原始 LLVIP 始终只读；
2. RGB、IR、VOC XML 作为不可拆分的一个样本对；
3. train/val/test 使用互不重叠的场景前缀；
4. 每个场景内部按文件编号均匀抽帧，避免大量相邻帧；
5. 自动排除缺失、无法解析或含非法边界框的样本；
6. 默认仅预览，只有显式传入 --execute 才复制文件。
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# 真实数据检查显示：前两位编号形成 26 个互斥分组，与官方 26 个场景一致。
# 官方 train 中的 4 个完整场景被划给 val；官方 test 只用于最终 test。
SUBSET_SCENES = {
    "train": (
        "01", "02", "04", "05", "06", "07", "09", "10",
        "11", "12", "14", "15", "16", "18", "25",
    ),
    "val": ("03", "08", "13", "17"),
    "test": ("19", "20", "21", "22", "23", "24", "26"),
}
SOURCE_SPLITS = {"train": "train", "val": "train", "test": "test"}
TARGET_COUNTS = {"train": 700, "val": 150, "test": 150}


@dataclass(frozen=True)
class Candidate:
    split: str
    scene_id: str
    stem: str
    infrared_path: Path
    visible_path: Path
    annotation_path: Path
    person_boxes: int
    is_negative: bool


@dataclass(frozen=True)
class RejectedSample:
    source_split: str
    scene_id: str
    stem: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成场景隔离的 LLVIP 1,000 对子集。默认只预览。"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="完整 LLVIP 根目录，例如 .\\LLVIP",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="新子集输出目录，例如 .\\data\\LLVIP_1000",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际复制文件；不传此参数时仅输出选择预览。",
    )
    return parser.parse_args()


def collect_files(folder: Path, suffixes: Iterable[str]) -> Dict[str, Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    files: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            if path.stem in files:
                raise ValueError(f"同一目录出现重复 stem：{files[path.stem]} 和 {path}")
            files[path.stem] = path
    return files


def scene_id_from_stem(stem: str) -> str:
    if len(stem) != 6 or not stem.isdigit():
        raise ValueError(f"文件名不是预期的 6 位数字：{stem}")
    return stem[:2]


def parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def validate_annotation(path: Path) -> Tuple[bool, int, str]:
    """检查类别、尺寸和边界框；空标注是合法负样本。"""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        return False, 0, f"XML 无法解析：{error}"

    width = parse_int(root.findtext("size/width"))
    height = parse_int(root.findtext("size/height"))
    if width is None or height is None or width <= 0 or height <= 0:
        return False, 0, f"图像尺寸非法：{width}x{height}"

    objects = root.findall("object")
    person_boxes = 0
    for object_index, object_node in enumerate(objects, start=1):
        class_name = (object_node.findtext("name") or "").strip()
        if class_name != "person":
            return False, person_boxes, f"目标 {object_index} 类别不是 person：{class_name!r}"

        box = object_node.find("bndbox")
        if box is None:
            return False, person_boxes, f"目标 {object_index} 缺少 bndbox"
        xmin = parse_int(box.findtext("xmin"))
        ymin = parse_int(box.findtext("ymin"))
        xmax = parse_int(box.findtext("xmax"))
        ymax = parse_int(box.findtext("ymax"))
        coordinates = (xmin, ymin, xmax, ymax)
        if None in coordinates:
            return False, person_boxes, f"目标 {object_index} 坐标无法解析：{coordinates}"

        assert xmin is not None and ymin is not None
        assert xmax is not None and ymax is not None
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            return False, person_boxes, f"目标 {object_index} 越界：{coordinates}"
        if xmin >= xmax or ymin >= ymax:
            return False, person_boxes, f"目标 {object_index} 宽高非正：{coordinates}"
        person_boxes += 1

    return True, person_boxes, ""


def discover_candidates(
    source_root: Path,
) -> Tuple[Dict[str, List[Candidate]], List[RejectedSample]]:
    expected_dirs = (
        source_root / "infrared" / "train",
        source_root / "infrared" / "test",
        source_root / "visible" / "train",
        source_root / "visible" / "test",
        source_root / "Annotations",
    )
    missing = [str(path) for path in expected_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("缺少 LLVIP 目录：" + ", ".join(missing))

    xml_files = collect_files(source_root / "Annotations", {".xml"})
    candidates: Dict[str, List[Candidate]] = collections.defaultdict(list)
    rejected: List[RejectedSample] = []

    for source_split in ("train", "test"):
        infrared = collect_files(
            source_root / "infrared" / source_split, IMAGE_SUFFIXES
        )
        visible = collect_files(
            source_root / "visible" / source_split, IMAGE_SUFFIXES
        )
        all_stems = sorted(set(infrared) | set(visible))
        for stem in all_stems:
            scene_id = scene_id_from_stem(stem)
            if stem not in infrared or stem not in visible:
                rejected.append(
                    RejectedSample(source_split, scene_id, stem, "RGB/IR 配对不完整")
                )
                continue
            annotation_path = xml_files.get(stem)
            if annotation_path is None:
                rejected.append(
                    RejectedSample(source_split, scene_id, stem, "缺少 VOC XML")
                )
                continue
            valid, person_boxes, reason = validate_annotation(annotation_path)
            if not valid:
                rejected.append(RejectedSample(source_split, scene_id, stem, reason))
                continue
            candidates[scene_id].append(
                Candidate(
                    split="",
                    scene_id=scene_id,
                    stem=stem,
                    infrared_path=infrared[stem],
                    visible_path=visible[stem],
                    annotation_path=annotation_path,
                    person_boxes=person_boxes,
                    is_negative=person_boxes == 0,
                )
            )

    for scene_candidates in candidates.values():
        scene_candidates.sort(key=lambda item: int(item.stem))
    return dict(candidates), rejected


def allocate_counts(total: int, scenes: Sequence[str]) -> Dict[str, int]:
    base, remainder = divmod(total, len(scenes))
    return {
        scene_id: base + (1 if index < remainder else 0)
        for index, scene_id in enumerate(scenes)
    }


def evenly_spaced(items: Sequence[Candidate], count: int) -> List[Candidate]:
    if len(items) < count:
        raise ValueError(f"候选样本只有 {len(items)} 个，无法抽取 {count} 个")
    if count == 1:
        return [items[len(items) // 2]]

    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("均匀抽样产生了重复索引")
    return [items[index] for index in indices]


def select_subset(candidates: Dict[str, List[Candidate]]) -> Dict[str, List[Candidate]]:
    expected_scenes = set().union(*[set(value) for value in SUBSET_SCENES.values()])
    observed_scenes = set(candidates)
    if observed_scenes != expected_scenes:
        missing = sorted(expected_scenes - observed_scenes)
        extra = sorted(observed_scenes - expected_scenes)
        raise ValueError(f"场景集合不符合预期；缺少={missing}，额外={extra}")

    selected: Dict[str, List[Candidate]] = {}
    for split, scenes in SUBSET_SCENES.items():
        allocation = allocate_counts(TARGET_COUNTS[split], scenes)
        split_samples: List[Candidate] = []
        for scene_id in scenes:
            sampled = evenly_spaced(candidates[scene_id], allocation[scene_id])
            split_samples.extend(
                Candidate(
                    split=split,
                    scene_id=item.scene_id,
                    stem=item.stem,
                    infrared_path=item.infrared_path,
                    visible_path=item.visible_path,
                    annotation_path=item.annotation_path,
                    person_boxes=item.person_boxes,
                    is_negative=item.is_negative,
                )
                for item in sampled
            )
        selected[split] = sorted(split_samples, key=lambda item: int(item.stem))
    return selected


def validate_split_isolation(selected: Dict[str, List[Candidate]]) -> None:
    scene_sets = {
        split: {sample.scene_id for sample in samples}
        for split, samples in selected.items()
    }
    split_names = list(scene_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = scene_sets[left] & scene_sets[right]
            if overlap:
                raise RuntimeError(f"{left}/{right} 场景泄漏：{sorted(overlap)}")

    stems: Dict[str, str] = {}
    for split, samples in selected.items():
        for sample in samples:
            previous_split = stems.setdefault(sample.stem, split)
            if previous_split != split:
                raise RuntimeError(
                    f"样本 {sample.stem} 同时出现在 {previous_split}/{split}"
                )


def print_plan(
    selected: Dict[str, List[Candidate]], rejected: Sequence[RejectedSample]
) -> None:
    print("LLVIP 1,000 对子集计划")
    print("-" * 72)
    for split in ("train", "val", "test"):
        samples = selected[split]
        scene_counter = collections.Counter(item.scene_id for item in samples)
        object_count = sum(item.person_boxes for item in samples)
        negative_count = sum(item.is_negative for item in samples)
        scene_text = ", ".join(
            f"{scene}:{count}" for scene, count in sorted(scene_counter.items())
        )
        print(
            f"{split:5s}: {len(samples):4d} 对，{object_count:4d} 个 person 框，"
            f"{negative_count} 个负样本"
        )
        print(f"       场景分配：{scene_text}")
    print(f"排除的原始异常样本：{len(rejected)}")
    for item in rejected[:10]:
        print(f"  - {item.stem}.xml: {item.reason}")


def copy_subset(
    selected: Dict[str, List[Candidate]],
    rejected: Sequence[RejectedSample],
    source_root: Path,
    output_root: Path,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖已停止：{output_root}")

    staging_root = output_root.with_name(output_root.name + ".building")
    if staging_root.exists():
        raise FileExistsError(f"临时构建目录已存在，请先人工检查：{staging_root}")

    staging_root.mkdir(parents=True)
    manifest_rows: List[Dict[str, object]] = []
    copied_bytes = 0

    try:
        for split in ("train", "val", "test"):
            infrared_dir = staging_root / "images" / split / "infrared"
            visible_dir = staging_root / "images" / split / "visible"
            annotation_dir = staging_root / "annotations_voc" / split
            infrared_dir.mkdir(parents=True)
            visible_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)

            for sample in selected[split]:
                destinations = (
                    (sample.infrared_path, infrared_dir / sample.infrared_path.name),
                    (sample.visible_path, visible_dir / sample.visible_path.name),
                    (sample.annotation_path, annotation_dir / sample.annotation_path.name),
                )
                for source, destination in destinations:
                    shutil.copy2(source, destination)
                    if destination.stat().st_size != source.stat().st_size:
                        raise OSError(f"复制后文件大小不一致：{source} -> {destination}")
                    copied_bytes += destination.stat().st_size

                manifest_rows.append(
                    {
                        "split": split,
                        "source_split": SOURCE_SPLITS[split],
                        "scene_id": sample.scene_id,
                        "stem": sample.stem,
                        "infrared": (
                            Path("images") / split / "infrared" / sample.infrared_path.name
                        ).as_posix(),
                        "visible": (
                            Path("images") / split / "visible" / sample.visible_path.name
                        ).as_posix(),
                        "annotation_voc": (
                            Path("annotations_voc") / split / sample.annotation_path.name
                        ).as_posix(),
                        "person_boxes": sample.person_boxes,
                        "is_negative": int(sample.is_negative),
                    }
                )

        with (staging_root / "manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

        summary = {
            "source_root": str(source_root),
            "total_pairs": sum(len(items) for items in selected.values()),
            "split_counts": {key: len(value) for key, value in selected.items()},
            "scene_assignments": {
                key: list(value) for key, value in SUBSET_SCENES.items()
            },
            "person_boxes": {
                key: sum(item.person_boxes for item in value)
                for key, value in selected.items()
            },
            "negative_samples": {
                key: sum(item.is_negative for item in value)
                for key, value in selected.items()
            },
            "excluded_invalid_samples": [asdict(item) for item in rejected],
            "copied_bytes": copied_bytes,
            "sampling": "按场景分组，并在每个场景内按编号均匀抽取",
        }
        (staging_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 最终计数检查。
        for split, expected_count in TARGET_COUNTS.items():
            checks = (
                staging_root / "images" / split / "infrared",
                staging_root / "images" / split / "visible",
                staging_root / "annotations_voc" / split,
            )
            for folder in checks:
                actual_count = sum(1 for path in folder.iterdir() if path.is_file())
                if actual_count != expected_count:
                    raise RuntimeError(
                        f"复制后计数错误：{folder}={actual_count}，预期={expected_count}"
                    )

        staging_root.rename(output_root)
    except Exception:
        print(f"构建未完成，临时目录保留用于检查：{staging_root}", file=sys.stderr)
        raise

    print(f"子集已创建：{output_root}")
    print(f"复制数据量：{copied_bytes / (1024 ** 2):.1f} MiB")
    print("已生成 manifest.csv 和 summary.json。")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    try:
        candidates, rejected = discover_candidates(source_root)
        selected = select_subset(candidates)
        validate_split_isolation(selected)
        print_plan(selected, rejected)
        if args.execute:
            copy_subset(selected, rejected, source_root, output_root)
        else:
            print("\n当前为预览模式，没有复制文件。确认后添加 --execute。")
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        print(f"[失败] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
