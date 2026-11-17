"""只读检查下载并解压后的 LLVIP 原始数据集。

本脚本不下载数据、不修改图像或 XML 标注，也不生成 YOLO 标签文件。
它仅使用 Python 标准库，适合在完成官方数据集下载和解压后运行。
"""

from __future__ import annotations

import argparse
import bisect
import collections
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
OFFICIAL_PAIR_COUNTS = {"train": 12025, "test": 3463}


@dataclass
class PairRecord:
    """一对同名的红外/可见光图像及其已读取的尺寸。"""

    split: str
    stem: str
    infrared_path: Path
    visible_path: Path
    infrared_size: Optional[Tuple[int, int]]
    visible_size: Optional[Tuple[int, int]]


@dataclass
class AnnotationReport:
    """VOC XML 检查结果汇总。"""

    xml_count: int = 0
    missing_xml: int = 0
    parse_errors: int = 0
    empty_annotations: int = 0
    objects: int = 0
    invalid_boxes: int = 0
    xml_filename_mismatches: int = 0
    xml_size_mismatches: int = 0
    categories: collections.Counter = None  # type: ignore[assignment]
    examples: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.categories = collections.Counter()
        self.examples = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读检查 LLVIP 的目录、配对、尺寸和 VOC XML 标注。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="LLVIP 解压后的根目录，例如 .\\data\\LLVIP",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="每类问题在控制台最多显示多少个示例（默认：5）。",
    )
    parser.add_argument(
        "--adjacent-gap",
        type=int,
        default=5,
        help="检查 train/test 数字文件名相近程度的阈值（默认：5）。",
    )
    return parser.parse_args()


def heading(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_examples(title: str, values: Sequence[str], limit: int) -> None:
    if not values:
        return
    print(f"{title}（显示 {min(len(values), limit)}/{len(values)}）：")
    for value in values[:limit]:
        print(f"  - {value}")


def collect_image_index(folder: Path) -> Tuple[Dict[str, Path], List[str]]:
    """读取一个官方图像目录，并报告同 stem 的重复图像。"""
    index: Dict[str, Path] = {}
    duplicates: List[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in index:
            duplicates.append(f"{path.stem}: {index[path.stem].name}, {path.name}")
        else:
            index[path.stem] = path
    return index, duplicates


def collect_xml_index(folder: Path) -> Tuple[Dict[str, Path], List[str]]:
    """读取官方平铺的 Annotations 目录。"""
    index: Dict[str, Path] = {}
    duplicates: List[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".xml":
            continue
        if path.stem in index:
            duplicates.append(f"{path.stem}: {index[path.stem].name}, {path.name}")
        else:
            index[path.stem] = path
    return index, duplicates


def read_image_size(path: Path) -> Tuple[int, int]:
    """用标准库读取 PNG/JPEG/BMP 尺寸，避免新增 Pillow 依赖。"""
    with path.open("rb") as image_file:
        header = image_file.read(24)

        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(
                ">II", header[16:24]
            )
            return width, height

        if header.startswith(b"BM") and len(header) >= 26:
            width, height = struct.unpack("<ii", header[18:26])
            return abs(width), abs(height)

        if not header.startswith(b"\xff\xd8"):
            raise ValueError("不支持的图像格式或文件头损坏")

        image_file.seek(2)
        while True:
            marker_prefix = image_file.read(1)
            while marker_prefix == b"\xff":
                marker_prefix = image_file.read(1)
            if not marker_prefix:
                break

            marker = marker_prefix[0]
            # 无长度字段的 JPEG 标记。
            if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue

            length_bytes = image_file.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                break
            segment = image_file.read(segment_length - 2)

            # Start Of Frame（排除 Huffman/算术编码相关标记）。
            if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
                if len(segment) < 5:
                    break
                height, width = struct.unpack(">HH", segment[1:5])
                return width, height

    raise ValueError("未在 JPEG 中找到有效的尺寸信息")


def safe_image_size(path: Path, errors: List[str]) -> Optional[Tuple[int, int]]:
    try:
        return read_image_size(path)
    except (OSError, ValueError, struct.error) as error:
        errors.append(f"{path}: {error}")
        return None


def parse_integer(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def add_example(report: AnnotationReport, text: str, limit: int) -> None:
    if len(report.examples) < limit:
        report.examples.append(text)


def inspect_one_annotation(
    xml_path: Path,
    record: PairRecord,
    report: AnnotationReport,
    example_limit: int,
) -> None:
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as error:
        report.parse_errors += 1
        add_example(report, f"XML 无法解析：{xml_path.name} ({error})", example_limit)
        return

    xml_filename = root.findtext("filename")
    if xml_filename and Path(xml_filename).stem != record.stem:
        report.xml_filename_mismatches += 1
        add_example(
            report,
            f"XML 文件名不匹配：{xml_path.name} 中为 {xml_filename}",
            example_limit,
        )

    width = parse_integer(root.findtext("size/width"))
    height = parse_integer(root.findtext("size/height"))
    if record.infrared_size and (width, height) != record.infrared_size:
        report.xml_size_mismatches += 1
        add_example(
            report,
            f"XML 尺寸不匹配：{xml_path.name} 中为 {(width, height)}，"
            f"红外图为 {record.infrared_size}",
            example_limit,
        )

    objects = root.findall("object")
    if not objects:
        report.empty_annotations += 1

    for object_node in objects:
        class_name = (object_node.findtext("name") or "").strip()
        report.categories[class_name or "<缺少类别>"] += 1
        report.objects += 1

        box = object_node.find("bndbox")
        coordinates = (
            parse_integer(box.findtext("xmin")) if box is not None else None,
            parse_integer(box.findtext("ymin")) if box is not None else None,
            parse_integer(box.findtext("xmax")) if box is not None else None,
            parse_integer(box.findtext("ymax")) if box is not None else None,
        )
        xmin, ymin, xmax, ymax = coordinates
        invalid = (
            None in coordinates
            or xmin is None
            or ymin is None
            or xmax is None
            or ymax is None
            or xmin >= xmax
            or ymin >= ymax
        )

        if not invalid and record.infrared_size:
            image_width, image_height = record.infrared_size
            invalid = (
                xmin < 0
                or ymin < 0
                or xmax > image_width
                or ymax > image_height
            )

        if invalid:
            report.invalid_boxes += 1
            add_example(
                report,
                f"非法框：{xml_path.name}，类别={class_name!r}，坐标={coordinates}",
                example_limit,
            )


def inspect_annotations(
    records: Sequence[PairRecord],
    xml_index: Dict[str, Path],
    example_limit: int,
) -> AnnotationReport:
    report = AnnotationReport(xml_count=len(xml_index))
    for record in records:
        xml_path = xml_index.get(record.stem)
        if xml_path is None:
            report.missing_xml += 1
            add_example(report, f"缺少 XML：{record.split}/{record.stem}", example_limit)
            continue
        inspect_one_annotation(xml_path, record, report, example_limit)
    return report


def numeric_stem(stem: str) -> Optional[int]:
    return int(stem) if stem.isdigit() else None


def continuous_runs(stems: Iterable[str]) -> Tuple[int, int, List[Tuple[int, int]]]:
    """返回纯数字编号总数、相邻编号链接数和最长的连续编号区间。"""
    numbers = sorted(n for stem in stems if (n := numeric_stem(stem)) is not None)
    if not numbers:
        return 0, 0, []

    adjacent_links = 0
    runs: List[Tuple[int, int]] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            adjacent_links += 1
        else:
            runs.append((start, previous))
            start = number
        previous = number
    runs.append((start, previous))
    longest = sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:3]
    return len(numbers), adjacent_links, longest


def near_cross_split_ids(
    train_stems: Iterable[str], test_stems: Iterable[str], gap: int
) -> List[Tuple[int, int]]:
    """找出 train/test 中数值编号差不超过 gap 的候选相邻帧。"""
    train_ids = sorted(n for stem in train_stems if (n := numeric_stem(stem)) is not None)
    near_pairs: List[Tuple[int, int]] = []
    for test_id in sorted(n for stem in test_stems if (n := numeric_stem(stem)) is not None):
        position = bisect.bisect_left(train_ids, test_id)
        for index in (position - 1, position):
            if 0 <= index < len(train_ids) and abs(train_ids[index] - test_id) <= gap:
                near_pairs.append((train_ids[index], test_id))
    return near_pairs


def format_size_counter(counter: collections.Counter) -> str:
    if not counter:
        return "无可读取图像"
    return "; ".join(
        f"{width}x{height}: {count}"
        for (width, height), count in counter.most_common(8)
    )


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    example_limit = max(1, args.max_examples)

    heading("LLVIP 原始数据只读检查")
    print(f"数据根目录：{root}")
    print("本脚本仅读取数据，不会下载、复制、删除或转换任何文件。")

    if not root.is_dir():
        print("\n[失败] 数据根目录不存在或不是目录。")
        print("请先按 README 的目录结构解压完整 LLVIP 数据集。")
        return 2

    expected = {
        "infrared/train": root / "infrared" / "train",
        "infrared/test": root / "infrared" / "test",
        "visible/train": root / "visible" / "train",
        "visible/test": root / "visible" / "test",
        "Annotations": root / "Annotations",
    }
    missing_folders = [name for name, path in expected.items() if not path.is_dir()]
    heading("1. 官方目录结构")
    for name, path in expected.items():
        state = "存在" if path.is_dir() else "缺失"
        print(f"[{state}] {name}: {path}")
    if missing_folders:
        print("\n[失败] 缺少官方预期目录：" + ", ".join(missing_folders))
        print("提示：如果解压后多了一层 LLVIP 文件夹，请把 --data-root 指向那一层。")
        return 2

    image_indexes: Dict[Tuple[str, str], Dict[str, Path]] = {}
    duplicate_examples: List[str] = []
    for modality in ("infrared", "visible"):
        for split in ("train", "test"):
            index, duplicates = collect_image_index(root / modality / split)
            image_indexes[(modality, split)] = index
            duplicate_examples.extend(
                f"{modality}/{split}/{item}" for item in duplicates
            )
            print(f"{modality}/{split}: {len(index)} 个图像文件")
    show_examples("同 stem 重复图像", duplicate_examples, example_limit)

    heading("2. RGB/IR 同名配对与图像尺寸")
    records: List[PairRecord] = []
    image_read_errors: List[str] = []
    size_mismatches: List[str] = []
    split_stems: Dict[str, Set[str]] = {}

    for split in ("train", "test"):
        infrared = image_indexes[("infrared", split)]
        visible = image_indexes[("visible", split)]
        infrared_stems = set(infrared)
        visible_stems = set(visible)
        paired_stems = sorted(infrared_stems & visible_stems)
        only_infrared = sorted(infrared_stems - visible_stems)
        only_visible = sorted(visible_stems - infrared_stems)
        split_stems[split] = set(paired_stems)

        print(
            f"{split}: 配对 {len(paired_stems)}，"
            f"仅红外 {len(only_infrared)}，仅可见光 {len(only_visible)}"
        )
        show_examples(f"{split} 仅红外", only_infrared, example_limit)
        show_examples(f"{split} 仅可见光", only_visible, example_limit)

        for stem in paired_stems:
            infrared_size = safe_image_size(infrared[stem], image_read_errors)
            visible_size = safe_image_size(visible[stem], image_read_errors)
            if (
                infrared_size is not None
                and visible_size is not None
                and infrared_size != visible_size
            ):
                size_mismatches.append(
                    f"{split}/{stem}: IR={infrared_size}, RGB={visible_size}"
                )
            records.append(
                PairRecord(
                    split=split,
                    stem=stem,
                    infrared_path=infrared[stem],
                    visible_path=visible[stem],
                    infrared_size=infrared_size,
                    visible_size=visible_size,
                )
            )

    cross_split_duplicates = sorted(split_stems["train"] & split_stems["test"])
    if cross_split_duplicates:
        show_examples("同一文件名同时出现在 train/test", cross_split_duplicates, example_limit)
    else:
        print("train/test 没有完全相同的已配对文件名。")

    infrared_sizes = collections.Counter(
        record.infrared_size for record in records if record.infrared_size
    )
    visible_sizes = collections.Counter(
        record.visible_size for record in records if record.visible_size
    )
    print("红外尺寸分布：" + format_size_counter(infrared_sizes))
    print("可见光尺寸分布：" + format_size_counter(visible_sizes))
    show_examples("无法读取尺寸的图像", image_read_errors, example_limit)
    show_examples("同一对图像尺寸不一致", size_mismatches, example_limit)

    heading("3. VOC XML 标注与 YOLO 可转换性")
    xml_index, duplicate_xml = collect_xml_index(root / "Annotations")
    print(f"Annotations: {len(xml_index)} 个 XML 文件")
    show_examples("同 stem 重复 XML", duplicate_xml, example_limit)

    report = inspect_annotations(records, xml_index, example_limit)
    paired_stem_set = {record.stem for record in records}
    unpaired_xml = sorted(set(xml_index) - paired_stem_set)
    print(f"已配对样本缺少 XML: {report.missing_xml}")
    print(f"未匹配到图像对的 XML: {len(unpaired_xml)}")
    print(f"XML 解析失败: {report.parse_errors}")
    print(f"空标注 XML: {report.empty_annotations}")
    print(f"目标框总数: {report.objects}")
    print(f"类别与目标数量: {dict(report.categories)}")
    print(f"非法/越界框: {report.invalid_boxes}")
    print(f"XML 内 filename 不匹配: {report.xml_filename_mismatches}")
    print(f"XML 与红外图尺寸不匹配: {report.xml_size_mismatches}")
    show_examples("标注问题", report.examples, example_limit)
    show_examples("未匹配到图像对的 XML", unpaired_xml, example_limit)

    only_person = set(report.categories) == {"person"}
    conversion_ok = (
        report.missing_xml == 0
        and report.parse_errors == 0
        and report.invalid_boxes == 0
        and report.xml_size_mismatches == 0
        and only_person
    )
    if conversion_ok:
        print(
            "YOLO 转换检查：通过。可使用类别映射 person -> 0，将 "
            "(xmin, ymin, xmax, ymax) 按实际图像宽高归一化为 "
            "(x_center, y_center, width, height)。"
        )
    else:
        print(
            "YOLO 转换检查：发现需先处理或确认的问题；脚本没有写出任何标签文件。"
        )

    heading("4. 官方 train/test 划分的连续帧泄漏风险信号")
    for split in ("train", "test"):
        count = len(split_stems[split])
        expected_count = OFFICIAL_PAIR_COUNTS[split]
        number_count, adjacent_links, longest = continuous_runs(split_stems[split])
        longest_text = ", ".join(f"{start}-{end}" for start, end in longest) or "无"
        print(
            f"{split}: 已配对数 {count}（官方参考 {expected_count}）；"
            f"纯数字文件名 {number_count}；编号相邻链接 {adjacent_links}；"
            f"最长连续段 {longest_text}"
        )

    near_pairs = near_cross_split_ids(
        split_stems["train"], split_stems["test"], max(0, args.adjacent_gap)
    )
    if near_pairs:
        samples = [f"train={train_id}, test={test_id}" for train_id, test_id in near_pairs]
        show_examples(
            f"train/test 编号差不超过 {args.adjacent_gap} 的候选", samples, example_limit
        )
        print("结论：存在编号接近的跨集合样本，应人工核对是否来自同一连续拍摄序列。")
    else:
        print(
            f"未发现 train/test 编号差不超过 {args.adjacent_gap} 的样本。"
            "这不能证明场景隔离：文件名本身不含官方公开的场景 ID。"
        )
    print(
        "建议：用于 300–500 对小规模实验时，不直接随机抽样；先建立场景/连续序列清单，"
        "再按场景将样本整体分给 train、val、test。"
    )

    heading("5. 汇总")
    official_counts_match = all(
        len(split_stems[split]) == expected for split, expected in OFFICIAL_PAIR_COUNTS.items()
    )
    structural_ok = (
        not duplicate_examples
        and not cross_split_duplicates
        and not image_read_errors
        and not size_mismatches
        and official_counts_match
    )
    print(f"目录与图像配对检查：{'通过' if structural_ok else '需要查看上述问题'}")
    print(f"VOC/YOLO 可转换性检查：{'通过' if conversion_ok else '需要查看上述问题'}")
    print("下一步应基于真实检查结果编制场景清单；此脚本不会替你随机划分子集。")
    return 0 if structural_ok and conversion_ok else 1


if __name__ == "__main__":
    sys.exit(main())
