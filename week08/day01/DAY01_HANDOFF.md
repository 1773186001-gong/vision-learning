# Week08 Day01 学习交接：LLVIP 多模态行人检测数据准备

## 一、学习背景

学习者为软件工程硕士研一，已经完成 Week01–Week07，学习内容包括：

- OpenCV 与图像基础；
- PyTorch Dataset、DataLoader 和 CNN；
- ResNet18、迁移学习与模型评估；
- SE、CBAM、Coordinate Attention；
- FPN、PAN；
- 目标检测框、IoU 和检测损失。

Week08 开始学习红外图像、红外—可见光融合及小目标检测。Day01 的目标是准备 LLVIP 数据，并建立不依赖 YOLO 的 PyTorch 读取流程。

## 二、Day01 实际完成内容

### 1. LLVIP 官方数据调研

已确认：

- LLVIP 是低照度 RGB—热红外配对行人数据集；
- 当前版本共有 15,488 对图像，即 30,976 张 JPG；
- 官方 train 为 12,025 对，test 为 3,463 对；
- 图像尺寸为 `1280×1024`；
- RGB、IR 和 XML 使用相同文件名配对；
- 标注为 Pascal VOC XML，类别只有 `person`；
- 数据来自 26 个采集场景。

官方数据位于：

```text
D:\python\test\vision-learning\LLVIP
```

### 2. 完整原始数据检查

使用：

```text
week08/day01/inspect_llvip.py
```

对完整数据进行了只读检查，真实结果如下：

- 15,488 对 RGB/IR 全部正确配对；
- 所有图像均为 `1280×1024`；
- 15,488 个 XML 均存在；
- 总计 42,437 个 `person` 框；
- 有 2 个合法空标注样本；
- 有 5 个零宽非法框：
  - `020118.xml`
  - `020236.xml`
  - `090437.xml`
  - `091180.xml`
  - `170350.xml`

没有修改或修复原始 XML；后续子集生成时直接排除了这 5 个异常样本。

### 3. 场景隔离的 1,000 对实验子集

使用：

```text
week08/day01/prepare_llvip_subset.py
```

生成目录：

```text
D:\python\test\vision-learning\data\LLVIP_1000
```

子集共 1,000 对 RGB/IR，即 2,000 张 JPG，大小约 261.4 MiB。

| 集合 | 图像对 | 场景数 | person 框 |
| --- | ---: | ---: | ---: |
| train | 700 | 15 | 1,991 |
| val | 150 | 4 | 416 |
| test | 150 | 7 | 348 |
| 合计 | 1,000 | 26 | 2,755 |

场景划分：

```text
train: 01, 02, 04, 05, 06, 07, 09, 10, 11, 12, 14, 15, 16, 18, 25
val:   03, 08, 13, 17
test:  19, 20, 21, 22, 23, 24, 26
```

划分原则：

- 文件名前两位作为场景分组；26 个前缀与官方 26 个场景吻合；
- 同一场景只属于 train、val、test 中的一个集合；
- 每个场景内部按编号均匀抽帧，减少相邻帧冗余；
- RGB、IR、XML 始终作为不可拆分的一个样本对；
- 官方 test 场景只用于子集 test；
- 原始 LLVIP 始终只读。

子集包含：

```text
data/LLVIP_1000/
├── images/
│   ├── train/{visible,infrared}/
│   ├── val/{visible,infrared}/
│   └── test/{visible,infrared}/
├── annotations_voc/{train,val,test}/
├── manifest.csv
└── summary.json
```

### 4. 不使用 YOLO 的 PyTorch Dataset

最终决定不使用 YOLO 格式。VOC XML 由 PyTorch Dataset 直接解析。

主要文件：

```text
week08/day01/llvip_dataset.py
week08/day01/check_llvip_dataloader.py
```

`LLVIPDataset` 支持三种模式：

```text
visible   # RGB 单模态
infrared  # IR 单模态
paired    # 同时返回 RGB 和 IR
```

`paired` 模式返回：

```python
image = {
    "visible": rgb_tensor,     # [3, H, W]，float32，[0, 1]
    "infrared": ir_tensor,    # [3, H, W]，float32，[0, 1]
}

target = {
    "boxes": boxes,           # [N, 4]，xyxy 像素坐标
    "labels": labels,         # person = 1
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd,
}
```

TorchVision 检测类别约定：

```text
0 = background
1 = person
```

DataLoader 已使用真实数据验证：

- train/val/test 数量分别为 700/150/150；
- RGB 张量为 `(3, 1024, 1280)`；
- IR 张量为 `(3, 1024, 1280)`；
- 边界框为 `(N, 4)`；
- 合法负样本 `100030` 正确返回 `(0, 4)` 空边界框；
- 检查过程中没有训练模型，也没有修改数据。

### 5. RGB/IR 配对与边界框可视化

使用：

```text
week08/day01/visualize_llvip_pairs.py
```

输出目录：

```text
week08/day01/outputs/llvip_pair_visualizations
```

共生成 10 个配对可视化：train、val、test 各 3 个不同场景，加 1 个合法负样本，并生成 `overview.jpg` 总览图。

人工检查结论：

- RGB 与 IR 空间位置基本对齐；
- 同一边界框在两种模态中覆盖同一名行人；
- 未发现明显的整体框偏移；
- 低照度 RGB 中部分行人与背景对比度较低；
- IR 中人体通常更亮、轮廓更突出；
- 负样本没有绘制边界框，处理正确。

## 三、Day01 核心知识

1. 多模态检测的基本样本单位是“一对 RGB/IR + 一份标注”，不能拆开。
2. 视频连续帧不能随机分散到 train、val、test，否则会产生数据泄漏。
3. 场景隔离比普通随机划分更适合 LLVIP。
4. Pascal VOC XML 可以直接用于 PyTorch 检测，不必须转换成 YOLO TXT。
5. TorchVision 检测器将类别 `0` 留给背景，因此 `person` 使用类别 `1`。
6. 检测任务中每张图的目标数量不同，DataLoader 需要自定义 `collate_fn`，将图像和 target 保留为 list。
7. 空标注是合法负样本，`boxes` 必须保持 `(0, 4)`，不能变成 `(0,)`。
8. RGB 与 IR 后续进行翻转、缩放、裁剪时，必须使用完全相同的几何变换参数。
9. 红外图在低照度下能突出人体目标，但缺少可见光的颜色和纹理信息，两种模态具有互补性。

## 四、本地环境与执行限制

本地验证环境：

```text
Python:      D:\python\.venv\Scripts\python.exe
PyTorch:     2.14.1+cpu
torchvision: 0.29.1+cpu
Pillow:      12.3.0
```

Day01 没有执行以下操作：

- 没有连接 RTX 3060 服务器；
- 没有安装或升级依赖；
- 没有启动训练；
- 没有生成模型指标；
- 没有修改 Week01–Week07；
- 没有修改原始 LLVIP 数据；
- 没有使用 YOLO。

## 五、主要文件索引

```text
week08/day01/
├── README.md
├── DAY01_HANDOFF.md
├── inspect_llvip.py
├── prepare_llvip_subset.py
├── llvip_dataset.py
├── check_llvip_dataloader.py
├── visualize_llvip_pairs.py
└── outputs/llvip_pair_visualizations/
```

## 六、交给 ChatGPT 的续学提示词

可将下面内容直接交给 ChatGPT：

> 我已经完成 Week08 Day01：LLVIP 多模态行人检测数据准备。项目目录是 `D:\python\test\vision-learning`。已生成场景隔离的 `data\LLVIP_1000` 子集，划分为 train 700 对、val 150 对、test 150 对。当前不使用 YOLO，`week08\day01\llvip_dataset.py` 可以直接读取 RGB、IR 和 VOC XML，类别约定为 `0=background、1=person`。DataLoader 和 RGB/IR 标注可视化均已通过。请全程使用中文，每次只推进一个明确步骤，不连接服务器、不安装依赖、不开始大规模训练。下一阶段请先阅读 `week08\day01\DAY01_HANDOFF.md` 和现有代码，再规划 Week08 Day02。

## 七、Day01 状态

**Week08 Day01 已完成并终结。**

下一次学习应从 Week08 Day02 开始，不再重复下载、划分或转换 LLVIP 数据。
