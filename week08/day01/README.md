# Week08 Day01：LLVIP 数据准备与只读检查

本目录包含数据准备说明、原始数据只读检查脚本，以及场景隔离的子集生成脚本。**不包含 LLVIP 图像、模型权重或训练代码。**

## 目标

为后续“红外与可见光多模态行人检测”准备可复核的数据检查和子集筛选流程。不自动下载数据、不修改原始数据、不启动训练。

## 1. 从官方页面手动获取完整数据集

请在浏览器中打开 LLVIP 官方下载页：

- [LLVIP 官方下载页面（GitHub）](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)
- [LLVIP 官方项目主页](https://bupt-ai-cz.github.io/LLVIP/)

官方页面提供：

1. Google Drive 下载链接；
2. 百度网盘下载链接，提取码为 `14lc`。

请只下载常规的 **LLVIP dataset** 压缩包；页面中的 *raw data* 是未配准图像和视频，用于配准研究，本周不需要下载。官方没有按场景、模态或 300–500 对样本提供分片下载，因此若要严格使用官方数据，需要先人工下载完整数据集，再在本地制作实验子集。

LLVIP 仅限非商业用途，使用时应引用原论文并遵守官方许可：[Term of Use and License](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md)。

> 本项目不会自动调用下载命令。请在确认本机存储空间足够、并手动完成下载后再进入下一步。

## 2. 推荐的数据存放位置

推荐将压缩包解压到下面的目录。当前实际数据位于项目根目录的 `LLVIP`，运行命令时以实际路径为准：

```text
D:\python\test\vision-learning\data\LLVIP\
├── Annotations\
│   ├── 010001.xml
│   └── ...
├── infrared\
│   ├── train\
│   └── test\
└── visible\
    ├── train\
    └── test\
```

不要放入 `week08/day01`，也不要与已有的 `data/train`、`data/test` 猫狗练习数据混放。

有些解压工具会额外创建一层 `LLVIP` 目录。运行脚本时，`--data-root` 必须指向直接包含 `Annotations`、`infrared`、`visible` 的那一层。

官方发布的当前版本应包含 15,488 对图像：训练集 12,025 对、测试集 3,463 对。每对同名图像由一个同名 XML 标注对应。

## 3. 运行只读检查

数据准备好后，在项目根目录的 PowerShell 中运行：

```powershell
& D:\python\.venv\Scripts\python.exe .\week08\day01\inspect_llvip.py --data-root .\LLVIP
```

可选参数：

```powershell
# 每一类问题显示 10 个示例；将“编号接近”定义为差不超过 10
& D:\python\.venv\Scripts\python.exe .\week08\day01\inspect_llvip.py `
  --data-root .\LLVIP `
  --max-examples 10 `
  --adjacent-gap 10
```

脚本仅使用 Python 标准库，**不会安装依赖、写入 YOLO 标签、复制/删除文件或修改 XML**。它会检查：

1. `Annotations`、`infrared/{train,test}`、`visible/{train,test}` 是否齐全；
2. RGB 与 IR 是否按相同文件名一一配对，且尺寸一致；
3. train/test 图像对计数是否与官方的 12,025 / 3,463 对相符；
4. XML 是否可解析、类别是否为 `person`、边界框是否有效且未越界；
5. VOC XML 是否具备转换为 YOLO 格式的条件；
6. 文件编号中的连续段，以及 train/test 之间编号接近的风险信号。

真实数据检查结果：15,488 对 RGB/IR 配对完整，图像均为 `1280×1024`；15,488 个 XML 中类别均为 `person`。其中有 2 个合法空标注和 5 个零宽非法框。后续子集生成会保留合法负样本，并自动排除非法框样本，不修改原始 XML。

## 4. VOC XML 与 YOLO 的对应关系

LLVIP 的标注为 Pascal VOC XML。对于 `person` 类，后续会使用类别编号 `0`：

```text
0 x_center y_center width height
```

其中：

```text
x_center = ((xmin + xmax) / 2) / image_width
y_center = ((ymin + ymax) / 2) / image_height
width    = (xmax - xmin) / image_width
height   = (ymax - ymin) / image_height
```

官方仓库也提供 XML 转 YOLOv5 的工具，但其脚本将图像尺寸固定为 `1280×1024`。因此本项目会先用 `inspect_llvip.py` 实测图像与 XML 尺寸；转换脚本将在检查通过并得到确认后再编写。

## 5. 1,000 对子集的防泄漏方案

本次采用 **1,000 对**作为可复现实验子集：

| 划分 | 图像对数 | 场景数建议 |
| --- | ---: | ---: |
| train | 700 | 15 |
| val | 150 | 4 |
| test | 150 | 7 |

真实文件统计显示，六位文件名的前两位恰好形成 26 个互斥分组，与官方的 26 个采集场景一致。官方 train 包含 `01–18、25`，官方 test 包含 `19–24、26`。本项目将官方 train 中的 `03、08、13、17` 四个完整场景划为 val，其余 15 个场景用于 train；官方 test 的 7 个场景只用于 test。

后续的正确步骤是：

1. 以场景前缀为整体，分配到 train、val、test 三者之一；
2. 在每个场景内部按编号均匀抽帧，避免连续抽取大量相邻帧；
3. 自动排除缺失文件、异常类别和非法边界框；
4. 始终将同名 RGB、IR、XML 作为一个不可拆分的样本对；
5. 固定随机种子，并保存最终清单，确保实验可复现。

这样才能避免同一机位、相邻帧同时落入不同集合而造成过于乐观的评估结果。

先预览选择方案，不复制文件：

```powershell
& D:\python\.venv\Scripts\python.exe .\week08\day01\prepare_llvip_subset.py `
  --source-root .\LLVIP `
  --output-root .\data\LLVIP_1000
```

确认预览后，显式添加 `--execute` 才会复制到新目录：

```powershell
& D:\python\.venv\Scripts\python.exe .\week08\day01\prepare_llvip_subset.py `
  --source-root .\LLVIP `
  --output-root .\data\LLVIP_1000 `
  --execute
```

输出包含 RGB/IR 图像、原始 VOC XML、`manifest.csv` 和 `summary.json`。当前步骤不生成 YOLO 标签。

## 6. 直接读取 RGB、IR 和 VOC XML

不使用 YOLO 时，可通过 `llvip_dataset.py` 直接建立 PyTorch Dataset。类别约定为 `0=背景、1=person`，与 TorchVision Faster R-CNN 一致。

`paired` 模式返回：

```python
image = {
    "visible": rgb_tensor,     # [3, H, W]，float32，范围 [0, 1]
    "infrared": ir_tensor,    # 默认复制为 [3, H, W]
}

target = {
    "boxes": boxes,           # [N, 4]，xyxy 像素坐标
    "labels": labels,         # person = 1
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd,
}
```

只读取前两个 batch 验证 DataLoader：

```powershell
& D:\python\.venv\Scripts\python.exe .\week08\day01\check_llvip_dataloader.py `
  --data-root .\data\LLVIP_1000 `
  --split train `
  --modality paired `
  --batch-size 2 `
  --num-batches 2 `
  --num-workers 0
```

Windows 下初次验证使用 `num_workers=0` 最稳妥。该命令不使用 GPU、不训练模型，也不修改数据。

## 7. RGB/IR 配对与边界框可视化

从 train、val、test 各选择 3 个不同场景的代表样本，并额外可视化合法负样本：

```powershell
& D:\python\.venv\Scripts\python.exe .\week08\day01\visualize_llvip_pairs.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day01\outputs\llvip_pair_visualizations `
  --samples-per-split 3
```

每张结果图左侧为 RGB、右侧为 IR，两边绘制完全相同的 VOC 边界框。人工检查重点：

1. 同一个编号框是否在两种模态中覆盖同一名行人；
2. 框是否明显偏离目标；
3. RGB 在低照度下丢失的信息是否能由 IR 补充；
4. 负样本是否确实没有行人标注。

输出目录还包含 `overview.jpg` 总览图和 `visualization_manifest.csv` 清单。

## 8. Day01 完成状态

Week08 Day01 已完成并终结。完整执行结果、知识总结和交给 ChatGPT 的续学提示词见 [DAY01_HANDOFF.md](DAY01_HANDOFF.md)。下一次学习从 Week08 Day02 开始。
