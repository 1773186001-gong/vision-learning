# Week08 Day02 Step 1：多模态同步 Letterbox

## 本步骤的目标与结果

将同名 RGB、IR 等比例缩放到 640×640 的画布，并同步变换 VOC xyxy 像素框。
实际验证了 train/010001、val/030001、test/190001 和 train/100030 四个真实样本；最后一个为空标注样本。

全部原图为 W×H=1280×1024。实际变换为：

```text
r = min(640/1280, 640/1024) = 0.5
缩放后的内容尺寸：640×512
填充 left/top/right/bottom：0/64/0/64
最终画布：640×640
xmin' = xmin × 0.5
xmax' = xmax × 0.5
ymin' = ymin × 0.5 + 64
ymax' = ymax × 0.5 + 64
```

填充值为 `114/255`，两种模态均使用同一组缩放、插值和填充参数。
使用双线性插值及 antialias；target 的 area 根据变换后框重新计算，类别和其他字段不变。
输入 target 会复制，不原地改写。对空框的广播运算保持 `(0,4)`，area 保持 `(0,)`。

## 完整运行命令

在 PowerShell 中运行（输出目录应不存在或为空；脚本拒绝覆盖已有实验结果）：

```powershell
Set-Location D:\python\test\vision-learning
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day02\experiment_letterbox.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day02\outputs\step01_letterbox
```

需要再次运行时，改成一个新的输出目录，例如 `step01_letterbox_run02`。
`-B` 避免向 Day01 写入 Python 字节码缓存。

## 代码入口与 Dataset 接入

- `paired_letterbox.py`：同步图像/边界框变换和参数计算。
- `experiment_letterbox.py`：真实样本检查、集成检查、可视化及结果保存。

利用 Day01 已有的 transforms 参数接入，无需修改 LLVIPDataset：

```python
dataset = LLVIPDataset(
    root=data_root,
    split="train",
    modality="paired",
    transforms=PairedLetterbox(size=640),
)
```

## 真实实验记录

| 样本 | 原始框数 | 变换后框数 | 输出 RGB/IR |
| --- | ---: | ---: | --- |
| train/010001 | 6 | 6 | 3×640×640 |
| val/030001 | 2 | 2 | 3×640×640 |
| test/190001 | 2 | 2 | 3×640×640 |
| train/100030 | 0 | 0，shape=(0,4) | 3×640×640 |

实测通过：

1. 原始 RGB、IR、XML 尺寸一致、文件名一致。
2. 输出框与 `原框×r+[left,top,left,top]` 相同，逆变换恢复原框。
3. 输出框有限、正宽高、不越界；未通过裁剪或删除框掩盖问题。
4. 面积为原面积的 0.25 倍；labels、image_id、iscrowd 保持原值。
5. 传给两分支相同图像，输出逐元素相同，验证同步几何操作。
6. DataLoader 的 batch=2，RGB/IR 均为 `(3,640,640)`。
7. 单独构造 IR 少一行的反例，正确抛出尺寸不一致错误；该反例不是数据集缺陷。
8. Day01 的 Python/Markdown、manifest 和四组源 JPG/XML 的 SHA256 前后一致。

完整终端输出：`outputs/step01_letterbox/terminal_output.txt`。
全部原始框、输出框、参数、受保护文件哈希：`outputs/step01_letterbox/results.json`。

## 标注对比图

位于 `outputs/step01_letterbox/`：

- `train_010001_comparison.jpg`
- `val_030001_comparison.jpg`
- `test_190001_comparison.jpg`
- `train_100030_comparison.jpg`
- `overview.jpg`

每张对比图上排是原尺寸 RGB/IR，下排是原生 640×640 结果；下排周围的白色是排版画布，**灰色才是 Letterbox 的上下填充**。
人工查看总览图：四组样本未见明显的新增整体框偏移；正样本的框仍覆盖对应行人，负样本前后均没有框。

## 检查范围与边界

同名、同尺寸不等于像素级配准正确。本步骤对四组样本进行了视觉抽查，不声称自动验证整个数据集的跨模态配准，也不自动做配准修正。遇到尺寸不一致直接报错；若人工看到空间错位，应停止后续实验并报告具体编号。

任意其他原图尺寸下，缩放尺寸取 `round(W×r), round(H×r)`，存在最多半像素的整数栅格舍入，框使用同一名义比例 r；当前 LLVIP 的 1280×1024→640×512 没有舍入误差。

本步骤结束于 Letterbox 检查；尚未添加随机翻转或开始训练。
