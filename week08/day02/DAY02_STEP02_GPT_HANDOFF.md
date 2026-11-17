# Week08 Day02 Step 2：RGB/IR 同步随机水平翻转学习交接

## 1. 学习背景与当前状态

我是一名软件工程硕士研一学生，已学习 CNN、ResNet18、迁移学习、注意力机制、FPN/PAN、xyxy 检测框、IoU 和检测损失。

项目：`D:\python\test\vision-learning`。
解释器：`D:\python\.venv\Scripts\python.exe`。
本次真实运行 PyTorch 为 `2.14.1+cpu`，在本地 CPU 完成。

Day01 已准备 `data/LLVIP_1000`，train/val/test 分别为 700/150/150 对。
Dataset 直接读取 RGB、IR、VOC XML，person 标签为 1，背景预留 0。

Day02 Step 1 的 Letterbox 已完成并验证，本步骤复用已有实现，处理顺序为：

```text
RGB/IR + VOC XML
    → 现有 Letterbox（1280×1024 → 内容640×512，上下各64 → 640×640）
    → 一个共享的随机翻转决策
    → 同步翻转两种模态以及边界框
```

**Step 2 已编码并完成真实验证，尚未开始 Step 3；代码运行通过不代表我已经理解。**
请全程中文、每次只讲一个明确步骤，等待我反馈，不主动安排测验。

## 2. 实际新增文件

所有新增文件均位于 `week08/day02`：

| 文件 | 用途 |
| --- | --- |
| paired_horizontal_flip.py | 随机决策、同步翻转、组合 Letterbox 的训练/验证/测试处理流程 |
| experiment_horizontal_flip.py | 真实样本、强制分支、双翻转、随机复现、DataLoader 与文件哈希检查 |
| STEP02_README.md | 文件说明、复现命令和 Dataset 接入方式 |
| DAY02_STEP02_GPT_HANDOFF.md | 当前学习交接文档 |
| outputs/step02_horizontal_flip/ | 本次独立保存的日志、JSON、图片 |

没有修改之前的 Letterbox 或 LLVIPDataset，也没有覆盖 Step1 实验结果。

## 3. 实现逻辑与关键 Python/PyTorch 代码

### 每个样本对只抽取一次随机决策

```python
flip = self.sample_decision()
return self.apply(visible, infrared, target, flip=flip)
```

当 0<p<1 时，使用 `torch.rand((), generator=generator).item() < p`。
`p=1` 强制翻转，`p=0` 强制不翻转，两端情况不消耗随机数。
RGB 与 IR 共用 `flip`，不能各调用一次随机函数。

### 沿宽度轴翻转图像

```python
visible_out = torch.flip(visible, dims=[-1])
infrared_out = torch.flip(infrared, dims=[-1])
```

输入为 CHW，最后一维 `-1` 是 W。沿这一维反转得到水平镜像。
翻转不会改变空间尺寸，也不会改变上下 Letterbox 填充的位置。

### 框使用原始两列计算新坐标

```python
result = {key: value.clone() for key, value in target.items()}
boxes = target["boxes"]
result["boxes"][:, 0] = width - boxes[:, 2]
result["boxes"][:, 2] = width - boxes[:, 0]
```

`[:,0]` 是全部框的 xmin，`[:,2]` 是全部框的 xmax。
两次赋值必须读原 `boxes`；如果先原地覆盖 xmin，再拿它算 xmax，会产生错误。
ymin、ymax 保持原值。框保持 float32，不取整。

这里 W=640，采用连续像素边界坐标，公式是 `W-x`，不是图像离散像素索引的 `W-1-x`。
xyxy 框允许 xmax 等于 W；边界框和像素索引不要混淆。

### target 字段和输入保护

```python
new_boxes = result["boxes"]
result["area"] = ((new_boxes[:, 2] - new_boxes[:, 0])
                  * (new_boxes[:, 3] - new_boxes[:, 1]))
```

水平翻转保持宽高，因此本次 area 前后相同；labels、image_id、iscrowd 克隆保留。
`clone()` 避免输出与输入共享可写存储。强制不翻转时，图像也返回 clone。
空框 `(0,4)` 可以直接执行上述列运算，输出仍为 `(0,4)`，area 为 `(0,)`。
遇到尺寸不一致、非 float32 框、非有限坐标、越界框或非正宽高时会报错。

### 训练与验证/测试的区别

`PairedLetterboxAndFlip(split)` 组合既有 `PairedLetterbox(640)` 和新翻转类：

- train：默认 p=0.5，每次调用可以产生新的共享随机决策。
- val/test：p=0，默认不启用翻转，保留 Letterbox。

固定种子是为了复现实验序列，不意味着每个 batch 或每次短实验恰好有一半样本翻转。

## 4. 真实样本与全部坐标

读取训练集中正样本 `010001` 和合法空标注样本 `100030`。
两者先经过既有 Letterbox，之后的图像宽高都是 640×640。
下表“翻转前”是 **Letterbox 后** 的坐标，不是 1280×1024 原图坐标。
原图坐标也保存到了 results.json 的 `boxes_raw`。

正样本 `010001` 的强制翻转（p=1）实测：

| 框 | 翻转前 xyxy | 翻转后 xyxy | 前后相同的 area |
| --- | --- | --- | ---: |
| 1 | [143.5,278,175.5,395] | [464.5,278,496.5,395] | 3744.0 |
| 2 | [175.5,259.5,212,385] | [428,259.5,464.5,385] | 4580.75 |
| 3 | [233,247.5,275,371] | [365,247.5,407,371] | 5187.0 |
| 4 | [350,241,380.5,356.5] | [259.5,241,290,356.5] | 3522.75 |
| 5 | [352,322.5,403,461] | [237,322.5,288,461] | 7063.5 |
| 6 | [562,75,598,186.5] | [42,75,78,186.5] | 4014.0 |

例如第一个框：`640-175.5=464.5`，`640-143.5=496.5`。

强制不翻转（p=0）实测：以上 6 个框、RGB、IR、area 以及其他 target 字段与输入相同。

负样本 `100030` 实测：

```text
p=1 → boxes=[]，shape=(0,4)，area=[]
p=0 → boxes=[]，shape=(0,4)，area=[]
```

两组样本连续强制翻转两次后，RGB、IR 和所有 target 字段精确恢复；检查容差为 rtol=0、atol=0。

## 5. 真实运行命令与日志结果

从项目根目录运行：

```powershell
Set-Location D:\python\test\vision-learning
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day02\experiment_horizontal_flip.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day02\outputs\step02_horizontal_flip `
  --seed 42
```

执行退出码为 0。输出目录已存在且含结果；重跑需改用新目录名。
`-B` 避免在既有目录写入 Python 字节码缓存。

真实终端结果摘录（完整原文在 terminal_output.txt）：

```text
Week08 Day02 Step 2 | seed=42 | CPU | num_workers=0
环境 torch=2.14.1+cpu; dataset train=700
[train/010001] negative=False; W=640; boxes=(6, 4)
PASS: 强制两分支同步、框/面积/字段、输入不变且无共享存储、双翻转精确恢复
[train/100030] negative=True; W=640; boxes=(0, 4)
PASS: 强制两分支同步、框/面积/字段、输入不变且无共享存储、双翻转精确恢复
p=0.5，固定种子随机序列（True=翻转）：[False, False, True, False, True, False, True, False, False, True, False, False]
PASS: 两次独立同种子运行的决策、RGB/IR 和 boxes 相同
DataLoader train: p=0.5; decisions=[False, False]; RGB/IR=(3,640,640); boxes=[[6, 4], [0, 4]]; PASS
DataLoader val: p=0.0; decisions=[False, False]; RGB/IR=(3,640,640); boxes=[[2, 4], [3, 4]]; PASS
DataLoader test: p=0.0; decisions=[False, False]; RGB/IR=(3,640,640); boxes=[[2, 4], [7, 4]]; PASS
PASS: 拒绝 RGB/IR 尺寸不一致（人为构造的反例）
PASS: Day01、Step1 代码/产物、manifest 与所用源文件 SHA256 前后一致
ALL CHECKS PASSED
```

随机复现测试在同一个真实正样本上执行 12 个决策。本次观察到 4 次翻转；理论上的 p=0.5 不要求 12 次中必须翻转 6 次。
DataLoader 使用独立重置到 seed=42 的 Generator，所以训练 batch 的前两次决策为 False。这不代表训练增强被禁用；强制翻转及 12 次随机验证已经覆盖 True 分支。

## 6. 日志与图片位置

结果目录：`D:\python\test\vision-learning\week08\day02\outputs\step02_horizontal_flip`。

| 文件 | 内容 |
| --- | --- |
| terminal_output.txt | 完整 stdout/stderr 日志 |
| results.json | 原图/Letterbox 后/翻转后坐标、area、随机序列、batch 检查结果、受保护文件哈希 |
| train_010001_flip_comparison.jpg | 正样本强制翻转前后 RGB/IR |
| train_010001_no_flip_comparison.jpg | 正样本强制不翻转 |
| train_100030_flip_comparison.jpg | 空标注样本强制翻转 |
| train_100030_no_flip_comparison.jpg | 空标注样本强制不翻转 |
| overview.jpg | 四组对比总览 |

每张 comparison 图上排是翻转前 RGB/IR，下排是变换后。总览图左列是翻转、右列是不翻转，上半部分为正样本，下半部分为空标注样本。
已实际查看总览图：RGB/IR 一起镜像，标注框随行人一起移动；不翻转分支视觉保持一致，空标注前后均不画框。

## 7. 本次实际通过的验证清单

- 正样本和空标注各自覆盖 p=1、p=0。
- 每种模态输出与其自身的预期镜像逐元素一致，确保共用决策。
- 所有框坐标满足 W-x 公式，y 不变，float32 不变，框合法。
- 面积与 labels/image_id/iscrowd 正确。
- 源图张量和原 target 未被原地修改，输出使用独立存储。
- 两次水平翻转精确恢复两个真实样本的图像和 target。
- 固定种子下两次独立随机序列及输出一致。
- train/val/test 的真实 DataLoader batch 均通过；包含空框。
- 构造尺寸不一致输入，正确拒绝。这是人为异常测试，不是发现真实数据缺陷。
- Day01 代码/文档、Letterbox 与 Step1 实验脚本、Step1 全部结果文件、manifest 和本次使用的源 JPG/XML 哈希前后相同。

本次不是模型评估，没有 loss、AP 或训练指标。没有全量扫描 1,000 对图像的翻转输出。
显式 Generator 的复现验证限于本次 CPU、num_workers=0、固定读取顺序；以后启用多 worker 时应另行配置各 worker 种子。

## 8. 当前进度与交互式学习建议

当前已完成 Day02 Step 1（Letterbox）和 Step 2（同步随机水平翻转）的实现及验证。
本次停在这里，先进行原理学习，不开始 Step3。

建议学习顺序：先理解一个共享随机决策，再看 `torch.flip(..., dims=[-1])`，然后用上表演算框坐标，最后理解 clone、空框和固定种子。

给 ChatGPT 的开场提示词：

> 我已完成 Week08 Day02 Step 2 的真实实验，请根据这份交接文档辅导我。先只讲 RGB/IR 为什么必须共用同一个随机决策，再解释 torch.flip 的最后一维含义，讲完等待我反馈。后续再讲真实框的 W-x 公式、clone 防止原地修改、空框和固定种子。不要默认代码通过就代表我已理解，不主动安排测验，不开始 Step3 或模型训练。

普通 ChatGPT 不能仅凭本地路径访问本机文件。上传本文档即可开始原理学习；逐行讲实现时同时上传 `paired_horizontal_flip.py`，分析视觉结果时上传 overview.jpg，需要核对实验时上传 results.json 和 terminal_output.txt。
