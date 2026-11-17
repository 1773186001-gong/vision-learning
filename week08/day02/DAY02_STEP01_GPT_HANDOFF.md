# Week08 Day02 Step 1：给 ChatGPT 的学习交互与实验交接文档

## 1. 请 ChatGPT 如何使用本文档

这是一份已经实际执行的实验记录，用于辅导我理解现有代码和实验结果。
请先讲解本步骤，再根据我的反馈继续；不要将“代码已运行”理解成“我已经掌握”。

我是一名软件工程硕士研一学生，正在从基础学习计算机视觉和深度学习。已完成 CNN、ResNet18、迁移学习、模型评估、注意力机制、FPN/PAN、检测框、IoU 与检测损失。

请全程中文，每次只讲一个明确步骤，结合真实坐标和代码解释，不主动安排测验。
当前进度为 **Week08 Day02 Step 1 已实现并验证**；不代表整个 Day02 已完成。
暂未添加随机翻转。必须等我确认 Letterbox 学习与实验成功后，才讨论或实施下一步。

普通 ChatGPT 对话无法仅凭下面的本地路径读取文件。如果需要看具体实现，请让我上传相应代码或粘贴内容，不要声称已经读取本机文件。

## 2. 前置数据与环境

项目：`D:\python\test\vision-learning`

解释器：`D:\python\.venv\Scripts\python.exe`

Day01 已准备好 LLVIP 子集：`data/LLVIP_1000`。共 1,000 对 RGB/IR，train=700 对、val=150 对、test=150 对。保留 VOC XML，通过原有 `LLVIPDataset` 直接读取，不依赖 YOLO。

Dataset 返回：

```python
images = {
    "visible": rgb,   # float32，[3,H,W]，[0,1]
    "infrared": ir,   # 灰度复制为3通道，[3,H,W]，[0,1]
}
target = {
    "boxes": boxes,   # float32，[N,4]，xyxy 像素坐标
    "labels": labels, # person=1；0预留为背景
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd,
}
```

空标注必须是 `[0,4]`，不是 `[0]`。这是没有行人框的合法样本。

## 3. 本步骤解决什么问题

把 1280×1024 的 RGB/IR 输入放到 640×640 画布上，同时保持图像纵横比，并让标注框跟随图像移动。

直接把 1280×1024 拉伸成 640×640，会让横向缩放比例为 0.5、纵向为 0.625，导致形状改变。本实验采用 Letterbox：先等比例缩小，再填充。

尺寸统一采用 **宽 W × 高 H**；Tensor 则使用 **C,H,W**。不要混淆这两种顺序。

```text
r = min(640/W, 640/H)
new_W = round(W*r)
new_H = round(H*r)
left = (640-new_W)//2
top  = (640-new_H)//2
right  = 640-new_W-left
bottom = 640-new_H-top
```

本次真实数据均为 1280×1024，参数是：

```text
r = 0.5
内容尺寸：640×512
填充 left/top/right/bottom：0/64/0/64
输出尺寸：640×640
填充值：114/255（float32 图像的灰色）
```

RGB 和 IR 复用同一个参数对象，使用双线性插值和 antialias。没有分别计算两套比例，也没有随机增强。

## 4. 边界框如何跟随图像变换

边界框继续使用 xyxy **像素坐标**，没有归一化到 [0,1]：

```text
xmin' = xmin*r + left
ymin' = ymin*r + top
xmax' = xmax*r + left
ymax' = ymax*r + top
```

例如 train/010001 的第一个框：

```text
原始：[287, 428, 351, 662]
缩放：[143.5, 214, 175.5, 331]
加上填充：[143.5, 278, 175.5, 395]
```

这里保留小数是正常的，框坐标使用 float32，不应随意转成整数。

框面积重新计算为 `(xmax'-xmin')*(ymax'-ymin')`。本次面积为原面积的 0.25 倍；平移不会改变面积。

空框 `[0,4]` 变换后仍是 `[0,4]`，area 为 `[0]`。labels、image_id、iscrowd 保持不变。
变换复制 target，不原地改写输入。

## 5. 实际执行的样本及全部框坐标

以下结果来自已保存的真实 `results.json`，不是模拟检测结果。

| 集合/编号 | 原始框数 | 变换后 boxes 形状 |
| --- | ---: | --- |
| train/010001 | 6 | (6,4) |
| val/030001 | 2 | (2,4) |
| test/190001 | 2 | (2,4) |
| train/100030 | 0 | (0,4) |

这四组的 RGB/IR 原始尺寸均为 1280×1024，比例均为 0.5，LTRB 填充均为 (0,64,0,64)，输出均为 640×640。

| 样本 | 原始 xyxy | Letterbox 后 xyxy |
| --- | --- | --- |
| train/010001 #1 | [287,428,351,662] | [143.5,278,175.5,395] |
| train/010001 #2 | [351,391,424,642] | [175.5,259.5,212,385] |
| train/010001 #3 | [466,367,550,614] | [233,247.5,275,371] |
| train/010001 #4 | [700,354,761,585] | [350,241,380.5,356.5] |
| train/010001 #5 | [704,517,806,794] | [352,322.5,403,461] |
| train/010001 #6 | [1124,22,1196,245] | [562,75,598,186.5] |
| val/030001 #1 | [2,1,88,217] | [1,64.5,44,172.5] |
| val/030001 #2 | [342,3,462,220] | [171,65.5,231,174] |
| test/190001 #1 | [1018,330,1137,588] | [509,229,568.5,358] |
| test/190001 #2 | [1204,304,1280,531] | [602,216,640,329.5] |
| train/100030 | [] | []，shape=(0,4) |

## 6. 已实际通过的检查

1. 四组样本 RGB、IR、XML 同名且原始尺寸一致。
2. 变换后的框符合缩放和平移公式，逆变换恢复原坐标。
3. 输出框有限、宽高为正、没有越界；空框保持正确形状。
4. 面积更新正确，其他 target 字段和输入 target 保持不变。
5. 两个分支输入相同图像时，输出逐元素相同。
6. 通过原 Dataset 的 transforms 参数接入 DataLoader：batch=2，RGB/IR 都为 `(3,640,640)`。
7. 专门把 IR 截掉一行构造尺寸不一致反例，代码正确报错并拒绝强制对齐。这是异常处理测试，不是 LLVIP 真实缺陷。
8. Day01 Python/Markdown 文件、manifest 和这四组源图/XML 的 SHA256 前后一致。

原有 LLVIPDataset 未修改。本步骤没有修改原始数据、安装依赖、连接服务器或启动训练。

## 7. 可视化观察与限制

每个样本均生成一张前后对比图：上排是原尺寸 RGB/IR，下排是原生 640×640 结果。
下排图周围的白色属于排版画布，灰色才是 Letterbox 填充。

已查看总览图：本次抽查没有发现新增的明显整体框偏移，框仍覆盖对应行人；空标注样本变换前后均不画框。

这不代表已自动验证全部 1,000 对数据的像素级配准。同名和同尺寸不能证明空间配准正确。
如果人工检查发现 RGB/IR 对应位置不同，应报告具体样本，停止后续处理，不要静默强制配准。

对于其他尺寸，整数缩放尺寸存在最多半像素的栅格舍入，代码框坐标使用名义比例 r。本次 1280×1024→640×512 没有舍入误差。

## 8. 文件、结果和复现命令

本步骤文件：

```text
week08/day02/
├── paired_letterbox.py
├── experiment_letterbox.py
├── README.md
├── DAY02_STEP01_GPT_HANDOFF.md
└── outputs/step01_letterbox/
    ├── terminal_output.txt
    ├── results.json
    ├── overview.jpg
    ├── train_010001_comparison.jpg
    ├── val_030001_comparison.jpg
    ├── test_190001_comparison.jpg
    └── train_100030_comparison.jpg
```

完整实验根目录为 `D:\python\test\vision-learning\week08\day02`。

实际运行命令：

```powershell
Set-Location D:\python\test\vision-learning
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day02\experiment_letterbox.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day02\outputs\step01_letterbox
```

现有结果已保存，脚本拒绝覆盖非空输出目录；重跑需改用新目录名。

Dataset 接入方式：

```python
dataset = LLVIPDataset(
    root=data_root,
    split="train",
    modality="paired",
    transforms=PairedLetterbox(size=640),
)
```

## 9. 可直接发给 ChatGPT 的开场提示词

> 我正在学习 Week08 Day02 Step 1：LLVIP 多模态 Letterbox。请阅读我上传的交接文档，按其中的真实记录讲解。实现和真实样本验证已完成，但请不要默认我已理解代码。先只讲“为什么 1280×1024 变成 640×640 时，要先缩放到 640×512 再上下各填充64”，用本实验的框 [287,428,351,662] 演示同步坐标变换。讲完等我反馈。全程中文，不主动安排测验，不添加随机翻转、不开始训练；我确认后再继续。

如要逐行讲代码，建议同时上传 `paired_letterbox.py`；如要分析图片，上传 `overview.jpg` 或一张 comparison 图；如要核对完整实验，上传 `results.json` 和 `terminal_output.txt`。
