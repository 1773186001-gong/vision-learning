# Week08 Day02 Step3：DataLoader 批次与 CPU/GPU 传输交接

## 1. 当前学习状态

我是软件工程硕士研一学生，已完成 Week01–Week07 的视觉/深度学习基础、Day01 的 LLVIP 数据准备，以及 Day02 Step1 Letterbox 和 Step2 同步水平翻转。

本次 **Step3 本地 CPU 实验已完成，RTX 3060 GPU 验证待完成**。
没有获得本次远程连接、上传或执行的确认，所以没有连接服务器，也未检查其数据目录。
旧 PROJECT_HISTORY 中的 CUDA 验证属于历史记录，不能作为本次 LLVIP 批次 GPU 实验结果。

请 ChatGPT 全程中文，每次只讲一个明确步骤，等待反馈，不主动安排测验。不要默认代码运行通过就意味着我已理解。

## 2. 本次新增的文件

项目根目录：`D:\python\test\vision-learning`。

| 文件 | 用途 |
| --- | --- |
| week08/day02/experiment_batch_device.py | 真实 DataLoader、CPU/CUDA 传输、结构/数值检查、基础运算、同步计时及显存记录 |
| week08/day02/STEP03_README.md | 实验入口和本地命令 |
| week08/day02/DAY02_STEP03_GPT_HANDOFF.md | 本交接文档 |
| week08/day02/outputs/step03_batch_device_cpu/terminal_output.txt | 完整真实 CPU 日志 |
| week08/day02/outputs/step03_batch_device_cpu/results.json | 全部字段结构、真实 target 数值、计时、环境及保护文件哈希 |

复用的既有代码：

- `week08/day01/llvip_dataset.py` 中的 LLVIPDataset 和 detection_collate_fn。
- `week08/day02/paired_horizontal_flip.py` 中的 PairedLetterboxAndFlip。
- 上述组合继续使用原有 paired_letterbox.py。

没有重写图像预处理，没有修改 Day01、Step1、Step2 或其已有实验结果。
没有安装依赖、修改原始数据、训练模型或使用 YOLO。

## 3. 实际批次结构

batch_size=2、num_workers=0、shuffle=False、pin_memory=False、固定 seed=42。
用 Subset 选择真实样本：

| split | 样本编号 | 两张图的框数量 | 翻转配置 |
| --- | --- | --- | --- |
| train | 010001、100030 | 6、0 | p=0.5 |
| val | 030001、030014 | 2、3 | p=0 |
| test | 190001、190040 | 2、7 | p=0 |

train 中的 100030 是合法空标注样本。每个 batch 保留如下结构：

```python
images = [
    {"visible": rgb0, "infrared": ir0},
    {"visible": rgb1, "infrared": ir1},
]
targets = [target0, target1]
```

这不是一个默认 stack 的 RGB/IR 大张量。图像尺寸相同，但每张图的框数量不同，因此继续使用原有 collate_fn 保留列表。

训练 batch 的真实字段：

| 字段 | 样本0 | 样本1 | dtype | 本次实测 device |
| --- | --- | --- | --- | --- |
| visible、infrared | 各(3,640,640) | 各(3,640,640) | float32 | cpu |
| boxes | (6,4) | (0,4) | float32 | cpu |
| labels | (6,) | (0,) | int64 | cpu |
| image_id | (1,) | (1,) | int64 | cpu |
| area | (6,) | (0,) | float32 | cpu |
| iscrowd | (6,) | (0,) | int64 | cpu |

## 4. Tensor 如何移动到目标设备

关键实现：

```python
moved_images = [
    {k: v.to(device, non_blocking=False, copy=True) for k, v in image.items()}
    for image in images
]
moved_targets = [
    {k: v.to(device, non_blocking=False, copy=True)
        if isinstance(v, torch.Tensor) else copy.deepcopy(v)
     for k, v in target.items()}
    for target in targets
]
```

逐样本、逐字段移动，不把检测框强行堆成统一长度。没有改变 dtype，也没有开启半精度。
copy=True 使 CPU→CPU 验证也执行一次独立拷贝，而不只是返回同一个对象。

纯 float32→float32 或 int64→int64 的设备拷贝没有数学运算舍入，因此拷回 CPU 后用 torch.equal 精确比较，同时检查 shape/dtype。空 Tensor 也参与检查。

涉及基础运算（包括 mean 归约）时，CPU/CUDA 计算次序可能不同，采用 `rtol=1e-5, atol=1e-6`。
不能把所有浮点比较都机械地设成零容差，也不能对本应无损的纯拷贝用宽松容差掩盖错误。

## 5. 基础运算和 CUDA 检查设计

脚本在目标设备执行少量真实图像 Tensor 运算：

```python
blended = visible * 0.5 + infrared * 0.5
mean_value = blended.mean()
area = (xmax - xmin) * (ymax - ymin)
```

这里的平均只是验证运算的练习，不是训练融合模型，也没有测量检测性能。
结果会与 CPU 参考运算比较，确认计算输出确实位于指定设备。

CUDA 模式：

1. 检查实际解释器为 `D:\cv-server-env\Scripts\python.exe`。
2. 检查 CUDA 可用，并实际获取 GPU 名称，要求名称包含 RTX 3060。
3. 执行一个小 Tensor 运算初始化 CUDA，随后同步。
4. 重置本进程显存峰值记录。
5. 在传输计时前后执行 `torch.cuda.synchronize(device)`。
6. 在基础运算计时前后再次同步；计时区间不包含参考 CPU 运算及逐字段打印。
7. 记录 baseline、传输后、运算后的显存及整个三批次实验的峰值。

记录项为 allocated、reserved、peak_allocated、peak_reserved，以及设备总/可用显存（单位 byte）。
allocated/reserved 属于本进程 PyTorch 分配器统计，不等于显卡所有进程的总占用；设备总/可用显存会受系统和其他进程影响。
峰值在 CUDA 初始化后、三个 split 开始前重置一次，不是每个 split 分别重置。
小规模单次计时仅用于验证计时逻辑，不是稳定性能基准。

CUDA 模式不自动回退 CPU。CPU 模式不调用 CUDA 显存统计，也不假装已经传到 GPU。

## 6. 本地实际命令与结果

实际执行：

```powershell
Set-Location D:\python\test\vision-learning
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day02\experiment_batch_device.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day02\outputs\step03_batch_device_cpu `
  --device cpu `
  --seed 42
```

退出码=0，环境 `torch=2.14.1+cpu`、`CUDA build=None`。一次脚本总耗时约8秒。

真实日志摘录：

```text
Week08 Day02 Step3 | device=cpu | seed=42 | batch_size=2
Python=D:\python\.venv\Scripts\python.exe; torch=2.14.1+cpu; CUDA build=None
CPU 逻辑验证；GPU 名称、显存、CUDA 传输/计算均待服务器实测
[train] stems=['010001', '100030']; flip_p=0.5
PASS: 全部 Tensor 保持 shape/dtype/数值；基础运算结果与 CPU 一致
blend_means=[0.16638202965259552, 0.28582146763801575]; copy_ms=2.498; basic_ops_ms=2.915
[val] stems=['030001', '030014']; flip_p=0.0
blend_means=[0.28762203454971313, 0.26325833797454834]; copy_ms=2.467; basic_ops_ms=2.462
[test] stems=['190001', '190040']; flip_p=0.0
blend_means=[0.3289640545845032, 0.33761098980903625]; copy_ms=2.338; basic_ops_ms=2.043
PASS: 原代码/Step1/Step2产物和所用源数据哈希未变
当前设备验证通过；GPU status=pending_not_run_cpu
```

上述时间均是本地 CPU 时间，不是主机到 GPU 传输时间。
实际通过：三个真实 batch 的结构、dtype、CPU设备、Tensor 精确拷贝、空标注、运算容差比较、同种子复现，以及 val/test 重复读取不随机变化。
保护哈希覆盖 Day01 Python/Markdown、指定的 Step1/Step2代码及交接文档、两步的既有结果文件、manifest/summary 和这六组源 JPG/XML。没有声称全量重新校验所有源文件。

## 7. GPU 及服务器状态：待确认、未执行

| 项目 | 当前状态 |
| --- | --- |
| 服务器 D:\vision-learning-server\data\LLVIP_1000 是否存在 | 未连接，未检查 |
| 数据/代码上传 | 未执行 |
| 本步骤 RTX 3060 GPU 名称实测 | 待完成 |
| CPU→CUDA Tensor 拷贝实测 | 待完成 |
| CUDA 基础运算实测 | 待完成 |
| GPU allocated/reserved/peak 显存 | 待完成，JSON 中为 null，不是0 |
| GPU copy_ms/basic_ops_ms | 待完成，无实测数值 |

本步骤唯一的执行前置条件是用户对远程操作的确认，来自本次请求及 AGENTS.md。
CUDA 分支已编写，但未经 RTX 3060 实际执行；CPU 通过不能证明 CUDA 分支通过。

## 8. 确认后的服务器目录检查与安全 SFTP 方案

以下仅为待执行方案，本次没有执行任何远程命令。

### 第一阶段：用户确认后，只读检查

先核对当前服务器地址（历史IP为192.168.0.149，可能变化）。经用户确认远程连接后，在服务器 PowerShell 只读执行：

```powershell
Test-Path -LiteralPath 'D:\vision-learning-server\data\LLVIP_1000' -PathType Container
Test-Path -LiteralPath 'D:\vision-learning-server\data\LLVIP_1000\manifest.csv' -PathType Leaf
Test-Path -LiteralPath 'D:\cv-server-env\Scripts\python.exe' -PathType Leaf
```

若目录存在，先检查结构、数量、manifest 和待用文件，不直接覆盖或重复上传。
若不存在，明确报告缺失，再在上传获准后同步子集。

### 第二阶段：数据上传需用户确认

使用已配置的 PyCharm SFTP，先 Preview/Compare，再选择明确的子目录或文件 Upload，不运行全项目自动同步。

数据仅同步：

```text
本地 D:\python\test\vision-learning\data\LLVIP_1000
  → 服务器 D:\vision-learning-server\data\LLVIP_1000
```

本地本次只读盘点：3,002 个文件，274,200,584 bytes（约261.5 MiB）；包含2,000张JPG、1,000份XML、manifest.csv、summary.json。
不要同步完整原始 `LLVIP`、`.venv`、其他数据、旧模型或日志，不使用远端删除/镜像清理选项。
如果目标已经有不同内容，先比较差异并报告，不自动覆盖已通过的实验。

代码只需要以下四个文件（保持对应的服务器项目相对路径）：

```text
week08/day01/llvip_dataset.py
week08/day02/paired_letterbox.py
week08/day02/paired_horizontal_flip.py
week08/day02/experiment_batch_device.py
```

既有同内容代码无需重新上传；不同版本先比较。按需额外上传本交接文档和 STEP03_README.md。

上传后只读核对 train/val/test 的 RGB、IR、XML 数量均为700/150/150，且文件均可读取。
manifest 的本地 SHA256 为：

```text
8BFEFE942C968A7BBA05D694E5C6D25548F29D4BE35E5F91874FFAA274ECD4EB
```

服务器可用 `Get-FileHash -LiteralPath 'D:\vision-learning-server\data\LLVIP_1000\manifest.csv' -Algorithm SHA256` 比较清单。
这只是清单一致性检查，不代替图片/XML 的完整校验。

### 第三阶段：执行 GPU 实验也需确认

目录、代码和解释器确认后，使用服务器专用虚拟环境，待执行命令为：

```powershell
Set-Location D:\vision-learning-server
$env:PYTHONIOENCODING='utf-8'
& D:\cv-server-env\Scripts\python.exe -B .\week08\day02\experiment_batch_device.py `
  --data-root D:\vision-learning-server\data\LLVIP_1000 `
  --output-dir D:\vision-learning-server\week08\day02\outputs\step03_batch_device_gpu `
  --device cuda:0 `
  --seed 42
```

这是批次与基础运算验证，不启动训练管理器，不创建模型、不反向传播。
GPU 结果保存为独立目录，不能覆盖本地 CPU 结果或旧日志。若目录已存在，选一个新名称。
结果取回本地也应先获远程文件操作授权，并保存到独立目录，再据真实日志补充交接状态。

## 9. 给 ChatGPT 的学习开场提示词

> 我已完成 Week08 Day02 Step3 的本地 CPU DataLoader 与 Tensor 传输逻辑验证，RTX 3060 服务器未获得本次远程操作确认，GPU实测仍待完成。请根据交接文档讲解，不要编造GPU名称、显存或CUDA运行结果。先解释为何检测任务要保留images/targets列表、为什么两个样本的boxes可以是(6,4)和(0,4)，讲完等待我反馈；后续再解释逐字段.to(device)、纯拷贝精确比较、浮点运算容差、CUDA同步和allocated/reserved/peak的区别。全程中文，不主动安排测验，不连接服务器、不训练模型或开始其他步骤。

普通 ChatGPT 不能凭本地路径访问文件；可上传本文档和 experiment_batch_device.py。需要核对真实输出时同时上传 CPU 的 terminal_output.txt 和 results.json。
当前获准工作到此结束；下一步先学习原理，远程验证留待明确确认。
