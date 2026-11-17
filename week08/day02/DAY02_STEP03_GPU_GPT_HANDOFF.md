# Week08 Day02 Step3：RTX 3060 GPU 实测学习交接

## 1. 当前完成状态与证据来源

**本次 LLVIP 数据管线 GPU 实测成功：train、val、test 三组 batch 均通过，程序退出码为0。**

实验仅运行既有 `experiment_batch_device.py` 中的预处理、数据传输和少量基础 Tensor 运算。没有创建检测模型，没有模型前向、反向传播、参数更新或训练。

证据分为：

1. 用户在服务器 SSH 终端提供的环境、数据数量与 SHA256 输出。
2. 实验结束后实际查看的 PyCharm SSH 终端：`GPU status=passed`、`ExitCode=0`。
3. 用户明确许可后，通过现有 PyCharm SFTP 取回的原始 `terminal_output.txt` 和 `results.json`。
4. 取回后在本地执行的 JSON、设备、target、CPU结果和保护文件哈希复核。

退出码0来自终端的 `echo ExitCode=%ERRORLEVEL%`，不在 Python 自身的 terminal_output.txt 中。没有把本地 CPU 结果冒充 GPU 运行记录。

学习者为软件工程硕士研一，已学完 Week01–Week07 基础、Day01 数据准备、Day02 Step1 Letterbox 与 Step2 同步水平翻转。请 ChatGPT 全程中文，每次只讲一个明确步骤，不主动安排测验。代码通过不代表学习者已经理解。

## 2. 服务器检查和同步经过

服务器项目：`D:\vision-learning-server`；SSH 用户：`gxbb`；本次使用的地址：`192.168.0.149`。

最初通过命令行执行 SSH 只读检查时，非交互认证失败，报 `Permission denied (publickey,password,keyboard-interactive)`。没有绕过认证或修改服务器配置。

随后使用 PyCharm 已配置的 Laptop 远程文件浏览器检查目录，并由用户在现有 SSH 终端手动执行检查及实验命令。

发现和处理：

- 服务器已经有 `data/LLVIP_1000`；本次没有重复上传该子集或完整原始 LLVIP。
- Day02 的 Letterbox、翻转及 Step3 脚本已经存在。
- `week08/day01/llvip_dataset.py` 缺失；用户手动上传该文件，随后目录显示文件大小为10,043字节。
- 同步后的四个必要 Python 文件 SHA256 均与本地一致。

用户返回的实际数量检查：

| split | RGB | IR | XML |
| --- | ---: | ---: | ---: |
| train | 700 | 700 | 700 |
| val | 150 | 150 | 150 |
| test | 150 | 150 | 150 |

manifest.csv 的 SHA256：

```text
8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb
```

四个必要脚本的本地/远程一致哈希：

```text
llvip_dataset.py
03c2b8fef00f7bc12c68f6284a5b25a5d48003b5f52287211d1b631aa7d31c1a
paired_letterbox.py
43c797a374ccd5189775db18d8663957f5a0f1b79b5f2796585b9fe89698713d
paired_horizontal_flip.py
c68ffec6618c87b732be85d4ad60b5225c699d917b1961534d4add9030b2aa6c
experiment_batch_device.py
b760cca64791f7acdfac073e847b5613574bd89a63a64c157d55e2fd895eb4f4
```

文件数量与清单哈希不等于全量图片/XML内容校验。本次实验实际读取六组样本；JSON记录的44个受保护文件哈希，在运行前后保持一致，且取回后映射到本地对应文件，全部相符。

## 3. 实际 GPU 环境

| 项目 | 实际记录 |
| --- | --- |
| 服务器系统 | Windows 10（项目既有配置） |
| Python解释器 | D:\cv-server-env\Scripts\python.exe |
| Python版本号 | 本次保存的实验日志未记录具体版本号，不补造 |
| PyTorch | 2.14.1+cu126 |
| torchvision | 0.29.1+cu126（用户执行环境检查的真实输出） |
| torch.version.cuda | 12.6 |
| CUDA可用 | True（用户环境检查） |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 目标设备 | cuda:0 |
| 可见设备总显存 | 6,441,926,656 bytes = 6,143.5 MiB |

`torch.version.cuda=12.6` 表示 PyTorch 构建使用的 CUDA 版本；本次没有额外查询驱动版本或系统独立 CUDA Toolkit 版本。

## 4. 代码入口与实际运行命令

复用以下代码，没有重写预处理：

```text
week08/day01/llvip_dataset.py
week08/day02/paired_letterbox.py
week08/day02/paired_horizontal_flip.py
week08/day02/experiment_batch_device.py
```

主数据流程：

```text
LLVIPDataset → PairedLetterboxAndFlip → Subset
→ DataLoader(batch_size=2, collate_fn=detection_collate_fn)
→ 逐样本/逐字段复制到 cuda:0 → 基础运算 → 回传比较
```

用户在服务器 CMD 执行的实验命令：

```cmd
set PYTHONIOENCODING=utf-8
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day02\experiment_batch_device.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output-dir D:\vision-learning-server\week08\day02\outputs\step03_batch_device_gpu --device cuda:0 --seed 42
echo ExitCode=%ERRORLEVEL%
```

固定seed=42，num_workers=0，shuffle=False，pin_memory=False，batch_size=2。
`non_blocking=False`，不改变 dtype；没有开启半精度或AMP。
命令行明确选择 cuda:0，不自动回退CPU。已有输出目录不能覆盖，复现时需改用新的输出目录名称。

## 5. 三组实际 batch 结构

| split | 样本 | boxes形状 | 翻转概率 |
| --- | --- | --- | ---: |
| train | 010001、100030 | (6,4)、(0,4) | 0.5 |
| val | 030001、030014 | (2,4)、(3,4) | 0 |
| test | 190001、190040 | (2,4)、(7,4) | 0 |

100030为合法空标注样本。images和targets都保留为长度2的list，没有假设两个样本具有相同框数量。

| 每个样本的字段 | shape | dtype | 传输前 | 传输后 |
| --- | --- | --- | --- | --- |
| visible | (3,640,640) | float32 | cpu | cuda:0 |
| infrared | (3,640,640) | float32 | cpu | cuda:0 |
| boxes | (N,4) | float32 | cpu | cuda:0 |
| labels | (N,) | int64 | cpu | cuda:0 |
| area | (N,) | float32 | cpu | cuda:0 |
| image_id | (1,) | int64 | cpu | cuda:0 |
| iscrowd | (N,) | int64 | cpu | cuda:0 |

当N=0时，boxes仍为(0,4)，labels/area/iscrowd为(0,)，image_id仍为(1,)；这些空Tensor也验证了设备和dtype。

## 6. 传输和实际 GPU 运算结果

三组结果均为：

```text
transfer_exact = true
basic_ops_close = true
gpu_status = passed
current_device_checks_passed = true
```

纯设备拷贝保持float32/int64，回传后使用 `torch.equal` 精确比较全部Tensor，同时检查shape和dtype。三组全部一致。

基础运算在目标GPU执行：

```python
blended = visible * 0.5 + infrared * 0.5
mean_value = blended.mean()
area = (xmax - xmin) * (ymax - ymin)
```

每个输出实际位于cuda:0；与服务器本次运行内的CPU参考结果用 `rtol=1e-5, atol=1e-6` 比较，全部通过。
逐像素平均只是基础运算示例，不是融合检测模型训练。

取回后，还把GPU结果与此前本地CPU实验作了额外比较：三组的 `targets_cpu`、`targets_after` 与本地原CPU记录完全一致。

以下均值表是“此前本地CPU结果”与“本次服务器GPU结果”的对照；服务器本次运行内的CPU均值未另存为单独字段，不把两者混称为同一机器同一次测量。

| 样本 | 本地CPU均值 | 服务器GPU均值 | 绝对差 |
| --- | ---: | ---: | ---: |
| 010001 | 0.16638202965259552 | 0.1663820594549179 | 2.9802322387695312e-8 |
| 100030 | 0.28582146763801575 | 0.28582149744033813 | 2.9802322387695312e-8 |
| 030001 | 0.28762203454971313 | 0.2876220643520355 | 2.9802322387695312e-8 |
| 030014 | 0.26325833797454834 | 0.2632583677768707 | 2.9802322387695312e-8 |
| 190001 | 0.3289640545845032 | 0.32896408438682556 | 2.9802322387695312e-8 |
| 190040 | 0.33761098980903625 | 0.33761104941368103 | 5.960464477539063e-8 |

这些微小差异发生在浮点运算归约结果中，不能与“纯数据传输是否无损”的检查混淆。

## 7. 真实计时

计时前后已调用 `torch.cuda.synchronize(device)`。JSON的字段名为 `transfer_ms`，终端打印为 `copy_ms`，含义相同。

| split | CPU→GPU copy_ms | GPU basic_ops_ms |
| --- | ---: | ---: |
| train | 4.752100 | 6.420200 |
| val | 3.911600 | 0.508400 |
| test | 3.779500 | 0.652400 |

以上数值直接来自JSON。计时包括Python调度和同步等待，不是单纯CUDA内核时间，也不是稳定的性能基准。
首批运算较慢可能受首次运算初始化等因素影响，但本次没有进一步隔离其原因；不能据此断言瓶颈或计算训练吞吐。

## 8. 真实显存数据

单位全部为bytes；peak从CUDA初始化后重置一次，随后累计跨三个split，不是每批独立峰值。

| 记录阶段 | allocated | reserved | peak_allocated | peak_reserved |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 2,097,152 | 0 | 2,097,152 |
| train传输后 | 19,663,872 | 23,068,672 | 19,663,872 | 23,068,672 |
| train运算后 | 29,495,808 | 44,040,192 | 39,325,696 | 44,040,192 |
| val传输后 | 19,666,944 | 44,040,192 | 39,325,696 | 44,040,192 |
| val运算后 | 29,499,392 | 44,040,192 | 39,328,768 | 44,040,192 |
| test传输后 | 19,666,944 | 44,040,192 | 39,328,768 | 44,040,192 |
| test运算后 | 29,499,392 | 44,040,192 | 39,328,768 | 44,040,192 |
| 最终记录 | 1,024 | 44,040,192 | 39,328,768 | 44,040,192 |

全程峰值allocated约37.506836 MiB，峰值reserved为42 MiB。
最终设备可用显存为5,316,280,320 bytes（5,070 MiB），设备总显存为6,441,926,656 bytes（6,143.5 MiB）。

allocated表示本进程当前活跃Tensor等分配；reserved包含PyTorch缓存池；二者不等于整张显卡的总占用。最终reserved仍为42 MiB，不代表这42 MiB都被活跃Tensor使用。
最终allocated未归零，脚本作用域中仍可能保留标量等引用；本次不是内存泄漏压力测试。
本实验没有模型、梯度或优化器，不能用这个显存峰值推断后续检测训练只需要几十MiB。

## 9. 完整日志、JSON与本地复核

服务器原始结果目录：

```text
D:\vision-learning-server\week08\day02\outputs\step03_batch_device_gpu
```

获准后通过现有PyCharm SFTP“从此处下载”，下载了该目录中仅有的两个文件，之前已检查本地目标目录不存在。

本地保存为：

```text
D:\python\test\vision-learning\week08\day02\outputs\step03_batch_device_gpu\terminal_output.txt
D:\python\test\vision-learning\week08\day02\outputs\step03_batch_device_gpu\results.json
```

文件大小分别为8,420和34,891字节。完整原始日志保留在terminal_output.txt，未用摘要替换。

真实日志关键原文：

```text
Week08 Day02 Step3 | device=cuda:0 | seed=42 | batch_size=2
Python=D:\cv-server-env\Scripts\python.exe; torch=2.14.1+cu126; CUDA build=12.6
实际 GPU=NVIDIA GeForce RTX 3060 Laptop GPU
[train] stems=['010001', '100030']; flip_p=0.5
PASS: 全部 Tensor 保持 shape/dtype/数值；基础运算结果与 CPU 一致
blend_means=[0.1663820594549179, 0.28582149744033813]; copy_ms=4.752; basic_ops_ms=6.420
[val] stems=['030001', '030014']; flip_p=0.0
PASS: 全部 Tensor 保持 shape/dtype/数值；基础运算结果与 CPU 一致
blend_means=[0.2876220643520355, 0.2632583677768707]; copy_ms=3.912; basic_ops_ms=0.508
[test] stems=['190001', '190040']; flip_p=0.0
PASS: 全部 Tensor 保持 shape/dtype/数值；基础运算结果与 CPU 一致
blend_means=[0.32896408438682556, 0.33761104941368103]; copy_ms=3.779; basic_ops_ms=0.652
PASS: 原代码/Step1/Step2产物和所用源数据哈希未变
当前设备验证通过；GPU status=passed
```

取回后本地复核实际输出：

```text
PROTECTED_LOCAL_MATCH_COUNT=44 MISMATCHES=[]
LOCAL_REVIEW_PASSED
```

复核内容包括：三个batch、GPU状态passed、所有传输后Tensor device为cuda:0、target跨CPU/GPU和旧本地记录相同，以及JSON记录的44个文件哈希与本地对应文件一致。
没有发生本次实验中的CUDA OOM或其他报错；初期SSH认证失败、缺少Dataset文件以及用户最初CMD命令拼接错误均已在运行前处理。

## 10. 给 ChatGPT 的原理学习交互

建议依次理解：

1. 为什么检测任务images/targets保留list，空框可以与6个框的样本共存。
2. `.to(device)`必须覆盖RGB、IR和target的每个Tensor，float32/int64无需统一成一种类型。
3. 纯拷贝的精确比较与浮点运算的容差比较有什么区别。
4. GPU异步执行为什么要求同步后计时；本次计时具体包含什么。
5. allocated、reserved、peak及整卡free/total为何不同。
6. 为什么数据管线验证成功不等于检测模型已经训练或性能达标。

可复制给ChatGPT：

> 我已完成Week08 Day02 Step3的RTX 3060真实实验。请根据上传文档和原始结果辅导我，先讲为什么一个batch可以包含(6,4)和(0,4)的boxes，以及如何逐字段把双模态数据移动到cuda:0，讲完等待我反馈。之后再讲精确比较/容差、CUDA同步计时和显存指标。全程中文，不主动测验，不默认我已经掌握，不安排新训练或自动进入Day03。

普通ChatGPT不能通过本地路径自动读取文件。建议上传本文档；逐行讲代码时上传experiment_batch_device.py，需要核对细节时附上GPU的results.json、terminal_output.txt。

当前Day02 Step3的CPU和GPU数据管线验证都已完成。下一步先进行原理学习，再由用户决定后续实验；本次到此停止。
