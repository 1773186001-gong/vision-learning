# Week08 Day03：IR 单模态完整基线训练学习交接

## 1. 实际完成状态与证据

**RTX 3060 Laptop 上已完成冒烟测试、10轮正式训练、验证集选模和一次最终测试。最佳模型为第9轮；程序终端显示 ExitCode=0。**

用户授权“保存”后，通过已配置的 PyCharm Laptop SFTP 将 best、last、训练日志、指标、预测缓存和图片取回本地；未下载十份逐轮权重，服务器原文件保留。本地没有重新训练或重新运行测试集模型推理。

项目：`D:\python\test\vision-learning`。

本地结果目录：`D:\python\test\vision-learning\week08\day03\outputs\ir_baseline_run01`。

服务器结果目录：`D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01`。

证据来源：服务器终端的真实输出、取回的完整启动日志、10行CSV、10份验证JSON、完成标记、测试预测缓存及checkpoint。本地已严格加载best/last、比对checkpoint历史与CSV、从缓存复算指标，并检查10张PNG均可解码。**ExitCode=0来自此前查看的CMD终端；Python日志本身没有保存CMD的echo行。**

以下表格是实际结果的四舍五入摘要，原始精度见CSV/JSON。准备阶段旧文档已单独保存在 `DAY03_IR_PREPARATION_ARCHIVE.md`，不再代表当前状态。

## 2. 实际模型、环境与预训练权重

- 服务器：Windows 10，`DESKTOP-BFIO0RA`，RTX 3060 Laptop GPU，6GB。
- 解释器：`D:\cv-server-env\Scripts\python.exe`；未额外记录Python小版本号。
- PyTorch：`2.14.1+cu126`；torchvision：`0.29.1+cu126`；CUDA构建版本12.6，实际设备`cuda:0`。
- 用户批准安装matplotlib后，服务器实测版本为`3.11.2`；Pillow可用。没有安装torchmetrics或pycocotools，没有重装CUDA/PyTorch。
- 模型：`fasterrcnn_mobilenet_v3_large_320_fpn`，MobileNetV3 Large骨干、FPN、RPN和RoI检测头。使用TorchVision完整COCO预训练检测器，最终`FastRCNNPredictor`替换为背景0/person1两类。
- 使用预训练结构的FrozenBatchNorm；恢复checkpoint时显式重建FrozenBN，并`strict=True`加载，恢复可训练参数设置。

预训练权重来自TorchVision枚举对应的官方地址：

<https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth>

初查未缓存，用户明确授权下载后才获取；实际文件为77,844,807字节，SHA256：

```text
907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5
```

缓存位置：`C:\Users\gxbb\.cache\torch\hub\checkpoints\fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth`。训练脚本要求缓存存在并检查哈希前缀，未使用随机初始化替代预训练。

## 3. 数据、输入与完整训练参数

沿用Day01已完成的LLVIP_1000场景隔离子集；每对样本对应一张IR训练图。原始数据未修改。

| split | 图像对/IR图 | person框 | 场景前缀 |
| --- | ---: | ---: | --- |
| train | 700 | 1,991 | 01,02,04,05,06,07,09,10,11,12,14,15,16,18,25 |
| val | 150 | 416 | 03,08,13,17 |
| test | 150 | 348 | 19,20,21,22,23,24,26 |

划分和框数量沿用Day01记录；本次测试缓存也核实了150图、348个GT框。按场景前缀隔离降低同场景相邻帧跨集合泄漏风险；没有另做图像近重复检索，不能据此宣称绝对不存在所有形式的泄漏。

清单SHA256：`8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb`。

复用原有`LLVIPDataset`与`PairedLetterboxAndFlip`。IR灰度复制为3通道，Tensor为float32 `[3,640,640]`、范围[0,1]；VOC框为float32 `[N,4]` xyxy像素坐标，空标注为`(0,4)`。Dataset仍读取RGB做既有配对处理，但传入模型的只有IR Tensor，没有RGB信息或融合分支。

原图1280×1024经Letterbox得到内容640×512、上下各填64，输出640×640。模型名称虽含320，本实验显式设置内部`min_size=max_size=640`；保留模型归一化，避免默认缩至320。冒烟实测内部尺寸仍是640×640，框坐标不变。训练同步水平翻转概率0.5，val/test为0。

| 参数 | 实际设置 |
| --- | --- |
| Epoch | 10 |
| batch_size / num_workers | 1 / 0 |
| 随机种子 | 42 |
| 优化器 | SGD |
| 初始及全程学习率 | 0.001 |
| momentum / weight_decay | 0.9 / 0.0005 |
| scheduler / AMP | 无 / 关闭 |
| CPU线程数 | 2 |
| 模型选择 | val mAP50:95最大，分数相同保留较早轮 |
| 预测保留 | 每图最多100框，模型分数下限0.001 |
| P/R和案例阈值 | score≥0.5，IoU≥0.5 |

学习率0.001是batch=1下迁移学习的保守起点，没有做搜索，也不声称最优。没有根据test调参。

## 4. 冒烟测试真实结果

正式训练前，在真实正样本和合法空标注样本上分别执行前向、四项Loss、反向传播、有限梯度检查、参数更新检查。随后恢复模型初值并重置种子，冒烟更新不计入正式训练。

| 样本 | total | classifier | box_reg | objectness | rpn_box_reg |
| --- | ---: | ---: | ---: | ---: | ---: |
| train/010001，含框 | 1.518622 | 0.855075 | 0.645549 | 0.009443 | 0.008554 |
| train/100030，空标注 | 0.829882 | 0.785711 | 0.000000 | 0.044170 | 0.000000 |

`resize_boxes_unchanged=true`、`weights_restored_before_training=true`。冒烟峰值allocated为316,531,712字节。空标注没有回归GT，因此此样本两项回归Loss为0；分类与objectness仍有监督。

## 5. 每轮真实Loss与验证指标

Loss为该轮700个batch的算术平均；以下数值来自`training_log.csv`。

| Epoch | total_loss | classifier | box_reg | objectness | rpn_box_reg | val P | val R | val AP50 | val mAP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.740892 | 0.247945 | 0.474815 | 0.011580 | 0.006551 | 0.651072 | 0.802885 | 0.802591 | 0.382391 |
| 2 | 0.766579 | 0.242507 | 0.512537 | 0.006171 | 0.005364 | 0.740426 | 0.836538 | 0.849053 | 0.378580 |
| 3 | 0.764544 | 0.238993 | 0.514671 | 0.005816 | 0.005063 | 0.846154 | 0.793269 | 0.853548 | 0.424030 |
| 4 | 0.734852 | 0.223134 | 0.501761 | 0.005350 | 0.004608 | 0.810875 | 0.824519 | 0.850326 | 0.392710 |
| 5 | 0.763736 | 0.228671 | 0.525881 | 0.004691 | 0.004492 | 0.884211 | 0.807692 | 0.858510 | 0.405513 |
| 6 | 0.726472 | 0.218779 | 0.498696 | 0.004686 | 0.004311 | 0.863184 | 0.834135 | 0.861165 | 0.416356 |
| 7 | 0.741451 | 0.221410 | 0.511747 | 0.004104 | 0.004191 | 0.893333 | 0.805288 | 0.870336 | 0.423296 |
| 8 | 0.708792 | 0.203177 | 0.497912 | 0.003811 | 0.003894 | 0.891534 | 0.810096 | 0.858061 | 0.415099 |
| **9** | **0.759800** | 0.214223 | 0.538070 | 0.003567 | 0.003941 | **0.884319** | **0.826923** | **0.873139** | **0.431061** |
| 10 | 0.700655 | 0.202165 | 0.491308 | 0.003365 | 0.003818 | 0.837709 | 0.843750 | 0.871070 | 0.420892 |

| Epoch | lr | 训练+验证秒数 | peak_allocated 字节 | peak_reserved 字节 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.001 | 124.033809 | 393544192 | 499122176 |
| 2 | 0.001 | 124.846913 | 380265472 | 499122176 |
| 3 | 0.001 | 129.735878 | 374634496 | 499122176 |
| 4 | 0.001 | 132.750577 | 393544192 | 499122176 |
| 5 | 0.001 | 132.463030 | 380579840 | 499122176 |
| 6 | 0.001 | 132.993198 | 376098304 | 499122176 |
| 7 | 0.001 | 136.106997 | 376516096 | 499122176 |
| 8 | 0.001 | 137.040708 | 378291200 | 499122176 |
| 9 | 0.001 | 137.700070 | 369896448 | 499122176 |
| 10 | 0.001 | 136.702706 | 375158272 | 499122176 |

累计1324.373885秒，约22分04秒；每轮计时在CUDA同步后读取，包含训练和该轮验证，**不包含随后checkpoint落盘、曲线绘制、最终test及其他启动工作**。没有记录完整端到端墙钟耗时，不把该总和当成全流程精确耗时或稳定速度基准。

逐轮记录的峰值allocated最大393,544,192字节=375.313 MiB；peak_reserved最大499,122,176字节=476 MiB。它们是PyTorch allocator统计，并非nvidia-smi整卡占用，也不是最终测试阶段的单独显存测量。日志没有OOM、NaN/Inf或异常中断。

![真实训练Loss和验证AP曲线](D:/python/test/vision-learning/week08/day03/outputs/ir_baseline_run01/loss_curve.png)

## 6. 最佳模型与一次最终测试

最佳第9轮依据val mAP50:95选出，与第10轮Loss更低并不矛盾。最佳验证与最终测试结果：

| 集合/模型 | Precision | Recall | AP50 | mAP50:95 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val / epoch9 | 0.884319 | 0.826923 | 0.873139 | 0.431061 | 344 | 45 | 72 |
| test / epoch9 | 0.850416 | 0.882184 | 0.898907 | 0.471626 | 307 | 54 | 41 |

**评估协议边界：**单person类，按图按分数贪心一对一IoU匹配、再全数据集分数排序，101点插值AP；mAP取IoU=0.50:0.05:0.95的10档平均。AP使用模型已保留的低阈值预测；P/R使用固定0.5置信阈值。空GT图上的预测计FP。无COCO crowd/ignore、面积分层及多档maxDets，**这些是项目自定义单类指标，不是官方COCO API完整结果，不直接冒充论文可比成绩**。

test只在10轮结束且best确定后推理一次，`test_started.json`先保存best哈希，`test_predictions.pth`缓存150张图预测及GT；出图复用缓存，不额外运行测试模型。本地复算也只读缓存，没有再次测试推理。test表现高于val只能描述为此次不同场景划分的观测，不能据此推断泛化优势、无泄漏或改参依据。

| 权重 | 用途 | 字节数 | 本地严格加载 |
| --- | --- | ---: | --- |
| best.pth | epoch9，最终评估模型 | 151529396 | 通过 |
| last.pth | epoch10，最后完整训练状态 | 151529588 | 通过 |

SHA256：

```text
best.pth  9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2
last.pth  d1055247066cdd8e5c4e565e58419440b5a028a0042804feaf583838f02263f2
```

best哈希与服务器生成的test_started记录一致；last的上述哈希为本地计算值，未额外获取服务器last哈希作逐字节对照。两文件历史、优化器状态和epoch已核实。

## 7. 真实预测图片

图中绿色为GT、红色为score≥0.5预测，标注person和分数。路径均相对于本地结果目录：

- 成功：`prediction_examples/success_190001.png`、`success_190159.png`、`success_190199.png`。
- 漏检：`prediction_examples/miss_190040.png`、`miss_190080.png`、`miss_190119.png`。
- 误检：`prediction_examples/false_positive_190040.png`、`false_positive_190474.png`、`false_positive_190513.png`。
- 案例清单：`prediction_examples/selection.json`。

取每一类符合条件的前3个案例，没有人为制造失败图。190040同时有漏检和误检，因此出现于两类；检测评估的FP可以来自重复框或框定位不足，不一定是完全没有行人的位置。已实际查看曲线、成功190001、漏检190040、误检190474；全部9张预测图及曲线通过PNG解码检查。

## 8. 代码和产物说明

| 文件 | 作用 |
| --- | --- |
| ir_common.py | 复用IR Dataset/变换、自定义collate、设备传输、模型、缓存权重检查、指标、checkpoint恢复、曲线 |
| train_ir.py | 冒烟→10轮训练→val选模→一次最终test→缓存出图；异常停止和断点状态保存 |
| evaluate_ir.py | 独立val评估；最终test函数由完整流程调用 |
| predict_ir.py | 使用测试缓存选择成功/漏检/误检并出图，不重新推理 |
| preflight_ir.py | 只读检查服务器解释器、CUDA、文件及预训练缓存 |
| check_day03.py | 本地语法、合成指标边界与真实数据预处理检查，非模型训练 |
| plot_training.py | 从真实CSV补画曲线的可选工具；本次服务器已生成曲线 |
| README.md | 配置、运行方式、恢复机制及评估边界 |

取回的完整启动日志：

`D:\python\test\vision-learning\week08\day03\outputs\ir_baseline_run01_launch_1793326678503699800.log`

结果目录中还包含`console_1793326681737098000.txt`（正式逐轮日志，冒烟之前的信息见完整启动日志）、`config.json`、`smoke_test.json`、`training_log.csv`、`val_epoch_01.json`至`val_epoch_10.json`、`loss_curve.png`、`best.pth`、`last.pth`、`test_started.json`、`test_metrics.json`、`test_predictions.pth`、`completed.json`、`prediction_examples/`，以及服务器自动生成的原始结果交接文档。原始结果交接文档保留，本文是补充经过核对的完整学习交接。

额外保存本地复核记录`local_result_review.json`和取回文件的SHA256清单`artifact_sha256.json`。本地SHA清单用于后续检查文件是否变化，不代表已将每个文件与服务器逐一远程校验。

## 9. 实际运行命令与复现

以下在**服务器CMD**执行，`cd`与Python命令分开输入；不要将两个路径拼成一条`cd`命令。

本次实际完整训练命令：

```cmd
set PYTHONIOENCODING=utf-8
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\train_ir.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01 --device cuda:0
echo ExitCode=%ERRORLEVEL%
```

日志关键真实输出（省略冒烟详情，其完整数值见启动日志及第4节）：

```text
epoch=1 batch=100/700 loss=0.956828
epoch=10 batch=700/700 loss=0.637787
COMPLETED {'precision': 0.850415512465374, 'recall': 0.882183908045977, 'AP50': 0.8989071207511944, 'mAP50_95': 0.4716263012372224, 'TP': 307, 'FP': 54, 'FN': 41, 'score_threshold': 0.5, 'protocol': 'single_person_101pt_all_area_max100_no_ignore; not official COCO API'}
```

如以后获准重新训练，输出必须改为**新的空目录**，例如`ir_baseline_run02`；不要重跑到已有run01。下列工具命令为以后复现参考，本次取回后未执行新的val推理或训练。

独立验证集评估（输出文件必须尚不存在）：

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\evaluate_ir.py --checkpoint D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\val_review.json --device cuda:0
```

本地从已保存预测缓存重新出图（不再次测试推理，输出目录必须不存在）：

```powershell
Set-Location D:\python\test\vision-learning
& D:\python\.venv\Scripts\python.exe -B .\week08\day03\predict_ir.py --data-root .\data\LLVIP_1000 --predictions .\week08\day03\outputs\ir_baseline_run01\test_predictions.pth --output .\week08\day03\outputs\ir_baseline_run01\prediction_examples_review
```

断点恢复入口（**仅用于尚未完成10轮、且test尚未启动的同一run**）：

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\train_ir.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01 --device cuda:0 --resume D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\last.pth
```

本次run01已完成，恢复命令会被脚本拒绝，不应执行。checkpoint保存模型、优化器、可训练参数名、历史以及Python/Torch/CUDA/DataLoader随机状态，从最后完整epoch恢复；本次没有发生中断，**没有实测中断后续训**，仅验证checkpoint可加载且状态存在。模型checkpoint含Python状态，仅加载本项目可信文件。

## 10. 实际问题、结论与学习请求

实际遇到：服务器未缓存权重、缺少matplotlib，均先获得用户许可，再由用户安装/下载；CMD中曾把`cd`和解释器路径拼接导致找不到路径，后改为独立Python命令；命令行SSH认证不可用时未绕过认证，使用用户现有终端和已配置SFTP完成流程。数据未重复上传完整LLVIP。正式训练中没有OOM、非有限数值或失败记录，没有静默降尺寸、改batch或重试。

本地复核没有执行模型前向/反向；它验证了10轮记录、best/last严格加载、历史与CSV一致、best哈希与测试标记一致、150图测试缓存指标一致、10张PNG可解码，退出码0。结果审查没有修改Day01/Day02或原始数据。

观测：classifier与RPN相关Loss下降，box_reg有波动；total并非单调下降。第10轮total最低但val mAP低于第9轮，说明应按预设验证指标选模。AP50明显高于多IoU平均AP，提示需要关注更严格IoU下的定位表现；这只是当前结果的分析方向，尚未开展尺寸分组、定位误差归因等额外实验。

请ChatGPT结合本代码和真实数值，逐步讲解：

1. Faster R-CNN的骨干、FPN、RPN、RoI分别做什么，四项Loss各监督什么。
2. 空标注样本为何仍有分类/objectness Loss，以及如何参与训练。
3. IR复制3通道与COCO预训练的关系；为什么这仍是IR单模态实验。
4. 外部640 Letterbox与内部resize、归一化、框坐标的关系。
5. 为什么训练Loss最低不等于最佳验证模型，为什么test不参与调参。
6. 使用344/45/72和307/54/41解释P/R；AP50与mAP50:95及分数阈值的区别。
7. 查看成功、漏检、误检真实图，理解定位不足、重复框和小/遮挡目标可能如何影响指标，区分观察与尚未验证的推测。
8. allocated/reserved/peak的含义，以及checkpoint如何保存可恢复状态。

学习者为软件工程硕士研一；代码执行成功不代表已经理解。请全程中文，每次只讲一个明确步骤，等待反馈，不主动安排测验。普通ChatGPT不能仅凭本地路径读取文件，需要时请我上传相关代码、CSV或图片。

**当前完成Week08 Day03 IR基线；下一步先学习和分析本交接，不自动进入RGB、RGB+IR、YOLO或新的训练。**
