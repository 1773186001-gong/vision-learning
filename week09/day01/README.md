# Week09 Day01：Faster R-CNN RPN Proposal Recall 诊断

本实验只回答一个问题：融合模型在高 IoU 下的定位差异，是否已经出现在 RPN 候选框中。使用 Week08 的 IR、Fusion 已训练 `best.pth`，只读取 LLVIP_1000 的 **validation split**，不训练、不更新优化器、不重新评测 test，不修改 Week08 代码或模型权重。

本文件中的命令是运行说明，不能作为执行完成的证据。实际执行范围、成功或失败状态及指标，以各次新输出目录中的 `run.log`、`rpn_metrics.json` 和中文报告为准。`--limit 3` 只是冒烟验证，不代表整个验证集结果。

## 1. 已确认的模型和数据

本地项目根目录：`D:\python\test\vision-learning`。

| 项目 | 实际来源 |
| --- | --- |
| IR 构造与恢复 | `week08/day03/ir_common.py` 的 `build_model(False)`、`checkpoint_model()` |
| IR checkpoint | `week08/day03/outputs/ir_baseline_run01/best.pth` |
| Fusion 构造与恢复 | `week08/day05/fusion_model.py` 的 `build_model(False)`、`restore()` |
| Fusion checkpoint | `week08/day05/outputs/fusion_concat_run01/best.pth` |
| 数据集与配对 | `week08/day01/llvip_dataset.py`，`data/LLVIP_1000/manifest.csv` |
| 同步几何预处理 | `week08/day02/paired_letterbox.py`、`paired_horizontal_flip.py` |
| 训练与验证协议 | `week08/day03/ir_common.py`、`week08/day05/fusion_common.py` |
| 历史交接 | `week08/day03/DAY03_IR_TRAINING_GPT_HANDOFF.md`、`week08/day05/DAY05_FUSION_TRAINING_GPT_HANDOFF.md` |
| 历史定位分析 | `week08/day05/localization_analysis/DAY05_LOCALIZATION_ANALYSIS_GPT_HANDOFF.md` |

两份 checkpoint 都是含 `model` 状态字典及训练信息的字典，由原有恢复函数严格加载；不以随机权重替代，也不下载预训练权重。本实验重新构造的非参数配置来自原模型代码与当前 TorchVision，运行时须记录实际值，不能仅靠 checkpoint 文件推断。

Fusion 使用独立 RGB、IR MobileNetV3+FPN，逐尺度 Concat+1×1 Conv，随后共享 RPN 和 RoI Head。其实际特征键为 **`0`、`1`、`pool`**，RoI 使用 **`0`、`1`**；不改成 P2/P3/P4/P5。外部输入是含 `visible`、`infrared` 的配对字典，内部将两个三通道图像暂时拼成六通道，统一几何变换后再送入两个独立分支。

实际 manifest 的划分为：

| split | 图像对数 | person GT | 场景 |
| --- | ---: | ---: | --- |
| train | 700 | 1991 | 01、02、04、05、06、07、09、10、11、12、14、15、16、18、25 |
| val | 150 | 416 | 03、08、13、17 |
| test | 150 | 348 | 19、20、21、22、23、24、26 |

val 来自官方 train 场景，test 使用官方 test 场景，场景之间隔离。本地检查时 val 没有空 GT 图片；程序仍须安全处理空 GT。运行只按 manifest 的固定顺序读取 val，不重新划分数据。

本地检查的 manifest SHA256 为：

```text
8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb
```

服务器是否一致应以服务器本次运行记录的校验值确认。

## 2. 预处理和坐标系

沿用原 Dataset：RGB 转三通道 RGB；IR 转灰度并复制为三通道；像素为 `[0,1]` 的 float32。RGB、IR、VOC XML 必须同名配对且尺寸一致。VOC 框直接沿用原代码中的像素 `xyxy` 约定，不额外做减一操作。

验证集使用 `PairedLetterboxAndFlip('val')`：两模态同步 Letterbox 至 640×640，填充值为 114/255，不进行随机翻转。本地全部 val 原图为 1280×1024，因此实际缩放为 640×512、上下各补 64 像素，GT 同步变换为 `x'=0.5x`、`y'=0.5y+64`。

继续执行原模型内部 TorchVision transform，包括原归一化、resize 和 batch padding。模型 `min_size=max_size=640`，本数据的内部 resize 不改变 640×640 的几何尺寸；Fusion 的 mean/std 沿用原实现，分别应用于两组三通道。

脚本执行完整的原模型前向 `model(images, targets)`，通过临时只读 forward hooks 取得：

1. detector transform 返回的 GT 和 `image_sizes`；
2. RPN 返回的后处理候选框。

IR 的 detector 是模型自身，Fusion 的 detector 是 `model.detector`。GT 与 proposals 都在 **模型内部 transform 后的坐标系**中计算 IoU；不使用最终 RoI 检测框，不混用原图、Letterbox 和最终输出坐标。运行使用 `model.eval()`、`torch.inference_mode()`，结束后移除 hooks。

## 3. TopK 的实际含义和原模型上限

当前原构造器 `fasterrcnn_mobilenet_v3_large_320_fpn` 的测试配置为：

| RPN 参数 | 原值 |
| --- | ---: |
| `pre_nms_top_n['testing']` | 150（每个特征层的 NMS 前筛选） |
| `post_nms_top_n['testing']` | 150（每张图最终 RPN 上限） |
| `score_thresh` | 0.05 |
| `nms_thresh` | 0.7 |

以每次恢复后实际读取的配置为准。脚本记录配置，不提高原预算来凑出 300 或 1000 个框。`box_detections_per_img=100` 是最终 RoI 检测框上限，不能把它当作 RPN 上限。

TopK 是原 RPN 已完成解码、裁边、过滤、按层 NMS 和最终截断后，按其原始 objectness 分数从高到低排列的前 K 个框。TorchVision 原 `batched_nms` 返回按分数降序的索引，RPN 按此顺序返回 proposals；不按框面积或最终分类分数重新排序。

本次保留 TopK=100、300、1000 三档。当某图实际返回 N 个 proposals 时使用 `min(K,N)` 个，并记录实际数量和不足 K 的图片数。**原上限为150时，Top300 与 Top1000 都使用该图全部实际 proposals，两档结果应相同；它们不是独立的300框/1000框预算实验。** Top100 也可能因实际不足100而使用全部候选框。

## 4. 指标定义

对图像 i 的第 j 个有效 GT，计算其与所取 proposals 的最大 IoU：

```text
max_iou(i,j,K) = max(IoU(GT(i,j), proposal))
```

若该图没有 proposal，该 GT 的最大 IoU 记为0。一个 proposal 可以覆盖多个 GT；这项覆盖诊断不进行检测 AP 使用的贪心一对一匹配。

对阈值 t=0.50、0.75，全局指标为：

```text
Proposal Recall@K(t) = 全部图像中 max_iou >= t 的 GT 数 / 全部有效 GT 数
```

**分母是全部有效 GT 总数，不是逐图 Recall 的算术平均。** GT 最大 IoU 的总体均值、分位数及分布也按所有 GT 汇总。完整 val 预期检查到150图、416个GT；实际数量以读入与校验结果为准。

逐图表同时保留每图 GT 数、原始 proposal 数、各 K 实际使用数、覆盖 GT 数、Recall 和 Fusion−IR 差异。空 GT 图记录 proposal 数并计入图片统计，其 Recall 留空，不参与 Recall 分母；如果整次运行没有有效 GT，应明确报告指标不可定义，不能除零或伪造为0。

Proposal Recall 衡量候选框对 GT 的覆盖能力，**不是最终检测 Recall，也不是 AP75**。它没有最终分类、最终框回归和全局预测排序等环节。

## 5. 运行命令

先进行3张固定验证样本的冒烟，再运行完整 validation split。输出目录必须不存在；每次使用新的时间戳，保留历史输出。`--limit 0` 或省略 `--limit` 表示完整150张，非零值取 manifest 顺序前 N 张。默认随机种子42；`--threads` 控制 CPU 线程数。

### 本地 CPU 冒烟：3张

在 PowerShell 执行：

```powershell
Set-Location 'D:\python\test\vision-learning'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
& 'D:\python\.venv\Scripts\python.exe' -B '.\week09\day01\rpn_diagnostic.py' `
  --device cpu --threads 2 --seed 42 --limit 3 `
  --data-root 'D:\python\test\vision-learning\data\LLVIP_1000' `
  --ir-checkpoint 'D:\python\test\vision-learning\week08\day03\outputs\ir_baseline_run01\best.pth' `
  --fusion-checkpoint 'D:\python\test\vision-learning\week08\day05\outputs\fusion_concat_run01\best.pth' `
  --output "D:\python\test\vision-learning\week09\day01\outputs\cpu_smoke_$runTag"
```

### 服务器 GPU 冒烟：3张

以下命令在代码已同步的服务器 PowerShell 中执行，解释器固定为 `D:\cv-server-env\Scripts\python.exe`，不使用项目里同步过去的 `.venv`。命令本身不会训练；本 README 不表示已经连接服务器或执行诊断。

```powershell
Set-Location 'D:\vision-learning-server'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
& 'D:\cv-server-env\Scripts\python.exe' -B '.\week09\day01\rpn_diagnostic.py' `
  --device cuda:0 --threads 2 --seed 42 --limit 3 `
  --data-root 'D:\vision-learning-server\data\LLVIP_1000' `
  --ir-checkpoint 'D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth' `
  --fusion-checkpoint 'D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth' `
  --output "D:\vision-learning-server\week09\day01\outputs\gpu_smoke_$runTag"
```

### 服务器完整 validation split

GPU 冒烟成功后执行，新建独立输出目录：

```powershell
Set-Location 'D:\vision-learning-server'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
& 'D:\cv-server-env\Scripts\python.exe' -B '.\week09\day01\rpn_diagnostic.py' `
  --device cuda:0 --threads 2 --seed 42 `
  --data-root 'D:\vision-learning-server\data\LLVIP_1000' `
  --ir-checkpoint 'D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth' `
  --fusion-checkpoint 'D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth' `
  --output "D:\vision-learning-server\week09\day01\outputs\val_rpn_$runTag"
```

## 6. 输出与验收

每次运行写入自己的新输出目录：

| 文件 | 内容 |
| --- | --- |
| `rpn_metrics.json` | 执行范围、实际环境与命令、输入校验、两模型配置、各K Recall与最大IoU统计、配对差异 |
| `rpn_per_image.csv` | 逐图 proposal 数、GT 数、各K覆盖与Recall、IR/Fusion差异 |
| `rpn_per_gt.csv` | IR/Fusion长表：逐GT各K最大IoU、是否达到0.50/0.75，可按`stem`+`gt_index`配对计算Fusion−IR差异 |
| `proposal_recall_comparison.png` | IR/Fusion在各K及两个IoU阈值下的Recall比较 |
| `max_iou_distribution.png` | IR/Fusion的GT最大IoU分布 |
| `rpn_diagnostic_report.md` | 中文结果、预算限制、观测与解释、后续建议 |
| `run.log` | 实际命令、运行进度、真实输出和错误信息 |

记录 Python/PyTorch/TorchVision、系统、设备/GPU、随机种子、确定性设置、输入顺序、模型配置、checkpoint绝对路径与SHA256、manifest及相关源码校验。核对运行前后模型状态与checkpoint校验，保证没有更新参数或持久权重。失败要保留真实错误和未完成状态；不要从不完整运行生成“全验证集完成”的结论。

JSON 中的主要结构为 `run`、`inputs`、`models`、`comparison`。`models.IR` 和 `models.Fusion` 各有 `rpn_config`、`top_k`；`top_k` 的键为 `100`、`300`、`1000`，各档含 `recall`、`mean_max_iou` 和 `max_iou_statistics`。以实际生成文件为准。

`rpn_per_image.csv` 是包含两模型结果和直接差值的逐图宽表；`rpn_per_gt.csv` 按模型分别存放记录，不直接存储逐GT差值。逐GT比较时先以 `stem`、`gt_index` 配对，并核对GT坐标一致，再计算各K的Fusion−IR最大IoU差异。

验收时应核对：完整运行是否确为 val 150图/416GT；两模型读到的GT与顺序是否一致；proposal 数是否遵守原上限；Top300/1000是否相同；各K Recall是否随K非递减；全局覆盖计数是否可由逐GT表复算；模型状态和checkpoint校验是否保持一致。

## 7. 结论边界与 Day02 建议

报告分开写“真实观测”“合理假设”“仍需验证的机制”。只给数值差异及其覆盖的 GT 数，不凭一个阈值或少量样本宣称显著性或确定根因。

- **Fusion在IoU=0.75下RPN覆盖明显较弱：** 可以认为RPN输出阶段已经出现质量差异；Day02先在相同val上查看Fusion失去覆盖、IR仍覆盖的GT及候选框，按目标大小、场景和候选排名定位差异，并追踪这些GT经过RoI后的变化。不能由此认定RPN是唯一根因，也不能排除上游融合特征的作用。
- **两模型RPN覆盖接近，但最终检测AP75差异大：** Day02优先追踪RPN已经覆盖的同一批val GT，比较RoI框回归前后IoU、最终分类分数和排序、最终NMS/截断，检查候选框质量是否在后续环节发生变化。
- **不同K、不同阈值或GT组结果混合：** 先报告混合现象；Day02按逐GT配对差异检查具体案例，不能仅凭本实验确定根因。

Day02仍固定同一验证集和checkpoint，不重新训练，不用test反复选参数。Week08既有test结果只作为提出问题的历史背景；本次val RPN结果不能与test AP75直接视为同一数据集上的因果证据。
