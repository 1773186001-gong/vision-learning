# Day03：完整 IR Faster R-CNN 基线

已在RTX 3060 Laptop完成冒烟测试、10轮正式训练、验证选模及一次最终测试，终端退出码0。最佳为第9轮，test AP50=0.898907、mAP50:95=0.471626（下述自定义单类协议）。用户许可后已取回best/last、日志、CSV、JSON、预测缓存和图片至`outputs/ir_baseline_run01`。完整结果及学习说明见[正式交接文档](DAY03_IR_TRAINING_GPT_HANDOFF.md)。远程操作仍须按项目要求确认。

## 实验设置

- 模型：TorchVision fasterrcnn_mobilenet_v3_large_320_fpn，COCO_V1全检测器预训练，预测头替换为背景/person两类。
- IR灰度复制成3通道；旧Dataset仍读取RGB以完成配对检查，但其RGB像素不会返回给模型。
- 原有Letterbox到640×640；模型min_size=max_size=640，只有原有ImageNet归一化，不再缩小到320。
- train/val/test=700/150/150。训练p=0.5同步翻转，val/test p=0。合法空框参与训练。
- SGD：lr=0.001，momentum=0.9，weight_decay=0.0005，无调度、无AMP。batch_size=1，num_workers=0，seed=42，10轮。
- lr=0.001是小batch迁移学习的保守初值，不是已通过搜索得到的最优参数。十轮内不自动调参。
- 使用验证mAP50:95最高轮选best；相同分数保留较早轮。每个epoch写CSV、独立epoch_NN.pth和val指标；last/best为便利别名，按训练设计更新，完整epoch快照保留。

## 权重和评估协议

权重来源：https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth

服务器初查缓存缺失，用户授权后已下载官方权重，77,844,807字节，完整SHA256为`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`。训练前验证缓存SHA256前缀907ea3f9，缺失立即停止。任何联网下载须单独确认，脚本没有下载选项，也不允许随机初始化替代正式训练。

用户已授权服务器安装matplotlib，实测版本3.11.2，本次loss_curve.png已由服务器生成并取回。代码仍保留缺少matplotlib时保存CSV并打印CURVE DEFERRED的分支，预测图使用Pillow。若未来需要本地补画，在新的空输出目录执行以下参考命令（本次不需要重画已存在的曲线）：

```powershell
& D:\python\.venv\Scripts\python.exe -B .\week08\day03\plot_training.py --csv .\week08\day03\outputs\ir_baseline_run01\training_log.csv --output-dir .\week08\day03\outputs\curve_review
```

此时才从真实CSV生成loss_curve.png，不安装依赖、不伪造曲线。

评估使用纯PyTorch实现：单person类、每图最多100预测、模型score下限0.001，IoU=0.50:0.05:0.95，每张图按分数贪心一对一匹配、全数据集排序、101点插值AP。P/R采用固定score>=0.5、IoU>=0.5。无GT图上的预测算FP。

所有VOC person框都作为普通GT；没有COCO crowd/ignore、面积分层和maxDets多档汇总。报告中明确标记为自定义单类协议，不能声称是官方COCO API完整输出。无需安装torchmetrics/pycocotools。本地和服务器均已有matplotlib/Pillow。

## 一条完整流程命令（服务器授权后执行）

```cmd
set PYTHONIOENCODING=utf-8
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\train_ir.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01 --device cuda:0
```

正式训练前自动用真实正样本与空框样本做两次冒烟更新，检查四项Loss、梯度、参数更新、resize及峰值显存，随后恢复模型初值并重置种子，开始10轮训练。不需要手动从几个batch切换到正式训练。

十轮结束后加载best，才首次读取test；test_started标记先保存，避免中断后自动重复测试。缓存test预测后，从缓存选成功/漏检/误检各最多3张绘图，不再推理测试集。某类案例没有发生时记录空列表。

## 断点恢复

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\train_ir.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01 --device cuda:0 --resume D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\last.pth
```

保存模型/优化器、可训练参数名、历史及Python/Torch/CUDA/DataLoader RNG。按最后完整epoch恢复，本轮未完成部分重跑；保留原CSV备份和独立控制台日志。配置或manifest改变会停止。OOM/非有限Loss、梯度或参数立即抛错，不降尺寸、不缩batch、不无限重试。

10轮已完成但测试或出图中断时不自动重复test；先人工检查test_started和缓存，制定单独恢复方案。因进程中断发生在epoch快照与last提交之间等极小窗口时，同名epoch快照会阻止自动覆盖，也需人工核对。

## 独立验证和缓存出图

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\evaluate_ir.py --checkpoint D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\val_review.json --device cuda:0
```

evaluate_ir只提供val评估入口；正式test由完整训练流程执行一次。

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day03\predict_ir.py --data-root D:\vision-learning-server\data\LLVIP_1000 --predictions D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\test_predictions.pth --output D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\prediction_examples_review
```

主要输出：config.json、smoke_test.json、逐轮epoch_NN.pth、last.pth、best.pth、training_log.csv、loss_curve.png、val_epoch_NN.json、test_started.json、test_metrics.json、test_predictions.pth、prediction_examples/、completed.json、真实结果交接Markdown。仅从本项目可信来源加载checkpoint。

## 本地已验证与待验证

本地 `check_day03.py` 退出码0，检查了语法、合成AP边界情况、真实IR样本模型transform尺寸和坐标、合法空框；结果见checks/local_checks.json。这没有跑模型前向/反向，也没有训练。结构检查中的未加载权重模型只用于检查，不是正式训练的权重退化。

CUDA冒烟前向/反向和参数更新、正式COCO权重加载、10轮训练、val选模及一次test均已实际通过。best为第9轮、last为第10轮；取回后本地严格加载两文件、核对历史和测试缓存指标通过。断点恢复机制已实现并保存状态，但本次无中断，未实测中断后续训。run01已完成，禁止向其中重新训练；以后重新训练须改为新的空输出目录。
