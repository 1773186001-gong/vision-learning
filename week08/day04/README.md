# Day04：与IR公平对照的RGB单模态Faster R-CNN

当前为**10轮RGB训练、一次最终测试、结果取回及本地核验完成**。最佳第4轮，test AP50=0.800126、mAP50:95=0.368304，退出码0。实际本地结果目录为`outputs/rgb_baseline_retrieved_20261030_213029_fa237fce`，正式交接见[DAY04_RGB_TRAINING_GPT_HANDOFF.md](DAY04_RGB_TRAINING_GPT_HANDOFF.md)。旧准备版已归档；下面命令供将来明确授权后复现，不要再次启动已完成任务。服务器重启后关机仍需用户完成。

## 配置和复用关系

模型、两类预测头、权重缓存检查、FrozenBN重建、checkpoint严格加载、设备传输、推理、AP/P/R和曲线直接调用Day03 `ir_common.py`；四项Loss、反向及参数更新、真实正/空样本冒烟、原子checkpoint保存直接调用Day03 `train_ir.py`，没有复制另一套AP或Loss实现。

Day04只增加RGB loader和独立流程编排，不修改Day01–Day03。`LLVIPDataset(modality='visible')`仍执行原有配对检查和同步预处理，但返回给检测器的只有可见光Tensor，没有IR输入或特征。RGB从官方COCO检测器重新初始化，然后替换两类头；不使用IR微调权重。

固定设置与IR一致：700/150/150场景隔离子集，640×640 Letterbox，内容640×512、上下各64；模型内部min/max也为640；train翻转p=0.5、val/test=0。10轮、batch1、workers0、seed42、SGD lr=.001、momentum=.9、wd=.0005；无AMP及scheduler，CPU线程2。val mAP50:95选best，test仅在选模结束后一次正式推理，之后只读缓存出图。

完整预训练SHA256必须为`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`；清单必须为`8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb`。缓存缺失、依赖缺失、环境变化或数据不符均停止，不自动下载或安装。

评估直接复用IR的单person类101点AP；IoU=.50:.05:.95，每图最多100预测、模型score下限.001；P/R与案例固定score≥.5、IoU≥.5；无COCO crowd/ignore、面积分组及多档maxDets。不能作为官方COCO API完整指标。Day04记录Python/CUDA/TF32等额外环境信息；Day03未记录的选项无法事后证明逐位相同，不声称跨进程完全确定性。

## 新增文件

| 文件 | 用途 |
| --- | --- |
| rgb_common.py | RGB数据入口、固定对照参数、保护哈希、IR参考读取、对照汇总 |
| train_rgb.py | 自动检查/冒烟/10轮/选模/一次test/缓存出图/异常日志及续训 |
| evaluate_rgb.py | 独立val评估；正式test函数仅由完整流程调用 |
| predict_rgb.py | 从RGB测试缓存出图，优先IR案例编号，按真实匹配分类 |
| preflight_rgb.py | 检查环境、全部图像/XML/配对、历史保护、缓存及电源信息 |
| report_rgb.py | 成功后根据真实历史和缓存指标生成交接及RGB–IR对照 |
| check_day04.py | 本地真实数据/结构及防覆盖检查，不做模型前向或训练 |
| protected_inputs.json | 40个历史代码/IR原始产物的保护SHA256，作为只读对照 |
| REMOTE_READONLY_CHECK.ps1 | 初次服务器只读检查；必须先获远程授权 |
| start_rgb_background.ps1 | 新建独立Day04任务，WMI创建隐藏监控进程并验证父进程 |
| monitor_rgb.ps1 | 持久监控Python、stdout/stderr、PID、结束时间和退出码 |
| REMOTE_PLAN.md | 一次性授权范围、命令、上传/取回和停止条件 |

本地执行（不要求CUDA、不下载、不训练）：

```powershell
Set-Location D:\python\test\vision-learning
& D:\python\.venv\Scripts\python.exe -B .\week08\day04\check_day04.py
```

真实检查结果保存在`checks/local_checks_1793334271617412000.json`，退出码0；补充CLI防写IR路径、拒绝CPU正式训练及11项运行包哈希检查见`checks/safety_cli_1793334685398549900.json`，检查程序退出码0。负例中的1/2是预期拒绝码，不是GPU训练失败。三个PowerShell脚本均通过本地Parser语法检查。没有将合成案例分类断言当作RGB实验成绩。

## 服务器完整流程（授权后）

前台复现命令（**服务器CMD**，用于将来复现；已有run必须换新的空目录）：

```cmd
D:\cv-server-env\Scripts\python.exe -u -B D:\vision-learning-server\week08\day04\train_rgb.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01 --device cuda:0 --keep-awake
```

`--keep-awake`只在用户批准后使用，调用Windows线程级保持唤醒请求，退出时释放，不改系统电源设置；无法阻止合盖操作、强制睡眠、休眠、关机或断电。

无人值守优先使用新的独立后台启动器，不改已有train-manager/train-monitor，也不写它们的current-job.json：

```cmd
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File D:\vision-learning-server\week08\day04\start_rgb_session.ps1 -Action start -KeepAwake
```

启动器寻找首个不存在的`rgb_baseline_runNN`，所以旧目录即使为空也会保留。它建立`week08/day04/jobs/<唯一任务名>/`，通过WMI服务创建隐藏PowerShell监控进程；核实其父进程为`WmiPrvSE.exe`后才发出启动确认标记。监控器等待此标记，缺失时自行失败退出，不启动训练。没有创建计划任务、安装服务或存储凭据。

后台机制已在服务器实际完成任务：用户退出启动SSH后训练继续推进，最终state=SUCCEEDED、exit_code=0。默认执行策略问题通过新增start_rgb_session.ps1和进程级RemoteSigned处理，未永久修改策略。新任务仍应检查启动回执和日志进展，不能仅凭进程创建就宣称成功。

启动会打印准确`JobDir`；状态查询例子（替换为真实目录）：

```cmd
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File D:\vision-learning-server\week08\day04\start_rgb_session.ps1 -Action status -JobDir D:\vision-learning-server\week08\day04\jobs\实际任务目录
```

`state.json`记录monitor/Python PID、开始/结束时间、退出码、SUCCEEDED/FAILED；stdout.log和stderr.log持续保存。训练目录另有完整console日志、每100batch进度及每轮CSV/JSON。后台任务不等于服务器不会睡眠；休息前仍应插电并保持盖子打开，读取并报告电源状态，不擅自修改系统配置。也不能保证Codex本身关闭后还能持续代为监看/下载；服务器任务若实测持久运行可自行训练、评估、保存，取回需本会话继续可用或之后恢复。

## 输出和失败处理

输出严格限制在`week08/day04/outputs/<run>/`，不能指向IR结果或原数据。新run不接受非空旧目录；best/last及当次CSV/曲线在该run内部按训练设计更新，逐轮快照和日志保留。OOM、NaN/Inf或异常不重试、不改参数、不清理旧文件。

预期完整产物：`config.json`、`preflight.json`、`initialization.json`、`smoke_test.json`、`console_*.txt`、`progress.json`、`training_log.csv`、`val_epoch_01..10.json`、`epoch_01..10.pth`、`best.pth`、`last.pth`、`loss_curve.png`、`test_started.json`、`test_metrics.json`、`test_predictions.pth`、`prediction_examples/`、`result_checks.json`、`protected_after.json`、`rgb_vs_ir.json`、`completed.json`、`exit_status.json`、`DAY04_RGB_TRAINING_GPT_HANDOFF.md`。

预测例子最多成功/漏检/误检各3张，优先IR既有编号，但不改变匹配结果来凑案例；类别可以重叠。不足时在case_details记录缺额。

仅对尚未完成10轮且test从未启动的同一run，可人工核对后显式续训：

```cmd
D:\cv-server-env\Scripts\python.exe -u -B D:\vision-learning-server\week08\day04\train_rgb.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01 --device cuda:0 --keep-awake --resume D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01\last.pth
```

保存模型、优化器、可训练参数、历史和Python/Torch/CUDA/DataLoader RNG；原CSV及退出记录保留，按最后完整epoch恢复。遇到同轮快照/未提交.tmp、环境或配置变化时停止，先人工核对。没有在本地进行中断续训训练测试。

独立val评估（输出须不存在）：

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day04\evaluate_rgb.py --checkpoint D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01\best.pth --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01\val_review.json --device cuda:0
```

本地根据取回的RGB缓存重画案例（没有模型推理；输出目录必须不存在）：

```powershell
& D:\python\.venv\Scripts\python.exe -B .\week08\day04\predict_rgb.py --data-root .\data\LLVIP_1000 --predictions .\week08\day04\outputs\rgb_baseline_run01\test_predictions.pth --output .\week08\day04\outputs\rgb_baseline_run01\prediction_examples_review
```

当前学习交接：[DAY04_RGB_TRAINING_GPT_HANDOFF.md](DAY04_RGB_TRAINING_GPT_HANDOFF.md)。成功后应以实际取回的运行报告、完整日志、CSV/JSON和图片为依据完成最终交接，不能把本README中的计划写成已运行结果。


## 2026-10-30 结果取回与核验

实际下载目录：`outputs/rgb_baseline_retrieved_20261030_213029_fa237fce`。使用`download_rgb_results.ps1`由用户在本地CMD输入SSH密码取回，后运行`review_rgb_results.py`通过。检查记录见该目录`local_result_review.json`，原始日志及报告保留。原关机守候器已随服务器关机结束，历史waiting状态不代表当前存活；尚未确认此次重启后的关机。
