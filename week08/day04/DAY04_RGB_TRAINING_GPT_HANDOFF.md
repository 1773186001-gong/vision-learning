# Week08 Day04：RGB 可见光行人检测基线正式交接

更新日期：2026-10-30。真实 RTX 3060 训练、10轮验证选模、一次最终测试、结果取回和本地核验均已完成。最佳第4轮，进程退出码0，后台监控状态 SUCCEEDED。此次整理没有重新训练或执行测试模型推理。服务器重启后的关机尚未执行，不把原守候器的历史状态当作当前进程状态。

## 1. 模型、数据和环境

模型为 TorchVision `fasterrcnn_mobilenet_v3_large_320_fpn`，使用官方 COCO 预训练检测器，替换 FastRCNNPredictor 为背景/person 两类。RGB 从官方预训练重新初始化，不使用 Day03 IR 微调 checkpoint。Dataset 仍为既有配对预处理读取两模态，但只有 visible RGB Tensor 输入检测器，IR 不参与模型计算。

权重来源：https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth。训练时缓存已存在，SHA256：`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`。未退化为随机初始化，Day04 未安装或升级依赖。

服务器 Windows 10，Python 3.14.6；PyTorch 2.14.1+cu126；torchvision 0.29.1+cu126；CUDA 12.6；GPU NVIDIA GeForce RTX 3060 Laptop GPU，设备 cuda:0。解释器：`D:\cv-server-env\Scripts\python.exe`。

LLVIP_1000：train/val/test=700/150/150对，person框=1991/416/348。原图1280×1024等比例缩到640×512，上下各填64像素，最终640×640；检测器内部 min_size=max_size=640，避免再次改变尺寸。train 同步水平翻转p=0.5，val/test p=0；合法空标注 train/100030 正常参与训练。

场景前缀严格隔离：train=01,02,04,05,06,07,09,10,11,12,14,15,16,18,25；val=03,08,13,17；test=19,20,21,22,23,24,26。预检查完整解码2000幅图像、解析1000份XML。清单SHA256：`8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb`。

## 2. 实际训练配置和冒烟测试

10 Epoch；batch_size=1；num_workers=0；seed=42；CPU线程=2；SGD lr=0.001、momentum=0.9、weight_decay=0.0005；无AMP、无学习率调度。学习率沿用已完成的IR基线，以保持对照条件一致，并非宣称最优。验证集 mAP50:95 选 best，last 保存第10轮及优化器/RNG状态。测试集不参与调参。

实际冒烟包括正样本010001与空样本100030，完成四项Loss、反向传播和更新；总Loss分别为1.3820651769638062和0.9247241020202637，空样本两项框回归Loss均为0。resize_boxes_unchanged=true，冒烟后模型与两类头精确恢复、seed重置为42，然后进入正式训练。冒烟峰值allocated=318372352 bytes，reserved=413138944 bytes。

## 3. 十轮真实训练和验证记录

下表保留6位小数；完整精度以 [training_log.csv](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/training_log.csv) 和逐轮JSON为准。每轮lr均为0.001，显存单位为bytes。

|Epoch|total|classifier|box_reg|objectness|rpn_box|val P|val R|val AP50|val mAP50:95|秒|peak allocated|peak reserved|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|0.506035|0.182500|0.292804|0.022808|0.007924|0.715942|0.593750|0.642439|0.244894|128.296562|395375104|484442112|
|2|0.561106|0.195942|0.344784|0.013746|0.006635|0.696237|0.622596|0.637320|0.238748|128.024973|394217984|486539264|
|3|0.609644|0.203031|0.388923|0.011541|0.006149|0.779762|0.629808|0.665828|0.246970|129.746651|392636928|486539264|
|4|0.607672|0.199229|0.392790|0.009913|0.005740|0.677184|0.670673|0.676349|0.264635|132.265144|394009088|486539264|
|5|0.634413|0.206509|0.412777|0.009502|0.005624|0.713137|0.639423|0.670006|0.263865|134.888796|393999872|486539264|
|6|0.629659|0.198611|0.417487|0.008340|0.005221|0.759104|0.651442|0.661033|0.251245|136.449270|381565952|486539264|
|7|0.620926|0.197094|0.410201|0.008464|0.005165|0.777448|0.629808|0.666329|0.264234|136.906467|394217984|486539264|
|8|0.602126|0.190048|0.399625|0.007469|0.004984|0.756233|0.656250|0.670617|0.260349|137.308055|379776000|486539264|
|9|0.623210|0.187932|0.423306|0.007139|0.004833|0.661058|0.661058|0.643049|0.242347|138.724401|384073728|486539264|
|10|0.619015|0.190770|0.416792|0.006698|0.004755|0.770349|0.637019|0.651752|0.255090|139.098995|381775872|486539264|

最佳第4轮：val P=0.6771844660、R=0.6706730769、AP50=0.6763494645、mAP50:95=0.2646349136。不是最后一轮，也不是训练Loss最低的一轮。

![训练曲线](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/loss_curve.png)

## 4. 最终测试与 IR 对照

共同评估协议：单person类，101点插值AP，IoU=.50:.05:.95，每图最多100预测，模型score下限=.001；P/R和图片案例使用score≥.5、IoU≥.5。未实现COCO crowd/ignore及面积分层，不能称为官方COCO API完整评估。两模态使用同一个评估函数。

|测试指标|IR：最佳第9轮|RGB：最佳第4轮|RGB减IR|
|---|---:|---:|---:|
|precision|0.850416|0.491776|-0.358639|
|recall|0.882184|0.859195|-0.022989|
|AP50|0.898907|0.800126|-0.098781|
|mAP50_95|0.471626|0.368304|-0.103322|
|TP|307.000000|299.000000|-8.000000|
|FP|54.000000|309.000000|+255.000000|
|FN|41.000000|49.000000|+8.000000|

真实观察：RGB测试误检309个，IR误检54个；RGB AP50低9.878个百分点，mAP50:95低10.332个百分点。RGB的Recall仍为0.8592，说明主要差异包含大量额外误检，不能只用召回率判断检测质量。单次、单seed、固定小子集结论不能推广成所有RGB场景都弱于IR，也不能推断融合结果。

曲线中 total_loss 从第1轮0.5060上升到后期约0.60–0.63，主要包含ROI框回归Loss上升；RPN两项Loss则下降。不能把这条曲线描述为持续下降或完全收敛。检测器训练时提议框和采样目标会变化，Loss不是固定样本的同一误差测量；这只是解释方向，本次没有另做诊断来证明具体原因。验证mAP第4轮最高，此后波动，本次未据测试集改参或重训。

## 5. 显存与时间

逐轮训练+验证累计 1341.709314 秒（约22分21.7秒）；脚本实际运行 1403.201208 秒（约23分23.2秒）。后台开始12:57:26.936，结束13:20:53.333（2026-10-30，中国标准时间）。逐轮计时不含随后checkpoint保存、绘图和最终测试；脚本总计时包括这些流程，外部启动器耗时另算。

逐轮最大 peak_allocated=395375104 bytes（377.059 MiB），peak_reserved=486539264 bytes（464.000 MiB）。这是PyTorch分配器统计，不是整卡总占用；不能据此保证随意增大batch不会OOM。没有观察到OOM、NaN/Inf、训练中断或自动改参。stderr.log为0字节，成功退出。耗时是单次运行，不是稳定性能基准。

## 6. 预测图及目视检查

保存成功/漏检/误检各3张，共9张；绿色为GT，红色为预测框，标注person与置信分数。案例允许重叠，分类来自固定阈值匹配而非人工制造。已目视检查曲线和以下3张代表图，其余PNG通过完整性验证。靠近右边界的文本可能被画布截断，应结合case_details.json读取数量。

- [success_200183.png](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/prediction_examples/success_200183.png)：TP=1、FP=0、FN=0，预测与行人框基本重合。
- [false_positive_190001.png](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/prediction_examples/false_positive_190001.png)：TP=2、FP=5、FN=0；背景暗区和车辆附近出现额外预测。同编号IR图属于成功案例，可对照。
- [miss_190040.png](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/prediction_examples/miss_190040.png)：TP=6、FP=1、FN=1，局部拥挤/边缘区域需结合GT查看。该描述不是漏检原因的因果验证。

完整案例计数：[case_details.json](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/prediction_examples/case_details.json)。测试缓存共150张、348个GT框。此次只复算缓存指标，没有第二次执行测试模型推理。

## 7. 结果取回与核验

服务器原目录：`D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01`。

实际本地目录：[rgb_baseline_retrieved_20261030_213029_fa237fce](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce)。注意本地使用独立下载目录，不能误写成已存在本地rgb_baseline_run01。逐轮epoch_01至10权重保留在服务器，本地取回best/last和全部必要小文件及测试缓存。

用户在本地CMD执行download_rgb_results.ps1，通过OpenSSH输入密码完成取回；脚本不存密码，不改远程文件。未继续操作PyCharm。关机后重开服务器只用于取回，未重启训练。

本地核验命令（已实际执行，退出码0；输出JSON已存在，脚本采用独占写入，不要向同目录重复运行）：

```powershell
& D:\python\.venv\Scripts\python.exe -B .\week08\day04\review_rgb_results.py --output .\week08\day04\outputs\rgb_baseline_retrieved_20261030_213029_fa237fce --job .\week08\day04\checks\retrieval_20261030_213029_fa237fce\job
```

实际通过：两checkpoint严格加载；best=4、last=10；CSV与checkpoint历史和全部验证JSON一致；150张缓存指标复算与test_metrics/completed一致；best SHA与test_started一致；40个历史保护文件与11个运行文件哈希一致；10张PNG（9案例+曲线）可解码；后台SUCCEEDED与进程退出码0一致。没有进行本地模型推理或训练。断点续训代码存在，但此次未实测中断后恢复。

核验记录：[local_result_review.json](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/local_result_review.json)；本地文件哈希清单：[artifact_sha256.json](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/artifact_sha256.json)。这些证明本地内容一致性；没有将全部文件与服务器重新逐文件计算SHA对比，不能夸大为全量远端哈希核验。

best.pth：151536180 bytes；SHA256 `d37d901d5459ffe1d70472b03a1637dc5a075f6a22c63ed249f96e5f6748df32`。

last.pth：151537332 bytes；SHA256 `bbc33ca8655c7b68b1b14502890d37d4355c328c76eb0449defd35b3df367eb4`。

完整训练终端日志：[console_1793336249285875400.txt](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/console_1793336249285875400.txt)；后台记录：[state.json](D:/python/test/vision-learning/week08/day04/checks/retrieval_20261030_213029_fa237fce/job/state.json)、[stdout.log](D:/python/test/vision-learning/week08/day04/checks/retrieval_20261030_213029_fa237fce/job/stdout.log)、[stderr.log](D:/python/test/vision-learning/week08/day04/checks/retrieval_20261030_213029_fa237fce/job/stderr.log)。服务器自动生成的原始报告原样保留：[DAY04_RGB_TRAINING_GPT_HANDOFF.md](D:/python/test/vision-learning/week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/DAY04_RGB_TRAINING_GPT_HANDOFF.md)。原准备版已归档为[DAY04_RGB_PREPARATION_ARCHIVE.md](D:/python/test/vision-learning/week08/day04/DAY04_RGB_PREPARATION_ARCHIVE.md)。

## 8. 实际问题与处理

1. 用户曾把cd、目录和命令连成一行，导致CMD路径错误；改为直接执行完整命令。
2. Windows默认执行策略阻止原启动脚本；新增start_rgb_session.ps1，以进程级RemoteSigned启动，未永久修改系统策略，旧脚本保留。
3. 用户退出启动SSH后，后续训练记录仍推进；最终监控状态和10轮完整产物确认后台任务完成。
4. 早期取回曾因工具额度与服务器关机中断。恢复后SFTP握手失败，用户确认服务器已关闭；后来重新开机，改用用户执行的命令行下载脚本成功取回。
5. 代理命令行SSH免密认证被拒绝，未索取密码或改动认证配置。
6. 原关机守候器状态文件仍为waiting，但服务器已关机重开，旧PID不能证明进程存活。未上传ready_to_shutdown标记，当前没有代为关机成功的证据。完成本地保存后可由用户在自己的已认证终端关机。

## 9. 代码文件与复现命令

训练核心：rgb_common.py（RGB loader/共享配置与对照）、train_rgb.py（全流程/续训）、evaluate_rgb.py（验证与一次test）、predict_rgb.py（缓存出图）、preflight_rgb.py（环境/数据/权重检查）、report_rgb.py（自动报告）。检查与运行支持：check_day04.py、protected_inputs.json、upload_manifest.json、monitor_rgb.ps1、start_rgb_background.ps1、start_rgb_session.ps1、REMOTE_READONLY_CHECK.ps1。收尾新增：review_rgb_results.py（本地核验）、download_rgb_results.ps1（命令行下载）、shutdown_after_rgb.py/arm_rgb_shutdown.ps1（原关机守候器，重启后不再运行）。README及REMOTE_PLAN保留操作说明。没有修改Day01–Day03代码、原始数据和IR结果。

实际Python训练命令（记录用途，本次不要重新运行）：

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day04\train_rgb.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01 --device cuda:0 --keep-awake
```

实际后台启动命令：

```cmd
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File D:\vision-learning-server\week08\day04\start_rgb_session.ps1 -Action start -KeepAwake
```

将来获准重新复现时，必须使用新的空输出目录，例如将上面--output改成`D:\vision-learning-server\week08\day04\outputs\rgb_baseline_reproduction_01`；其余参数保持相同。训练脚本自动冒烟、10轮、选模、一次test和出图。不要为了生成报告再跑test。

独立验证命令（本次未重复执行；输出必须不存在）：

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day04\evaluate_rgb.py --checkpoint D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01\best.pth --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01\val_review.json --device cuda:0
```

本地从缓存重画案例（本次未重复执行；不进行模型推理，输出必须不存在）：

```powershell
& D:\python\.venv\Scripts\python.exe -B .\week08\day04\predict_rgb.py --data-root .\data\LLVIP_1000 --predictions .\week08\day04\outputs\rgb_baseline_retrieved_20261030_213029_fa237fce\test_predictions.pth --output .\week08\day04\outputs\rgb_baseline_retrieved_20261030_213029_fa237fce\prediction_examples_review
```

## 10. 交给ChatGPT的学习任务

请依据本报告和真实图片，全程中文、每次一个明确步骤，不主动测验。先解释为什么RGB Precision明显下降但Recall仍较高，再解释AP与固定阈值P/R的区别、四项Loss和非单调曲线、val选best而不是选last、640外部Letterbox与内部resize、预训练重新初始化与IR微调初始化的区别、allocated/reserved计数、同编号两模态预测差异。区分已有证据、合理假设和仍需验证的原因。

当前进度：Day01数据、Day02同步预处理及GPU管线、Day03 IR基线、Day04 RGB基线及本地核验完成。下一步先理解和分析IR/RGB真实差异；未开展融合、多seed或额外调参实验，不自动进入Day05。
