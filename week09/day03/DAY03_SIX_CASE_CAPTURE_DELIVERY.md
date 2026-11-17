# Day03 固定六例RPN筛选追踪交付

本轮已完成授权的五张完整validation图片捕获，IR/Fusion各一次，共10次新增原eval前向。030269/G1只复用原始19项单图产物，未再次前向。服务器进程退出码0；下载校验及本地离线复算退出码均为0。到此停止，未调参、训练、运行test或启动新实验。

## 1. 文件与范围

实际核对项目AGENTS.md、Day03 README/方案/核心模块、capture_single_val.py和verify_single_capture.py，原Day01/Day02捕获逻辑、ir_common.py、fusion_model.py、fusion_common.py及selected_cases.csv。研究记忆通过shell检索，仅供定位，不作为本次指标来源。

新增：

- capture_remaining_val.py：固定五图入口；拒绝创建030269的实时适配器。以独立模块命名空间复用已验收单图函数，原单图代码及数值流程不变。
- verify_merged_capture.py：从保存的原始张量、实际Top-N/NMS索引离线重建阶段，生成六例报告；不加载checkpoint、不执行模型。
- test_remaining_capture.py：6项人工测试，含固定列表/顺序、拒绝复用例前向、命名空间隔离、裁剪后才出现达标框、阈值精度、动态导出范围。
- 本交付文档；checks下新的人工核验、源包、复用哈希清单、传输和独立核验记录。
- outputs/remaining_five_merged_run01及下载压缩包：所有真实捕获、表格、轨迹、环境和日志。

修改：仅Day03 README，追加实际结果。核心trace_rpn_stages.py、原单图入口、Week08/Day01/Day02以及全部旧日志均未修改。

实际新增顺序以selected_cases.csv为准：

|顺序|案例|image_index|完整图片新增前向|
|---|---|---:|---:|
|1|030308/G1|24|IR 1，Fusion 1|
|2|030205/G0|16|IR 1，Fusion 1|
|3|170100/G1|120|IR 1，Fusion 1|
|4|130126/G3|85|IR 1，Fusion 1|
|5|170455/G1|145|IR 1，Fusion 1|

GT索引从0开始；分析指定GT，但正常输入整张配对图像，未裁GT或改变候选生成。原恢复入口中Fusion构造会执行既有的CPU零张量backbone结构探测；这不是额外真实图片前向，未改动该原始恢复行为。

## 2. 实际环境与原权重

重新交互登录后核对主机DESKTOP-BFIO0RA；SSH无密码批处理探测失败，随后既有交互认证成功。没有保存凭据，完成传输后已退出SSH。项目没有Git元数据，使用文件SHA检查保留状态。

服务器Windows10，Python3.14.6，Torch2.14.1+cu126，TorchVision0.29.1+cu126，NVIDIA GeForce RTX3060 Laptop GPU，CUDA可用，cuDNN91002。seed=42、threads=2、cudnn_deterministic=True、benchmark=False。原数据/预处理源码和五图RGB/IR/XML哈希在前向前与Day02匹配。

|模型|服务器best路径|SHA256|
|---|---|---|
|IR|D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth|9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2|
|Fusion|D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth|9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a|

实际特征顺序为0、1、pool；每层pre-NMS=150，三个层合计Top-N后450；每张图post-NMS上限150；score阈值0.05、NMS阈值0.7保持原值。含Letterbox padding的整张640×640是模型坐标系。

## 3. 主要真实结果

下表均为与Day02一致的CPU float32 torchvision.box_iou，非float64诊断口径。12项与Day02的实际差值全部为0，起始atol=1e-6、rtol=0未放宽。覆盖阈值始终严格按IoU≥0.75判断。

|案例|IR解码最大IoU|Fusion解码最大IoU|IR最终最大IoU|Fusion最终最大IoU|主要观测|
|---|---:|---:|---:|---:|---|
|030269/G1（复用）|0.793667018|0.750537336|0.793667018|0.605291903|Fusion唯一达标框在NMS丢失|
|030308/G1|0.878793776|0.712219298|0.878793776|0.712219298|Fusion七阶段均未达标|
|030205/G0|0.851542413|0.729380310|0.851542413|0.696796298|Fusion七阶段未达标，NMS进一步降低最佳IoU|
|170100/G1|0.769244015|0.845996797|0.578785658|0.845996797|IR唯一达标框在score过滤丢失|
|130126/G3|0.875696659|0.846598208|0.664702535|0.846598208|IR解码最佳更高，但5个达标框全部在NMS丢失|
|170455/G1|0.750012934|0.674085140|0.750012934|0.674085140|IR临界覆盖保持；Fusion七阶段均未达标|

七阶段依次为：解码 → 逐层Top-N → 裁剪 → 小框过滤 → score过滤 → 按层NMS → 最终预算截断。完整84行最大IoU、候选数量、达标数量和最佳ID见merged_stage_summary.csv及six_case_report.md。所有阶段都检查了，未因decoded不达标而跳过裁剪。

达标候选数量按上述七阶段顺序：

|案例|IR|Fusion|
|---|---|---|
|030269/G1|2→2→2→2→2→2→2|1→1→1→1→1→0→0|
|030308/G1|5→5→5→5→4→2→2|0→0→0→0→0→0→0|
|030205/G0|8→8→8→8→8→2→2|0→0→0→0→0→0→0|
|170100/G1|1→1→1→1→0→0→0|3→3→3→3→3→3→3|
|130126/G3|5→5→5→5→5→0→0|2→2→2→2→2→1→1|
|170455/G1|1→1→1→1→1→1→1|0→0→0→0→0→0→0|

这12项的最大IoU和达标数量在Top-N、裁剪、小框过滤处均未变化。这不表示所有候选坐标都不变，真实同ID裁剪差异另存clip_changes.csv。GT完整处于裁剪范围内，同ID裁剪IoU不降的float64核验通过；没有对未通过Top-N的候选作反事实裁剪。

130126的IR/Fusion在NMS后分别有171/201个候选，最终确实截到150；指定GT的最佳IoU和达标数量在该截断处未下降。因此本次没有证据将这些GT的最终失败归于post-NMS预算。其余图最终框数均小于150。

## 4. 同一候选的证据

flat索引只在对应图片和模型内标识身份，完整ID带模型/案例前缀。跨模型同flat索引不表示预测坐标相同。阶段最佳框每阶段重新选择，固定候选则从同一原索引持续追踪。

- 030269/Fusion：ID2201、层0，IoU=0.7505373358726501，objectness=0.9004508256912231；进入NMS前分数排名27、层内11，实际NMS保留索引中无此ID。证明删除阶段，未证明实际抑制者。
- 170100/IR：ID13256、pool层，IoU=0.7692440152168274；objectness=0.0320378914475441，低于原0.05阈值。小框过滤后仍在，score过滤后不在。此前在450个候选中的分数排名165、层内21，因此可以通过逐层Top-N，随后仍被score阈值删除。这直观说明“每层150”不是“全图分数前150”。
- 130126/IR：5个达标ID为3216、3516、3521、9216、9516，层0或1。它们均进入NMS，实际返回索引全部不含这些ID。最佳ID3516的IoU=0.8756966590881348、分数=0.9649204015731812；分数最高的达标ID3521分数=0.9991126656532288，也被NMS删除。这里不能用“分数过低”笼统解释，更不能凭相邻框猜测抑制者。
- 030205/Fusion：阶段最佳ID9506的IoU=0.7293803095817566、分数=0.9613238573074341，NMS前分数排名4、层内3；NMS删除后最佳切换为9492，最终IoU=0.6967962980270386。该例虽始终没达到0.75，仍有实际筛选损失。

其余阶段最佳框及所有曾达标候选的完整去留、分数、排名和坐标见merged_candidate_trajectory.csv及各模型candidate_trajectory.csv。未实现NMS抑制者配对。

## 5. 验收与真实日志

服务器真实命令：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day03\capture_remaining_val.py --device cuda:0 --reuse D:\vision-learning-server\week09\day03\outputs\single_030269_run01 --output D:\vision-learning-server\week09\day03\outputs\remaining_five_merged_run01
~~~

真实终端摘录（完整内容保存在run.log）：

~~~text
PREFLIGHT_PASSED ... protected= 493
CASE_VERIFIED 030308/G1 new_forwards=2
CASE_VERIFIED 030205/G0 new_forwards=2
CASE_VERIFIED 170100/G1 new_forwards=2
CASE_VERIFIED 130126/G3 new_forwards=2
MODEL_VERIFIED IR forward_calls=1 f32_cpu=0.7500129342079163 day02_delta=0 f64=0.7500129389053078 restored=True
CASE_VERIFIED 170455/G1 new_forwards=2
COMPLETED new_original_eval_forwards=10 reused_original_eval_forwards=0 verified_gt_model_items=12
PROCESS_EXIT_CODE=0
~~~

原方法正常调用并原样返回；原调用的Top-N/NMS索引传播ID。核心在同次原设备上精确核对重放NMS输入、最终框/顺序/分数；RPN后置hook精确核对实际RPN输出。模型state、checkpoint、配置及493项服务器受保护文件前后未变。正常路径所有包装/hooks恢复；异常恢复由本次完整复跑的人工测试覆盖，本次真实前向未人为注入异常。

本地37项人工测试完整复跑通过，退出码0，记录在checks/remaining_adapter_unit_run02；run01也保留。它们是人工单元核验，不是LLVIP实验结果。

本地独立离线核验命令：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B week09/day03/verify_merged_capture.py --capture-dir week09/day03/outputs/remaining_five_merged_run01 --output week09/day03/checks/independent_six_case_run01
~~~

输出12项OFFLINE_VERIFIED，全部day02_delta=0.0；末行为OFFLINE_ALL_12_PASSED no_model_forward no_checkpoint_load，退出码0。还核对19项复用文件与本地原单图逐项SHA一致、429项本地历史文件未变。独立核验应用捕获的真实NMS索引，不另外运行CPU NMS来替代原GPU同分决策。

下载压缩包10313484字节，服务器与本地SHA256均为：

`7b8728e791e0a46b085c6fad75f517556a2e3ed7ec2918e8f68b8ec79788a220`

104项产物SHA逐项通过；清单自身由压缩包整体SHA保护。传输记录见checks/remaining_transfer_run01，独立核验见checks/independent_six_case_run01。源包run01在补入服务器缺少的离线辅助模块和强制复用清单前已打包但未解包/运行；正式使用run02，两个包均保留。没有失败的真实图片前向或自动重跑。

## 6. 观察、已证实事实与仍待验证

已证实的操作事实：这些固定GT有上表所示的解码质量和筛选轨迹；NMS导致全部达标候选消失这一模式同时出现在一个Fusion例和一个IR例。score过滤造成全部消失只见于170100/IR；Fusion七阶段无达标框的情况在三个选定案例重复出现。130126/IR解码最佳比Fusion还高，最终却更低，说明仅看最终框无法判断解码阶段谁更好。

合理假设：候选分数与定位质量不完全一致、某些上游候选生成差异，可能值得进一步只读检查。当前证据不能把这些机制写成原因，不能把NMS现象全部归咎于融合模型。

仍未证实：融合特征或回归分支的根因、实际NMS抑制者、全验证集发生频率，以及最终RoI/AP75变化由哪一机制主导。这六例按结果规则选择，不能据此估计总体比例。RPN Proposal IoU不是最终检测AP75。

如后续获准，优先利用现有原始捕获离线检查丢失候选的分数/排名与保留框关系，明确是否需要另行记录NMS抑制证据；不直接修改融合模块。本轮没有实施该后续步骤。
