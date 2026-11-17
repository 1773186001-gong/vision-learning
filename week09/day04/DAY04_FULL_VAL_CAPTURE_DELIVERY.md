# Week09 Day04：完整validation RPN逐阶段覆盖交付

整理日期：2026-11-08（Asia/Shanghai）。原条件GPU捕获、六图复用、合并统计、下载SHA核对与本地独立核验全部完成。checks/independent_full_val_run01/verification.json实际状态为passed、exit_code=0。本轮结束，不启动模型改进实验。

## 1. 实际执行边界

完整原validation，每模型150张完整图片、416个有效GT。六图12份Day03捕获通过兼容性核对后原样复用；剩余144图，每张IR/Fusion各一次，共288次新增原eval前向。030001是剩余清单首图，两个模型的验收计入288次，没有额外真实图片冒烟前向。没有重复六图前向。

不训练、不运行test、不调阈值／预算、不修改融合结构，也不扩展NMS抑制者配对。原best、数据预处理、transform和特征键0／1／pool保持不变，Fusion RoI沿用0／1；逐层pre150、逐图post150、score0.05、NMS0.7。使用model.eval()、torch.inference_mode()，统计来自RPN候选，不是最终RoI检测框。

服务器重新交互登录后核对为DESKTOP-BFIO0RA，Python3.14.6、Torch2.14.1+cu126、TorchVision0.29.1+cu126、RTX3060 Laptop GPU。固定seed=42、threads=2、cudnn deterministic=True、benchmark=False。没有保存凭据；下载开始后已退出原SSH会话，后续只在本地计算。原Fusion恢复入口的CPU零张量backbone结构探测沿用原构造，不属于额外真实图片前向。

470项输入／源码SHA、manifest、原best SHA、配置及Day01参照文件SHA经过服务器预检。1048项服务器受保护文件的前后哈希一致。完整原权重路径及SHA见原预检／run_metadata.json，原IR/Fusion best仍为：

- IR：`D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth`，SHA256 `9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2`。
- Fusion：`D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth`，SHA256 `9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a`。

## 2. 新增能力与文件

新增代码仅在week09/day04：

- agnostic_capture.py：与指定GT无关的候选捕获及重放。保存原解码候选、真实Top-N／NMS索引、分数、阶段候选、完整[G,4] GT及稳定ID。G=0直接保存[0,4]，不使用假GT。
- capture_ledger.py：每个模型＋stem独立账本，临时文件写入、flush/fsync后原子提交；已验收且签名／SHA一致时禁止重复前向。
- capture_full_val.py：服务器预检、六图复用、144图串行捕获、首图正式验收、逐项检查和分块进度。每10张为进度块，最小落盘／验收单位仍是模型＋图片。
- full_val_analysis.py：新／旧捕获的统一离线核对、全部GT统计、Day01逐项对照及完整中文报告。
- test_full_capture.py：人工捕获／恢复／账本测试；已有14项统计测试保持不变。
- verify_full_val.py：本地独立坐标／导出表／账本核验入口，复用已验收verify_pilot.py的独立面积公式及汇总检查，不修改原核验源码。

未修改Day03核心或历史代码、未覆盖六图试运行及任何旧运行。Day04 README仅补充全量入口；完整val产物使用新目录outputs/full_val_stage_run01。

人工测试在本地和服务器各通过26项，均无真实检测模型前向：多GT、空GT、原返回对象不变、正常／异常恢复、配置／引用读取失败后的锁释放、并发拒绝不释放他人锁、原子提交、SHA／签名不匹配拒绝、完整捕获优先离线恢复、未知项停止及分块跳过已验收项等。真实剩余清单没有空GT图片，未制造额外真实空GT前向；不能把人工空GT测试写成真实GPU空GT实验。

## 3. 覆盖差距在解码时已有多少

主口径为与Day01／Day02一致的CPU float32 torchvision.box_iou，达标定义严格为IoU≥0.75。float64作为独立诊断另列，不把容差加入覆盖判定。整体分母是各模型416个GT，不平均逐图比例。

|阶段|IR覆盖GT|Fusion覆盖GT|IR−Fusion净差|净差较前阶段变化|
|---|---:|---:|---:|---:|
|解码|327/416|321/416|6|不适用|
|逐层Top-N|326/416|319/416|7|+1|
|裁剪|328/416|321/416|7|0|
|小框过滤|328/416|321/416|7|0|
|score过滤|326/416|321/416|5|−2|
|NMS全部输出|302/416|287/416|15|+10|
|最终150预算|302/416|285/416|17|+2|

解码时IR已经比Fusion多覆盖6个GT；后续操作的净变化合计为+11，最终净差17。逐步算术关系为：

`17 = 6 + 1 + 0 + 0 − 2 + 10 + 2`。

这说明差距并非全部在解码时形成，也不能全部归给后续筛选。NMS阶段在覆盖GT净差的算术变化中最明显；这不是“NMS造成全部AP75差异”或“融合特征没有问题”的因果结论。

阶段净差变化定义为：

`Δ(IR覆盖−Fusion覆盖) = (Fusion覆盖丢失−IR覆盖丢失) + (IR新增覆盖−Fusion新增覆盖)`。

该等式只是计数分解，不能把某个数直接称为模块的因果贡献。

## 4. 哪些阶段改变了覆盖

|模型|阶段|覆盖全部丢失GT／前阶段已覆盖GT|新增覆盖GT|部分达标候选减少但仍覆盖GT|
|---|---|---:|---:|---:|
|IR|Top-N|1/327|0|6|
|Fusion|Top-N|2/321|0|5|
|IR|裁剪|0/326|2|0|
|Fusion|裁剪|0/319|2|0|
|IR|小框过滤|0/328|0|0|
|Fusion|小框过滤|0/321|0|0|
|IR|score过滤|2/328|0|16|
|Fusion|score过滤|0/321|0|10|
|IR|NMS|24/326|0|255|
|Fusion|NMS|34/321|0|241|
|IR|最终预算|0/302|0|1|
|Fusion|最终预算|2/287|0|0|

“部分好框减少但仍覆盖”与“该GT的达标框全部消失”不同。NMS虽然删除大量重复达标框，多数相关GT仍被至少一个框覆盖；不能把候选删除数直接当作GT覆盖丢失数。

### 裁剪后首次达标是真实发生的

- IR：030116/G2、030410/G0。
- Fusion：030116/G2、130195/G3。

这些GT的完整覆盖序列均为0011111（七阶段按表中顺序），是裁剪后第一次达到0.75，而非此前已覆盖后的恢复。本次没有记录到“覆盖丢失后再次恢复”的GT。GT完整位于640×640裁剪范围内，含Letterbox padding；有限有效框的同ID裁剪IoU在精确几何下不降，浮点核查容差不改变0.75阈值。

### 最终预算也存在全量覆盖丢失

Fusion的081281/G7与130500/G5为1111110：原NMS全部输出阶段仍覆盖，最终截到150后不再覆盖。这两个例子没有出现在原六图试运行中。原六图“预算未降低指定GT覆盖”的观察仍然有效，但不能推广为完整val没有预算损失。

IR在最终预算处没有GT覆盖丢失，但有1个GT的达标候选数量减少、覆盖保持。这也说明“没有覆盖损失”不等于该操作没有删除候选。

### 始终未覆盖与最终失败不同

七阶段始终未覆盖：IR 87/416、Fusion 93/416；最终未覆盖还包括曾有达标候选但后来丢失的GT。完整名单、首次达标、首次全部消失和完整覆盖序列见gt_coverage_sequences.csv与never_covered_gt.csv。

## 5. 场景如何组成总体净差

|场景|GT数／模型|解码IR/Fusion覆盖|解码净差|最终IR/Fusion覆盖|最终净差|解码后净差变化|
|---|---:|---|---:|---|---:|---:|
|03|93|78／72|+6|78／63|+15|+9|
|08|121|99／101|−2|90／90|0|+2|
|13|131|98／98|0|91／92|−1|−1|
|17|71|52／50|+2|43／40|+3|+1|
|整体|416|327／321|+6|302／285|+17|+11|

最终净差按场景相加：`17 = 15 + 0 − 1 + 3`。场景03是总体IR优势的主要计数来源；这只是描述性组成，不是场景或融合模块的因果贡献。场景分母不同，不能直接平均各场景比例。

场景03的净差从6变15：裁剪的净差变化+1、NMS为+8。场景08从−2变0：Top-N与最终预算各+1。场景13从0变−1：裁剪−1、NMS−1、最终预算+1。场景17从2变3：score过滤−2、NMS+3。各场景变化相加还原全局各阶段变化，不能忽视相互抵消。

## 6. 配对结果与验收口径

最终416个GT配对：两模型均覆盖255，仅IR覆盖47，仅Fusion覆盖30，两者均未覆盖84，与Day01已有最终配对结果相符。

不仅核对302／285总数，还逐GT核对模型＋stem＋0起始gt_index、GT坐标、场景、实际候选数、最大IoU与0.75覆盖判定。832项最终IoU与Day01 max_iou_top300的实际差值全部为0，起始atol=1e-6、rtol=0未放宽。原post预算150，所以Day01 Top300包含全部实际最终候选。

原方法正常调用并原样返回；新捕获在同次原设备上精确重放NMS输入、最终坐标／顺序／分数，并与实际RPN输出核对一致。每项模型持久state前后相等且等于初始加载state，配置不变，临时包装／hooks恢复；权重及受保护文件哈希在运行前后核对。没有把不同设备或版本必然逐位一致当成前提。

## 7. 账本、未知状态与本次真实运行状态

本次只有attempt_001，288个新项均有started.json、raw_capture.pth、captured.json和verified.json，12个复用项保留原始字节及来源SHA。没有失败项、未知项或残留.tmp文件，没有发生真实重跑或resume。分块恢复策略的异常路径由人工测试覆盖，不声称已在人为中断的真实GPU任务中演练。

账本签名包含模型、图片索引、输入／源码哈希、原权重及配置。原始捕获先写临时文件，完成序列化与flush/fsync后原子提交。已验收且SHA／签名相符的项跳过；有完整捕获但统计失败时优先离线恢复；只有“已开始”而无完整捕获时判为执行状态未知并停止，不猜测或自动重复前向。

原进程可能在调用完成与原子提交之间中断，不能承诺任意故障下都能自动完成且严格一次；代码明确避免自动重复已验收／有完整捕获的项，并对未知状态停止报告。

旧六图复用来源字段保持原样，例如原六图试运行证据中的“current_server_or_weight_file_rechecked=False”描述的是当时本地阶段。本次实际服务器预检见全量目录preflight/，不能把旧来源字段误读为本轮没有检查服务器和权重。

## 8. 实际命令、下载和主要路径

服务器命令（已完成，exit_code=0）：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B -X utf8 D:\vision-learning-server\week09\day04\capture_full_val.py --output D:\vision-learning-server\week09\day04\outputs\full_val_stage_run01
~~~

日志末尾为FULL_VAL_COMPLETED、PROCESS_SCRIPT_EXIT 0；外层PowerShell记录FULL_VAL_PROCESS_EXIT_CODE=0。首项030001有FIRST_FORMAL_IMAGE_BOTH_MODELS_ACCEPTED标记，纳入288次。

压缩包206895547字节，服务器与本地SHA256均为：

`4c13eb117e9a250e662426cf3a80560753030c3515b00ce8f5116e569f75c6df`

1198项产物逐文件SHA匹配。传输记录：`D:\python\test\vision-learning\week09\day04\checks\full_val_transfer_run01\verification.json`。

本地全量目录：`D:\python\test\vision-learning\week09\day04\outputs\full_val_stage_run01`，包含：

- per_gt_stage.csv：832个模型×GT项的七阶段，共5824行；float32／float64另列。
- stage_summary.csv、never_covered_summary.csv、gt_coverage_sequences.csv、never_covered_gt.csv：整体／场景、覆盖丢失／新增／部分流失及完整序列。
- paired_per_gt.csv、paired_summary.csv、net_difference.csv：配对和算术净差分解。
- day01_per_gt_comparison.csv、verification.json：832项终点对照及真实差值。兼容旧统计接口的pilot_max_iou字段在这里指本次完整val终点IoU，不代表只统计六图。
- forward_ledger.csv、reuse_evidence.json：300个模型＋图片项及288次新前向／12项复用来源。
- items/block_000至block_014下每个模型＋stem的原始捕获、签名账本及验收；reuse/下12份旧捕获。
- attempts/attempt_001/run.log、run_metadata.json、execution_status.json：真实完整过程、环境、源文件／保护哈希及退出码；process_status_attempt001.json保存实际进程退出码。
- preflight/：服务器26项人工核验、环境与Day01参照哈希、470项输入／源码清单。source/保存服务器实际使用的Day04源码副本。
- full_val_report.md、artifact_manifest.json：原始中文统计报告和下载产物清单。

本地独立核验命令（只读原始坐标、CPU计算，无模型／checkpoint／NMS）：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day04/verify_full_val.py --capture-dir week09/day04/outputs/full_val_stage_run01 --output week09/day04/checks/independent_full_val_run01
~~~

该脚本复用verify_pilot.py中的独立面积公式／汇总核验，内存中仅适配服务器到本地的路径：原服务器保护快照前后相等单独核对，本地历史文件按已保存六图试运行快照核对。不会修改下载产物、旧脚本或原元数据文件。原始索引决定NMS保留集合，不运行CPU NMS替代原GPU决策。

真实独立验收结果：status=passed、exit_code=0；5824行逐GT统计、832个模型×GT项、140行阶段汇总及完整覆盖序列／配对／净差表通过，288次新前向和12项复用来源核对通过。独立Python binary64面积公式与float64诊断最大IoU误差为0，容差1e-12；1198项产物SHA通过，服务器1048项保护快照前后相同，本地256项历史文件与原试运行快照一致。全程无checkpoint加载、模型前向或NMS执行。

## 9. 给初学者的解释与结论边界

这次已经从“六个挑选的GT”推进到“完整val的全部GT”。我们知道有些GT在解码阶段就没有好框；有些曾有好框，后来在Top-N、score、NMS或最终预算处全部丢失；还有些只是减少了重复好框，GT仍覆盖。裁剪能够帮助边界目标首次达标，不能把它当作单纯删框阶段。

这些是固定模型下的操作事实与描述统计。NMS阶段Fusion的覆盖丢失比IR多10个GT，解释了净差如何在计数上变化；没有证明为什么objectness／回归／融合特征形成这种差别，也没有记录实际抑制者配对。RPN覆盖不是最终检测AP75，当前不能把最终AP75差异全部归因于某个阶段。

本轮不启动模型改进实验，不调阈值或预算。全量验收及中文解释完成后停止，等待用户反馈。
