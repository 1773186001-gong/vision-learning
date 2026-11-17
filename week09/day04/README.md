# Week09 Day04：RPN逐阶段全GT统计的本地准备

当前阶段：后续授权的完整validation固定条件GPU捕获、合并统计、下载SHA及本地独立核验均已完成，退出码均为0。全量交付见DAY04_FULL_VAL_CAPTURE_DELIVERY.md和outputs/full_val_stage_run01；独立验收见checks/independent_full_val_run01/verification.json。

下文“本地准备／六图试运行”为先前阶段的历史记录，保持原运行和日志。六图按Day02优势和临界规则选取，不能用其比例估计全部val；完整val使用独立的新目录。

本轮只新增Day04代码、测试、文档和产物。没有连接服务器、加载checkpoint、执行新模型前向、运行NMS、安装依赖、修改阈值／预算、训练或运行test。Day03核心和历史产物没有修改。

## 实现与输入

- coverage_statistics.py：独立数值模块，逐GT计算七阶段最大IoU、达标候选数、覆盖及最佳原ID；产生整体／场景／配对／净差统计。
- offline_full_gt.py：只读Day03既有捕获，检查原索引、全部GT及复用证据，再执行六图全GT统计。没有导入旧模型恢复适配器；原模型方法不参与运行。
- test_coverage_statistics.py、run_unit_checks.py：人工小数据单元核验，不是LLVIP实验结果。
- verify_pilot.py：独立读取原坐标与导出表，重算float32及独立binary64几何、汇总、序列、配对和净差。
- FULL_VAL_CAPTURE_PLAN.md：尚未实施的全val采集／分块恢复方案。

真实输入为Day03合并目录`D:\python\test\vision-learning\week09\day03\outputs\remaining_five_merged_run01`下12份raw_capture，及Day01全val逐GT表／指标、Day02 selected_cases、原manifest和历史配置／哈希记录。

Day03每份捕获虽只对一个指定GT导出过诊断，但保存的原始字段中包含全图transform_gt_all、七阶段所有候选、原展平ID目录、真实Top-N／NMS索引及全部原NMS输出。因此可以不重新前向，离线统计每张图所有GT。

## 实际样本与主结果

|stem|场景|image_index|实际GT数／模型|
|---|---|---:|---:|
|030269|03|21|2|
|030308|03|24|2|
|030205|03|16|1|
|170100|17|120|2|
|130126|13|85|5|
|170455|17|145|3|

每模型6图、15个GT，空GT图0张。共30个模型×GT项、210行逐GT阶段统计。不是每模型416个GT的完整val结果。

CPU float32为与Day01／Day02一致的主口径；float64仅另列诊断。IoU≥0.75判定没有加容差。各GT分别判断是否有至少一个达标框，一个GT即使被多个框覆盖，覆盖分子仍只计1；无需一对一分配候选。

|阶段|IR覆盖|Fusion覆盖|IR−Fusion覆盖净差|净差较前阶段变化|
|---|---:|---:|---:|---:|
|decoded|13/15|10/15|3|不适用|
|逐层Top-N|13/15|10/15|3|0|
|裁剪|13/15|10/15|3|0|
|小框过滤|13/15|10/15|3|0|
|score过滤|12/15|10/15|2|−1|
|原NMS全部输出|11/15|9/15|2|0|
|最终预算截断|11/15|9/15|2|0|

始终未覆盖：IR=2/15、Fusion=5/15。完整GT名单在never_covered_gt.csv；其中IR为130126/G1、170455/G2，Fusion另有030205/G0、030308/G1、170455/G1。

IR在score过滤丢失1/13个前阶段已覆盖GT，在NMS丢失1/12；Fusion在NMS丢失1/10。这些是条件率，分母不同于所有有效GT。两模型在NMS各丢失1个GT，所以净差变化为0，不能由“净差没变”推出“该阶段没有丢失”。

NMS时两模型各有8个GT出现“达标框数量减少，但仍至少有一个达标框”。这与覆盖全部丢失不同。原NMS全部输出到150预算截断没有造成这15个GT的覆盖丢失。六图中没有实际出现裁剪后首次达标或覆盖恢复，但人工测试覆盖了这两种情况，流程不会跳过它们。

最后阶段配对：两者都覆盖7、仅IR4、仅Fusion2、两者都未覆盖2，分母15。场景03／13／17在本次各有5个GT：最终IR/Fusion分别为5/2、3/4、3/3。场景08没有进入这组选中图片；不能用本次场景分布代替完整val场景分布。

净差定义D=IR覆盖GT数−Fusion覆盖GT数，阶段变化满足：

`ΔD = (Fusion丢失GT−IR丢失GT) + (IR新增覆盖GT−Fusion新增覆盖GT)`。

这只是算术分解，不是因果贡献；可能互相抵消。整体比例直接按GT计数，不平均逐图比例。

## 复用核查已完成的范围

- 104项Day03产物与原artifact_manifest一致；12份原始捕获的坐标、顺序、分数及ID按真实Top-N／NMS索引核查。
- 逐层pre150、逐图post150、score0.05、NMS0.7、实际键0／1／pool和原RPN／transform配置与Day01一致。
- 当前本地数据／预处理／模型恢复源码和manifest哈希与Day01／Day02／Day03记录相符；本地几何库Python源码与Day01记录相符。
- 六图RGB／IR／XML的当前SHA与原捕获前及Day02证据相符；全图GT顺序、坐标和数量与Day01逐GT表及manifest一致；两模型GT相同。
- checkpoint哈希、模型持久state前后不变及hooks／包装恢复根据原捕获记录核对，不代表本轮读取或重新核验了服务器当前磁盘权重。未来GPU运行仍须重新预检。

完整证据在run02/reuse_evidence.json。CPU float32用于离线IoU，不重新计算GPU sigmoid分数来代替保存分数，也不重跑CPU NMS来替代原保留决策。

## 实际命令与验收

从项目根目录执行过：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day04/run_unit_checks.py --output week09/day04/checks/synthetic_unit_run01

& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day04/offline_full_gt.py --output week09/day04/outputs/six_image_all_gt_run02

& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day04/verify_pilot.py --capture-dir week09/day04/outputs/six_image_all_gt_run02 --output week09/day04/checks/independent_pilot_run01
~~~

14项人工测试通过，exit_code=0。覆盖多GT、空GT、空候选、多框覆盖同一GT、部分／全部流失、裁剪后首次达标、Top-N丢失后裁剪恢复、GT加权分母、净差增减、严格阈值、相同坐标不同ID及并列最佳按原阶段顺序选择等情况。

正式试运行run02退出码0：原六个指定GT的84行阶段统计与Day03一致；f32精确一致、f64最大值比较atol=1e-12，数量和最佳ID一致。全部30项末阶段与Day01的max_iou_top300及覆盖判定相符，实际IoU差值全部为0，起始atol=1e-6、rtol=0未放宽。原post预算150，因此Day01 Top300终点包含全部实际最终候选。

独立导出核验退出码0：210行逐GT统计、112行整体／场景阶段汇总及其覆盖序列、配对和净差表核对通过；独立Python binary64面积公式与导出的float64最大IoU误差0，验收atol=1e-12。256项历史文件前后与独立复核时均未变。报告模型调用和Python NMS的禁止守卫已启用。

首轮six_image_all_gt_run01退出码1：旧Day03 CSV的best_id字段与新表best_candidate_id字段名称不同，读取映射报KeyError，尚未完成验收。只修正新Day04读取映射后写入新run02；run01失败日志和状态完整保留，历史文件未变。没有掩盖失败、放宽数值容差或重复模型计算。

## 产物

有效输出目录：`D:\python\test\vision-learning\week09\day04\outputs\six_image_all_gt_run02`。

|文件|含义|
|---|---|
|per_gt_stage.csv|模型＋stem＋GT索引＋阶段；场景、GT坐标、两种精度最大IoU／达标数／覆盖／最佳ID、相邻阶段变化|
|per_image.csv|每图实际GT数、空GT标记和各阶段候选总数|
|stage_summary.csv|整体／场景、模型、精度分别汇总；显式覆盖及条件率分母|
|gt_coverage_sequences.csv|每个GT完整覆盖序列、首次达标、全部消失、恢复、始终未覆盖|
|never_covered_gt.csv、never_covered_summary.csv|始终未覆盖名单及分母／比例；两精度用独立行标明|
|paired_per_gt.csv、paired_summary.csv|按stem＋gt_index配对的四类覆盖状态及整体／场景计数|
|net_difference.csv|IR−Fusion净差及阶段算术分解|
|reuse_evidence.json|离线复用来源、输入／源码／配置／原权重记录证据|
|verification.json|84项Day03、30项Day01对照及全val参考和剩余范围|
|remaining_val_manifest.csv|未捕获144张val图的确定性清单；只是未来计划输入|
|pilot_report.md、run.log、run_metadata.json、execution_status.json|中文结果、真实过程、环境／源码SHA及退出状态|

人工日志位于checks/synthetic_unit_run01，独立核验位于checks/independent_pilot_run01。新运行目录必须不存在，不覆盖已有记录。

## 尚未实施

完整150图每模型416GT的阶段统计、144图新增捕获、空GT真实模型采集适配、持久分块恢复代码和真实服务器预检均未实施。多GT／空GT当前验证的是离线统计模块，不能声称已验证GPU采集对空GT的适配。

全val具体方案见FULL_VAL_CAPTURE_PLAN.md：有条件复用6图15GT，剩余144图401GT／模型，预计288次新增完整图片前向；每图每模型最多一次，失败后优先复用已落盘捕获，不重跑已验收图片。当前方案仅为计划。

本轮到此停止，等待用户反馈，不扩展NMS抑制者配对或新模型实验。

## 后续授权的完整val步骤（已完成）

此节描述六图试运行之后的全量授权，与上文历史步骤分开。原150图、每模型416GT，6图12份旧捕获复用，144图各模型一次，共288次新增原eval前向；030001两模型的首项验收计入288。全val真实空GT图为0，没有额外制造真实空GT前向。

新增agnostic_capture.py、capture_ledger.py、capture_full_val.py、full_val_analysis.py、test_full_capture.py、verify_full_val.py；Day03代码、原统计模块和旧运行未改。GT无关采集保存完整[G,4]，不使用假GT；原调用并原样返回，按真实索引重放核对。人工测试本地／服务器各26项通过，涵盖多GT／空GT、返回对象、正常／异常恢复、原子落盘、未知状态停止和跳过已验收项。真实运行未发生失败／未知或resume。

主口径CPU float32 IoU≥0.75：解码IR/Fusion为327/321，Top-N后326/319，裁剪后328/321，小框过滤不变，score后326/321，NMS后302/287，最终302/285，GT分母均416。净差依次6、7、7、7、5、15、17，是描述性算术分解，不是因果贡献。场景03／08／13／17最终净差为15／0／−1／3，分母分别93／121／131／71。

全部832项终点IoU与Day01实际差值0；5824行逐GT阶段表和140行汇总独立核验通过。完整val还发现双方各2个GT裁剪后首次达标，以及Fusion的081281/G7、130500/G5在最终150预算处丢失覆盖；不能将六图试运行的预算观察推广全val。

有效新目录outputs/full_val_stage_run01包括原始捕获、300项forward_ledger.csv、复用来源、全部统计表、full_val_report.md、preflight/、source/、attempts/attempt_001/run.log及实际退出状态。压缩包206895547字节、SHA256为4c13eb117e9a250e662426cf3a80560753030c3515b00ce8f5116e569f75c6df，1198项文件SHA核对通过；服务器1048项受保护文件与本地256项历史文件未变。

本地独立验收命令（已执行，退出码0）：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day04/verify_full_val.py --capture-dir week09/day04/outputs/full_val_stage_run01 --output week09/day04/checks/independent_full_val_run01
~~~

完整命令、保护证据、模型／场景解释与当前边界详见DAY04_FULL_VAL_CAPTURE_DELIVERY.md。本次全量授权已完成，停止等待反馈，不调参、训练、运行test或启动模型改进实验。
