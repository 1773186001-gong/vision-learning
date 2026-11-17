# Week09 Day06：NMS前跨层达标支持与覆盖存活

本轮已完成一个固定的离线假设检验。失败模型的支持层数更少，只出现在31个不对称GT中的7个；23个相同，1个相反。L提供有限的描述性信息，不能解释多数同GT上的成败，也不足以支持立即修改网络。

## 实际范围与唯一口径

母集严格采用Day05核验后的297个共同风险GT（140张validation图片）：IR与Fusion在score_filtered阶段均至少有一个CPU float32 IoU≥0.75候选。不是完整416个GT，也不是重新选择的案例集。

L是在score_filtered阶段至少有一个达标候选的实际特征层数。真实键为0、1、pool；一个层有多个好框仍只计一层。层归属通过捕获的nms_levels、原展平索引、counts、layer_keys和catalogue交叉核对。没有根据框大小猜层。

阈值严格为IoU≥0.75，不加容差。594项模型×GT的L均为1、2或3；遇到L=0会报错，不静默丢弃。覆盖丢失使用原NMS全部输出索引判定，没有提前截到150，也没有重跑NMS。

主分析每份模型×图片捕获只加载一次，独立核验另作一次离线读取；两次各280份。没有连接服务器、加载checkpoint、执行模型前向、训练、调参、运行test或安装依赖。

## 同GT配对结果

ΔL＝失败模型L−保住覆盖模型L。负值符合假设方向，正值是反向案例。

|失败模型/范围|GT数|失败L更少|L相同|失败L更多|
|---|---:|---:|---:|---:|
|全部不对称GT|31|7（22.58%）|23（74.19%）|1（3.23%）|
|仅IR失败|11|1（9.09%）|9（81.82%）|1（9.09%）|
|仅Fusion失败|20|6（30.00%）|14（70.00%）|0|
|场景03|10|3|7|0|
|场景08|8|1|7|0|
|场景13|5|0|4|1|
|场景17|8|3|5|0|

297项丢失分类逐项复核：两者都丢失10、仅IR丢失11、仅Fusion丢失20、两者都保住256。31项完整名单在asymmetric_pairs.csv及day06_report.md，不筛掉相同或反向案例。

两类失败并非呈现相同程度的方向：Fusion失败时6/20为层数更少，IR失败时只有1/11，且IR有1项反例。多数仍是相同层数，不能把6/20解释成Fusion失败的完整机制。

反例为130126/G3：IR失败但L=2，Fusion保住而L=1。030269/G1则符合方向：Fusion失败L=1、IR保住L=2。两者都只是本轮配对观察，不是原GPU抑制配对或因果证据。

## 场景03

70个共同风险GT中，两模型都丢失1、仅IR丢失1、仅Fusion丢失9、都保住59。不对称的10项中，Fusion失败9项：3项L更少、6项L相同；IR失败的1项L相同。

符合方向的三个GT是030231/G0、030269/G1、030397/G1。场景03的结果仍以相同L为主，不能据此把该场景的Fusion额外覆盖损失全部归于层支持不足。其3/10的方向比例与其他场景有差异，但各场景不对称GT只有5至10个，不能作稳定的总体差异或因果推断。

## 完整297母集的分组背景

分母是“该模型、该L的共同风险GT数”，不是全部候选数或逐图比例平均。

|模型|L|风险GT数|NMS覆盖丢失数|条件丢失率|
|---|---:|---:|---:|---:|
|IR|1|34|9|26.47%|
|IR|2|174|12|6.90%|
|IR|3|89|0|0.00%|
|Fusion|1|33|15|45.45%|
|Fusion|2|189|15|7.94%|
|Fusion|3|75|0|0.00%|

两模型各组在此母集中呈现L较大、丢失率较低的描述性关联。但不同L组未必是相同GT，不能替代同GT配对。L=2仍分别丢失12和15项，说明多层支持不能保证存活；L=3本样本未见丢失，也不是未来绝不会丢失的保证。各层不假定独立。

场景03：IR的L=1/2/3分别丢失0/5、2/46、0/19；Fusion分别5/10、5/48、0/12。特别是0/5等小分母必须连同计数阅读。场景03/08/13/17全部计数及条件率保存在model_scene_L_summary.csv；程序对零分母留空，人工测试已覆盖此情形。

## 假设得到什么支持、仍不能解释什么

事实：主分析和独立核验一致，31项为7少、23同、1多；完整母集分组中较多层支持与较低丢失率并存。

判断：假设得到有限的描述性支持，尚不能作为多数失败的充分解释；若把假设加强为“失败侧总是支持层数更少”，反例已否定该说法。这里未实施显著性或因果检验。

直观地说，L只数“有几个层提供好框”。两侧都有两个层提供好框，最后仍可能一个保住、一个丢光；L不会告诉我们每个好框为何被删。

仍未解释：相同L为何会不同结局、候选与分数结构如何形成、原GPU实际逐次抑制关系、融合特征或框回归的根因、RoI及最终AP75差异。本轮没有临时增加分数间隔、目标大小等研究指标，也没有据此改网络。

## 新增文件

所有新增文件均在week09/day06；未修改Day01至Day05代码或产物。

- layer_support.py：唯一指标L、同GT配对、方向与分组汇总。
- analyze_layer_support.py：原始捕获读取、真实索引/层核对、CPU float32统计、输入及历史SHA保护、导出。
- test_layer_support.py、run_checks.py：必要人工数据测试与落盘日志。
- verify_layer_support.py：不用主统计函数，按原始坐标用独立面积公式及逐运算binary32舍入重算，另以累计层边界核对原索引所属层。
- README.md：本交付说明。

## 产物与证据

正式输出：outputs/layer_support_run01。

|文件|实际内容/行数|
|---|---|
|per_gt_support.csv|594项模型×GT，L、逐层计数、丢失状态|
|per_layer_support.csv|1782项模型×GT×层，原候选ID及数量|
|good_candidate_trace.csv|4046条达标候选×GT记录；不表示4046个不同候选|
|all_common_gt_pairs.csv|297项同GT配对|
|asymmetric_pairs.csv|31项完整不对称配对，失败/保住方向与ΔL|
|direction_summary.csv|15项整体/场景/失败模型汇总|
|model_scene_L_summary.csv|30项模型/场景/L背景计数及条件率|
|capture_provenance.json|280份原始捕获路径、SHA、真实层配置与所分析GT|
|analysis_summary.json、day06_report.md|机器摘要及含31项完整名单的中文报告|
|run_metadata.json、run.log、execution_status.json|真实命令、环境、来源SHA、历史保护、退出码|

输入来自Day04的outputs/full_val_stage_run01（forward_ledger.csv、per_gt_stage.csv、artifact_manifest.json及原始raw_capture.pth），Day05的outputs/full_val_nms_relations_run01/common_risk_pairing.csv、paired_summary.csv及独立核验记录。已先阅读适用AGENTS.md、Day04交付、Day05 README与相关源码，并通过shell检索研究记忆；结果以真实产物为准。

## 已执行命令与检查结果

工作目录：D:\python\test\vision-learning。

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day06/run_checks.py --output week09/day06/checks/synthetic_run01
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day06/analyze_layer_support.py --output week09/day06/outputs/layer_support_run01
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day06/verify_layer_support.py --analysis-dir week09/day06/outputs/layer_support_run01 --output week09/day06/checks/independent_run01
~~~

以上三项退出码均为0。

- 人工数据10项测试全通过：同层多好框只计一层、多层支持、严格0.75边界、L=0报错、真实层映射、失败方向、配对计数、零分母、风险分母、重复键拒绝。人工测试不是LLVIP实验结果。
- 主统计核对1198项Day04清单文件SHA，并保存输入SHA及代码SHA；1533个历史文件前后相同。
- 独立核验重读280份原始捕获，逐项核对594项L、1782项逐层数量和ID、4046条达标候选追溯、297项状态、31项方向及所有汇总。达标候选IoU独立计算与导出值的最大差为0，无阈值容差。
- 全部原NMS索引、顺序及实际层归属用于核验，未重新执行任何NMS。
- 初次独立脚本语法检查退出1：布尔表达式缺少括号，已仅修正Day06新脚本后检查通过。保留checks/syntax_check_run01/run.log，不以成功覆盖失败记录；另有一次对已结束分析会话的状态轮询返回Unknown process id，最终状态由已落盘execution_status.json及终端OFFLINE_EXIT 0确认。
- 文档第一次写入工具调用有JavaScript语法错误，未执行shell或写入文件；修正调用后才写入本文件。
- 主运行环境：Windows 11，Python 3.14.7，PyTorch 2.14.1+cpu，TorchVision 0.29.1+cpu。没有安装或升级依赖。

人工检查见checks/synthetic_run01/run.log及verification.json；独立核验见checks/independent_run01/run.log及verification.json。各运行目录使用排他创建，已有目录会拒绝覆盖。

本轮只完成这一个离线假设检验，到此停止，等待反馈。
