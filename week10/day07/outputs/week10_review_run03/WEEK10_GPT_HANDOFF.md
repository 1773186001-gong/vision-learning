# Week10权威技术交接

范围：只含这条Week10固定任务链。Day01—Day06实际产物、日志、退出码、manifest与来源已复核后才整理。Day编号是学习任务编号，实际时间/环境/命令见execution_ledger.json；缺失起止时间保留null，不编造工作时长。

## Day01
问题：为何同L/T仍可有不同结局？
为什么与方法：还原实际候选比继续增设指标更能回答问题。复用Day04 GPU候选，仅离线核对4项。
事实：T只计NMS前直接高分竞争关系，未表达竞争者自身是否保留。030078双方L2/T1，IR有8好框而Fusion有2；不能把摘要值当成完整结构。
边界：几何边不能命名GPU实际抑制事件。

## Day02
问题：被删竞争者如何与好框存活相容？
为什么与方法：完整输入独立重放后比较完整序列，避免只验证目标。两项RPN主重放，无新前向。
事实：030129的8862在step0首次删除8872；030078的8847在step4首次删除8841。原输出分别51/36，全序列匹配；各自好框8877/9141保留。
边界：步号与首次删除者是重放解释，不是GPU事件。

## Day03
问题：RPN覆盖是否传递到最终检测？
为什么与方法：两图×IR/Fusion4次CPU同次捕获RPN与完整输出。
事实：030129/IR的RPN好框仍在，最终最大IoU仅0.700329，且对应分类分数0.002493；最高分框IoU约0.598007。历史GPU/CPU数值与部分排序不同。
边界：两个最大IoU不是同一候选身份，不能拼接历史链或推断RoI根因。

## Day04
问题：030129/IR的8877到底去了哪里？
为什么与方法：单次CPU前向用真实RoI行与实际索引建立身份链，精确复现Day03。
事实：8877从0.766334回归到0.625646，分数0.995233，随后在类别NMS删除；另外5框回归达标，最高0.908178，经过分数和小框过滤，NMS后全部丢失；最终最贴合来自2571。
边界：定位下降和筛选删除不同；仍不能只凭原索引给出内核实际抑制者。

## Day05
问题：5个回归好框为何不在原NMS输出？
为什么与方法：读取44个原RoI类别NMS输入，按分类分数与类别分组独立重放一次。
事实：完整匹配3个原索引[0,7,23]：4个好框由8862在step0首次删除，8571由2571在step2首次删除；8877为非好框参照也由8862删除。
边界：这只是经完整输出核验的独立重放归因，不是内核事件跟踪，不是干预因果证据。

## Day06
问题：成功覆盖是否就是原RPN好框一直保留？
为什么与方法：030078/IR单次前向精确复现Day03，030129仅读表对照。
事实：9141从0.757896改善到0.794379，分类分数0.945614，但被NMS删除。最终达标来自8847：RPN0.723833→回归0.758343，分数0.981489。两例NMS前各5好框，之后1与0。
边界：保住0.75覆盖不等于保住最高IoU框；两例均IR，不能写成Fusion内部失败根因已解释。

## 计数与状态

原字段去重后累计：完整validation前向6次、checkpoint加载4次、Fusion恢复结构探测1次、独立主NMS重放3次。唯一模型×图片组合4项，实际执行6次（两项IR各复现一次）；候选张量读取不是checkpoint加载。原模型内部NMS与独立重放分开，原NMS缺少统一独立计数时不编造总数。训练/test推理/服务器连接均为0；部分天没有统一零计数字段，依据执行源码控制流与限定运行，详见账本限制说明。

## 精确证据入口

- E01：同L/T不等于完整结构与去留相同；030078双方L2 T1；源文件`D:\python\test\vision-learning\week10\day01\outputs\candidate_review_run01\case_summary.csv`；行身份`{"stem": "030078"}`；完整精确字段见evidence_index.json。
- E02：030129/IR好框8877保留，直接竞争者8872删除；源文件`D:\python\test\vision-learning\week10\day01\outputs\candidate_review_run01\selected_candidates.csv`；行身份`{"model": "IR", "stem": "030129"}`；完整精确字段见evidence_index.json。
- E03：030078/IR好框9141保留，竞争者8841删除；源文件`D:\python\test\vision-learning\week10\day01\outputs\candidate_review_run01\selected_candidates.csv`；行身份`{"model": "IR", "stem": "030078"}`；完整精确字段见evidence_index.json。
- E04：两项上游几何与原去留；源文件`D:\python\test\vision-learning\week10\day02\outputs\upstream_replay_run01\case_relations.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E05：完整RPN重放的步骤与首次删除者；源文件`D:\python\test\vision-learning\week10\day02\outputs\upstream_replay_run01\replay_steps.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E06：80/65完整输入分别匹配51/36完整输出序列；源文件`D:\python\test\vision-learning\week10\day02\outputs\upstream_replay_run01\case_verification.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E07：4项CPU同次RPN与最终检测端点；源文件`D:\python\test\vision-learning\week10\day03\outputs\rpn_final_comparison_run01\summary.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E08：历史GPU与CPU坐标/分数及排序差异；源文件`D:\python\test\vision-learning\week10\day03\outputs\rpn_final_comparison_run01\historical_differences.csv`；行身份`{"stem": "030129", "model": "IR"}`；完整精确字段见evidence_index.json。
- E09：8877回归定位下降后在类别NMS删除；源文件`D:\python\test\vision-learning\week10\day04\outputs\roi_trace_run01\trace_8877.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E10：另外5框回归达标，NMS处全部失去覆盖；源文件`D:\python\test\vision-learning\week10\day04\outputs\roi_trace_run01\stage_coverage.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E11：分数与背景展开重建精确接到实际小框输入；源文件`D:\python\test\vision-learning\week10\day04\outputs\roi_trace_run01\stage_candidates.csv`；行身份`{"stage": "score_filtered"}`；完整精确字段见evidence_index.json。
- E12：最终最贴合来自2571，最高分类分数来自8862；源文件`D:\python\test\vision-learning\week10\day04\outputs\roi_trace_run01\final_sources.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E13：5好框首次删除分为8862四项、2571一项，8877非好框参照；源文件`D:\python\test\vision-learning\week10\day05\outputs\roi_nms_replay_run01\focus_attributions.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E14：44输入完整重放匹配3个原输出及顺序；源文件`D:\python\test\vision-learning\week10\day05\outputs\roi_nms_replay_run01\output_comparison.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E15：9141回归改善到0.7943789958953857，随后被NMS删除；源文件`D:\python\test\vision-learning\week10\day06\outputs\success_case_trace_run01\trace_9141.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E16：最终达标来源8847，原RPN不达标，经回归达标；源文件`D:\python\test\vision-learning\week10\day06\outputs\success_case_trace_run01\final_sources.csv`；行身份`{"output_index": 0}`；完整精确字段见evidence_index.json。
- E17：保住0.75覆盖不等于保住最高IoU框；源文件`D:\python\test\vision-learning\week10\day06\outputs\success_case_trace_run01\stage_coverage.csv`；行身份`{}`；完整精确字段见evidence_index.json。
- E18：两例IR回归后各5好框，NMS后1与0；源文件`D:\python\test\vision-learning\week10\day06\outputs\success_case_trace_run01\two_case_comparison.csv`；行身份`{}`；完整精确字段见evidence_index.json。

## 统一解释规则

候选与GT的IoU≥0.75用于指定GT覆盖；候选与候选的NMS IoU阈值是RPN按层0.7、RoI按类别0.5，不能混用。RPN objectness不是RoI分类分数。重建已与实际张量相符不等于直接捕获；重放全输出相符不等于内核事件记录。两个端点最大IoU不能构造身份链。只保住覆盖不等于保住最佳定位。两例不构成总体率、因果效应或AP75结果。

## 未解决与下一步

Fusion内部RoI尚未追踪；模型为什么形成这些分数与回归量、对不同场景能否成立、模型改进能否提升性能、真正独立数据上的效果，均未解决。Week10诊断步骤完成不等于研究目标实现。优先补齐两图Fusion的最多2次前向，见week11_plan.md；本轮未执行该计划。

## 同次CPU终点对照（原表精度）

历史GPU值、CPU值分列；下表是端点诊断，不建立候选回归身份。

| 模型/图片/G0 | 历史RPN最大IoU | CPU RPN最大IoU | 最终最大IoU | 对应分类分数 | RPN/最终覆盖 |
|---|---:|---:|---:|---:|---|
| IR/030078/G0 | 0.7576177716255188 | 0.7578962445259094 | 0.7583428025245667 | 0.9814887642860413 | True/True |
| IR/030129/G0 | 0.7660496830940247 | 0.7663343548774719 | 0.7003291249275208 | 0.0024934702087193727 | True/False |
| Fusion/030078/G0 | 0.7057521343231201 | 0.7057053446769714 | 0.6152200698852539 | 0.9612109661102295 | False/False |
| Fusion/030129/G0 | 0.7339736819267273 | 0.7340067028999329 | 0.7362606525421143 | 0.9919335246086121 | False/False |

## 两例身份追踪对照（原表精度）

| IR图片/G0 | 原RPN好框 | 该框回归IoU/分类分数 | 该框最终去留 | 最终最贴合来源 | 最终IoU/分类分数 | NMS前/后好框数 |
|---|---|---|---|---|---|---|
| 030078/G0 | IR/030078/9141 | 0.7943789958953857 / 0.9456144571304321 | 删除 | IR/030078/8847 | 0.7583428025245667 / 0.9814887642860413 | 5/1 |
| 030129/G0 | IR/030129/8877 | 0.6256458759307861 / 0.9952331185340881 | 删除 | IR/030129/2571 | 0.7003291249275208 / 0.0024934702087193727 | 5/0 |

## 按天执行账本

| 学习任务 | validation完整前向 | checkpoint加载 | 结构探测 | 独立主NMS重放 |
|---|---:|---:|---:|---:|
| Day01 | 0 | 0 | 0 | 0 |
| Day02 | 0 | 0 | 0 | 2 |
| Day03 | 4 | 2 | 1 | 0 |
| Day04 | 1 | 1 | 0 | 0 |
| Day05 | 0 | 0 | 0 | 1 |
| Day06 | 1 | 1 | 0 | 0 |
| 合计 | 6 | 4 | 1 | 3 |

精确计数字段、命令、环境、时间缺口与去重依据见execution_ledger.json。Day07读取整理不增加上述执行次数。
