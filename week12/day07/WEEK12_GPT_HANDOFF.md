# Week12 技术交接｜RoI 类别 NMS 丢失归因

**状态：Day01–Day07 已完成。** 在进入类别 NMS 前两模型都覆盖的 297 个相同目标中，Fusion 比 IR 多丢 25 个。最大的类别计数差来自**整图最高分框没有直接移除任何达标框**，Fusion 多 18 个。把这类目标按首次抑制步骤拆开，最大的一组是**第 3 步或更晚才首次移除达标框**，Fusion 多 12 个。

## 研究问题与范围

[Week11](../../week11/day07/WEEK11_GPT_HANDOFF.md) 已定位到 RoI 类别 NMS 是两模型覆盖差反转的阶段。两个 Fusion 个案中，整图最高分框都在第一步移除了该目标的全部达标框。Week12 要查的是这种过程在固定验证集的丢失目标里有多常见，以及其他保留框在什么时候移除达标框。

- **全丢失队列**：在验证集 150 张图、416 个 `person` 标注中，筛出 NMS 前有达标框、NMS 后失去全部达标框的目标。IR 97 条、Fusion 124 条，共 221 条模型×目标记录。按 `model + stem` 去重后，IR 68 项、Fusion 83 项。同一模型的一张图只重放一次，图内目标分别归因。
- **同目标对照**：取两模型在 NMS 前都覆盖的 297 个 `stem + gt_index`。其中 IR 丢 80 个，Fusion 丢 105 个。97/124 是各模型全丢失队列的分母，297 是同目标比较的共同分母。
- **两个阈值**：候选对标注的 IoU ≥ 0.75 算达标，原 RoI 类别 NMS 比较两个预测框，删除条件为 IoU 严格大于 0.50。沿用捕获中的框坐标、分类分数、类别及输入顺序。RPN 不参与本周归因重放。
- **归因规则**：对每个原先达标的候选，记录重放中第一次移除它的保留框。统计整图最高分框首次移除了该目标的全部、部分或零个达标框。输出顺序不符或归因存在歧义时记为未核验。「零个」表示覆盖仍然丢失，达标框均由其他保留框移除。

## 各日工作

| 学习日 | 做了什么，为什么 | 实际结果 |
|---|---|---|
| [Day01](../day01/README.md) | 先固定丢失队列与验收口径，检查历史结果是否能直接纳入 | 151 份原始捕获摘要匹配。3 份历史重放可复用，包括 Week10 的 `IR/030129` 和 Week11 的两个 Fusion 案例。 |
| [Day02](../day02/README.md) | 按冻结顺序选每个模型的首个待处理项，在新图上跑通核验 | `IR/030052` 与 `Fusion/030027` 完整保留顺序精确一致，共归因 4 个目标。 |
| [Day03](../day03/README.md) | 批量处理剩余 IR 项，建立同口径的归因结果 | 新增 66 次重放，IR 共 97 个丢失目标全部通过归因核验。 |
| [Day04](../day04/README.md) | 用原脚本和相同规则处理 Fusion | 新增 80 次重放。124 个目标中 122 个归因通过，`030397` 的 2 个保留为未核验。 |
| [Day05](../day05/README.md) | 按同图同 GT 配对，拆分两模型的覆盖损失 | 在共同分母 297 中，净差 +25 可拆为全部 +8、部分 −3、零个 +18、未核验 +2。 |
| [Day06](../day06/README.md) | 追踪「零个」这组，区分原始分类分数排名和 NMS 保留步骤 | 读 92 份决定表，核对 122 条模型×目标和 522 条达标框与首次抑制者的对应。 |
| [Day07](README.md) | 复算计数和执行账本，检查未核验项是否会改变结论 | 核对 612 个已有产物的 SHA256。未核验项的任何假设分类均不改变最大净差所在组。 |

## 关键结果

| 队列 | 模型 | 最高分移除全部 | 移除部分 | 移除零个 | 未核验 | 丢失合计 |
|---|---|---:|---:|---:|---:|---:|
| 各模型全丢失队列 | IR | 28 | 3 | 66 | 0 | 97 |
| 各模型全丢失队列 | Fusion | 37 | 0 | 85 | 2 | 124 |
| 共同覆盖的 297 个目标 | IR | 25 | 3 | 52 | 0 | 80 |
| 共同覆盖的 297 个目标 | Fusion | 33 | 0 | 70 | 2 | 105 |

297 个共同目标中，两模型都保住 167 个、都丢 55 个、仅 IR 丢 25 个、仅 Fusion 丢 50 个。仅 Fusion 丢失的 50 个里，31 个属于「最高分框移除零个」，18 个属于「全部」，1 个未核验。这说明只检查整图最高分框会漏掉大量直接抑制关系。[配对汇总](../day05/outputs/roi_nms_paired_run01/summary.json)

在共同覆盖队列里，进一步只看「最高分框移除零个」的 52 条 IR 和 70 条 Fusion 记录：

| 达标框首次被移除的步骤 | IR | Fusion | Fusion − IR |
|---|---:|---:|---:|
| 全部在第 2 步 | 26 | 31 | +5 |
| 部分在第 2 步，其余更晚 | 0 | 1 | +1 |
| 第 3 步或更晚才开始移除 | 26 | 38 | +12 |

Fusion `030116/G2` 给出了具体例子：候选 `1211` 在 NMS 开始前按分类分数排第 19，却是 NMS 第 3 个保留框。它在这一步首次移除了该目标的全部 5 个达标框。前面有高分候选被去掉，所以**原始分数排名和保留步骤是两个量**。[逐候选证据](../day06/outputs/other_suppressors_run01/other_suppressors_per_candidate.csv)

## 未核验项与结论边界

Fusion `030397` 的完整保留索引与顺序精确一致。重放实际比较的两个框（输入位置 6、47）的 float32 IoU 为 `0.4999999403953552`，落进预设的临界检查范围。按保守规则，该图的 G0、G1 保留为未核验。临界检查只标记不确定性，实际重放仍使用严格的 `IoU > 0.50`。[临界检查](../day04/outputs/fusion_roi_nms_batch_run01/items/030397/attribution_checks.json) · [原序列对比](../day04/outputs/fusion_roi_nms_batch_run01/items/030397/output_comparison.json)

Day07 检查了两个未核验项的所有假设归类。如果都归为「全部」，该组净差最多 +10，仍小于「零个」已核验的 +18。若两项均归为「零个」且都在第 2 步被全部移除，第 2 步这一组最多 +7，仍小于第 3 步或更晚的 +12。正式结果保留两项未核验。[边界复算](verification.json)

首次抑制者来自与原完整输出一致的离线重放。上述差值是固定验证集内的计数分解。「哪类更多」不能说明改动 NMS 后精度或误检会怎样变化，几何覆盖也与 AP75 的一对一匹配和全局分数排序有别。

## 复查入口和执行账本

| 要复查的内容 | 入口 |
|---|---|
| 丢失目标、原始达标候选编号、原捕获路径及 SHA256 | [冻结队列](../day01/outputs/frozen_roi_nms_queue_run01/replay_queue.json) |
| 历史重放与试跑证据 | [历史复用核验](../day01/outputs/frozen_roi_nms_queue_run01/reused_replay_checks.json)、[试跑索引](../day02/outputs/roi_nms_pilot_run01/pilot_index.json) |
| 每张图的重放结果与候选决定表入口 | [IR 批次索引](../day03/outputs/ir_roi_nms_batch_run01/batch_index.json)、[Fusion 批次索引](../day04/outputs/fusion_roi_nms_batch_run01/batch_index.json) |
| 同一个目标在两模型中的去留和类别 | [297 行配对表](../day05/outputs/roi_nms_paired_run01/paired_target_attributions.csv)、[配对交叉表](../day05/outputs/roi_nms_paired_run01/paired_outcome_matrix.csv) |
| 其他保留框在第几步移除候选 | [122 条逐目标记录](../day06/outputs/other_suppressors_run01/other_suppressors_per_gt.csv)、[522 条逐框记录](../day06/outputs/other_suppressors_run01/other_suppressors_per_candidate.csv) |

复查时按 `model + stem` 查冻结队列里的 `capture_path`，在 SHA256 匹配后读取原 `nms.args` 与 `nms.result`。批量重放沿用 [Day02 单图函数](../day02/replay_pilot.py)。复算已有表格时可直接使用下列命令，输出目录须为新目录。

```powershell
# 在 D:\python\test\vision-learning 执行，只读已有的决定表和结果表。
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week12/day05/summarize_paired.py' --output 'week12/day05/outputs/roi_nms_paired_run02'
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week12/day06/analyze_other_suppressors.py' --output 'week12/day06/outputs/other_suppressors_run02'
```

**执行账本**：Week12 新增离线重放 148 次（试跑 2、IR 批量 66、Fusion 批量 80），复用历史重放 3 次。151 项完整保留顺序全部匹配。221 条丢失目标记录中，219 条归因通过、2 条未核验。新增模型前向和 checkpoint 加载均为 0。[本周核验](verification.json)

**协作交接**：用中文，每次推进一个明确步骤。每个 Day 完成后主动讲清做了什么、为什么这样做，以及结果支持到哪里。
