# Week11 技术交接｜IR 与 Fusion 的 RoI 覆盖诊断

**状态：Day01–Day07 已完成。** 本周从两张图的候选追踪扩展到完整验证集。在固定权重和推理配置下，Fusion 进入 RoI 类别 NMS 前覆盖 **324** 个目标，IR 覆盖 **322** 个。NMS 后变为 **200 对 225**，覆盖差在这一关反转。两张 Fusion 图的独立重放还查明了达标框的首次抑制者。全量统计给出了丢失的阶段和数量，具体抑制关系仍只核验了这两个案例。

## 研究问题与口径

Week10 已追踪 `030078/G0`、`030129/G0` 的 IR 候选，但 Fusion 只有 RPN 和最终检测两个端点。Week11 先补齐 Fusion 的 RoI 身份链，检查回归能不能把候选变成好框，以及这些框最后在哪里消失。再用完整验证集判断类别 NMS 的覆盖差是否只出现在选定案例里。

- **数据和身份。** 验证集 150 张图、416 个有效 `person` 标注。IR 和 Fusion 使用同一批标注，按 `stem + gt_index` 配对。候选框按同次运行的 `RPN flat → RoI 输入行 → person 分支 → 筛选索引 → 最终输出` 对齐，跨模型或跨运行的同名 `flat` 各自独立。
- **达标和覆盖。** 一个预测框对该 GT 的 IoU ≥ 0.75 就算达标。至少有一个达标框时，该 GT 在这一阶段算有覆盖。IoU 用 CPU float32，空候选集合记为无覆盖、最大 IoU 为 `null`。这个统计描述目标的几何覆盖。AP75 还需要一对一匹配及跨图的分数排序。
- **推理条件。** 原 IR、Fusion `best.pth`，`strict=True`、CPU `eval/inference_mode`，沿用 640×640 Letterbox。RPN 使用原分数阈值 0.05、按层 NMS 0.7 和 150/150 设置。RoI 严格按分类分数 `> 0.001` 筛选，类别 NMS 阈值 0.5，最终上限 100 框。类别 NMS 的 0.5 比较两个预测框，诊断里的 0.75 比较预测框和标注。权重路径、SHA256、模型状态摘要及完整配置见[冻结清单](../day04/outputs/full_val_scope_run02/scope.json)。

## 各日工作和结果

| Day | 做了什么 | 实际结果与来源 |
|---|---|---|
| Day01 | 捕获两张 Fusion 图的 RoI 内部阶段，用 Week10 IR 表做同图对照 | 完整 RPN 顺序、最终框/分数/类别及顺序均精确复现 Week10 Day03 CPU。两个 Fusion 目标的 RPN 达标数都是 0，回归后分别有 3、12 个框达标，全部在类别 NMS 后消失。[两图对照](../day01/outputs/fusion_roi_trace_run02/ir_fusion_comparison.csv) |
| Day02 | 对 `030129` 的 54 个原类别 NMS 输入做一次独立贪心重放 | 6 个保留索引及顺序与原结果一致。第 1 步最高分框 `2861` 移除全部 12 个达标框，包括定位最贴合的 `8571`。[案例报告](../day02/outputs/fusion_nms_030129_run01/case_report_zh.md) |
| Day03 | 用相同规则重放 `030078` 的 58 个输入 | 7 个保留索引及顺序与原结果一致。第 1 步最高分框 `2832` 移除全部 3 个达标框，包括 `2841`。[案例报告](../day03/outputs/fusion_nms_030078_run01/case_report_zh.md) |
| Day04 | 固定 150 张图、416 个标注、300 个模型×图片项，逐份核验可复用的历史捕获 | 4 份捕获通过。正式采集只需要 296 次新增前向。[范围和统计约定](../day04/README.md) |
| Day05 | 按冻结清单逐图捕获和核验 | IR/Fusion 各新增 148 项、各复用 2 项。300 项全部通过身份链核验，保存原始捕获的路径和 SHA256。[捕获索引](../day05/outputs/full_val_roi_capture_run01/capture_index.json) |
| Day06 | 从已有捕获统计阶段变化，按同一 GT 配对两个模型 | 得到 7,488 行阶段明细、832 个模型×GT 和 416 行配对。四个 Day01 模型×案例的关键阶段数量与原表一致。[结果汇总](../day06/outputs/full_val_roi_analysis_run01/summary.json) |
| Day07 | 复算配对四格，复核产物、清单来源和执行账本 | 59 个已有产物的 SHA256 与清单一致，配对结果复算一致。[本周核验](verification.json) |

## 关键结果

| 阶段或事件 | IR | Fusion |
|---|---:|---:|
| RPN 输出覆盖 | 302 | 285 |
| 回归首次产生覆盖 | +54 | +60 |
| 回归使原覆盖消失 | −33 | −19 |
| 回归解码后覆盖 | 323 | 326 |
| RoI 类别 NMS 前覆盖 | 322 | 324 |
| 类别 NMS 使覆盖消失 | 97/322（30.1%） | 124/324（38.3%） |
| 类别 NMS 后及最终覆盖 | 225 | 200 |

RoI 分数筛选使 IR 和 Fusion 分别少覆盖 1 个、2 个目标。小框筛选和 100 框上限没有进一步降低覆盖。Fusion 在 NMS 前领先 2 个，在这一关比 IR 多丢 27 个，所以最终少覆盖 25 个。[全阶段汇总](../day06/outputs/full_val_roi_analysis_run01/stage_summary.csv)

各模型的 NMS 前覆盖数不同，因此再比较**进入类别 NMS 前两个模型都覆盖的 297 个同一目标**。其中两者都保住 167 个、都丢 55 个、仅 IR 丢 25 个、仅 Fusion 丢 50 个。用相同的起点，Fusion 仍有更多目标失去覆盖。[逐目标配对表](../day06/outputs/full_val_roi_analysis_run01/paired_per_gt.csv)

两例重放说明分类排序和定位质量可能不一致。例如 `030129` 的 `8571` 对 G0 的 IoU 是 0.873180，分类分数 0.119025。先保留的 `2861` 分类分数为 0.991934，对 G0 的 IoU 为 0.736261。两个预测框彼此的 IoU 为 0.755406，超过类别 NMS 阈值 0.5。在与原完整输出一致的离线重放里，`2861` 因此先移除了 `8571`。原捕获保留的是 NMS 输入及输出，逐步抑制者来自重放。[030129 达标框逐项归因](../day02/outputs/fusion_nms_030129_run01/good_box_attributions.csv)

## 复查入口与执行账本

| 接手时要做的事 | 文件入口 |
|---|---|
| 看每个 GT 在各阶段有哪些达标框 | [逐 GT 阶段明细](../day06/outputs/full_val_roi_analysis_run01/per_gt_stage.csv) |
| 查看 NMS 前最高分框、最高 IoU 框及其原始去留 | [逐 GT 覆盖与候选身份](../day06/outputs/full_val_roi_analysis_run01/per_gt_coverage.csv) |
| 读取原始 RoI NMS 输入和保留索引 | [Day05 捕获索引](../day05/outputs/full_val_roi_capture_run01/capture_index.json)，按 `model + stem` 找 `path` 和 `sha256`，实际数据在 `.pth` 的 `nms.args` 与 `nms.result` |
| 复算现有指标 | [Day06 分析脚本](../day06/analyze_full_val.py)和[运行记录](../day06/outputs/full_val_roi_analysis_run01/verification.json) |
| 对照两例重放的完整输出 | [`030129` 核验](../day02/outputs/fusion_nms_030129_run01/output_comparison.json)、[`030078` 核验](../day03/outputs/fusion_nms_030078_run01/output_comparison.json) |

原始 `.pth` 产物保存在本地并受 Git 忽略规则约束。换工作区时先确认索引里的文件是否齐全。已有捕获可以直接用于离线复算，核对 `sha256` 后用 `torch.load(..., map_location='cpu', weights_only=True)` 读取。离线复算使用本地解释器，以新输出目录保留历史结果作为对照。

```powershell
# 在 D:\python\test\vision-learning 执行。该命令只复算已保存捕获。
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week11/day06/analyze_full_val.py' --output 'week11/day06/outputs/full_val_roi_analysis_run02'
```

| 执行阶段 | 验证图完整前向 | checkpoint 加载 | 恢复时结构探测 | 独立 NMS 重放 |
|---|---:|---:|---:|---:|
| Day01 首次尝试 | 0 | 1 | 1 | 0 |
| Day01 通过核验的运行 | 2 | 1 | 1 | 0 |
| Day02–Day03 | 0 | 0 | 0 | 2 |
| Day04 | 0 | 0 | 0 | 0 |
| Day05 | 296 | 2 | 1 | 0 |
| Day06–Day07 | 0 | 0 | 0 | 0 |
| **本周合计** | **298** | **4** | **3** | **2** |

Day01 首次在恢复模型后、图片前向前退出。配置中的元组和 JSON 列表直接比较导致误报，修正后两图均通过。Day04 首次预检也因尺寸的元组与列表表示不同出现误报，归一化后通过，两次预检均为只读。模型恢复、结构探测和图片完整前向分别计数。[Day01 失败记录](../day01/outputs/fusion_roi_trace_run01/verification.json) · [Day04 缺口记录](../day04/outputs/full_val_scope_run01/scope.json) · [Day07 账本复核](verification.json)

## 下一步要回答的问题

验证集里已知有 **97 个 IR、124 个 Fusion** 目标在类别 NMS 失去覆盖。要查的是，在这些目标中，图内最高分框首次移除了该目标的**全部、一部分或零个**达标框，各有多少。这三个类别的分母是各模型的 NMS 丢失目标数。另用 NMS 前双方都覆盖的 297 个目标做配对对照。

建议先把丢失队列和 `model + stem` 对应的原始捕获固定下来。同一图片的多个目标共用一次完整 NMS 重放，原保留索引及顺序核验通过后再统计首次抑制者。并列分数、输出不一致或证据缺失的项保留原因，单列核验缺口。该分析可从现有捕获完成，所需重放次数由去重后的图片队列决定。验证集上的描述性结果与独立数据上的泛化结论分别报告。

交接约定：用中文，每次推进一个明确步骤。每个 Day 完成后主动说明做了什么、为什么这样做，以及结果支持到哪里。
