# Week11 Day06 · 全验证集 RoI 阶段统计

**已完成。** 离线读取 Day05 的 300 份捕获，对每个模型的 416 个标注目标逐阶段统计。候选与同一个标注目标的 IoU ≥ 0.75 时计为达标。一个目标只要还有至少一个达标框，就记为有覆盖。

## 做了什么，为什么

按实际 RoI 行和筛选索引追踪候选，生成 7,488 行逐阶段记录和 416 行 IR/Fusion 同目标对照。这样可以区分「从未有好框」「回归首次产生好框」和「已有好框在筛选时丢失」。重新核对 Day01 的四个模型×案例，关键阶段的达标数量和最大 IoU 与原表一致。

## 怎么读结果

| 阶段 | IR 覆盖目标数 | Fusion 覆盖目标数 |
|---|---:|---:|
| RPN 输出 | 302 | 285 |
| RoI 回归解码后 | 323 | 326 |
| 类别 NMS 前 | 322 | 324 |
| 类别 NMS 后及最终输出 | 225 | 200 |

- **回归阶段同时有所得和所失。** IR 新覆盖 54 个、失去 33 个，因此是 `302 + 54 − 33 = 323`。Fusion 新覆盖 60 个、失去 19 个，因此是 `285 + 60 − 19 = 326`。
- **本次总覆盖差在类别 NMS 阶段反转。** Fusion 在 NMS 前多覆盖 2 个，NMS 后少覆盖 25 个。IR 丢失 `97/322`（30.1%），Fusion 丢失 `124/324`（38.3%）。分数筛选分别少了 1 个和 2 个目标，100 框预算没有继续减少覆盖。
- **配对比较使用同一分母。** 在两模型 NMS 前都覆盖的 297 个目标中，两者都保住 167 个、都丢 55 个、仅 IR 丢 25 个、仅 Fusion 丢 50 个。相同目标上的差异与各自全体的趋势一致。

这是固定验证集上的目标几何覆盖。最高分框的实际去留已经逐目标保存。多人图的最高分框可能属于另一人。要统计某个高分框首次抑制了哪些达标框，需要另外追踪逐次抑制关系。AP75 则还需要评估时的一对一匹配和全局分数排序。

## 运行与核验

本轮读取候选捕获 300 份，完整模型前向、checkpoint 加载、独立 NMS 重放均为 0。运行退出码 0。逐项核对输出 SHA256，并从生成的逐目标表独立重算覆盖总量和 NMS 配对四格计数。

```powershell
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week11/day06/analyze_full_val.py' --output 'week11/day06/outputs/full_val_roi_analysis_run01'
```

- [全阶段汇总](outputs/full_val_roi_analysis_run01/stage_summary.csv)与[逐目标阶段表](outputs/full_val_roi_analysis_run01/per_gt_stage.csv)
- [两模型配对表](outputs/full_val_roi_analysis_run01/paired_per_gt.csv)与[汇总 JSON](outputs/full_val_roi_analysis_run01/summary.json)
- [最高分类分数与最高 IoU 对照](outputs/full_val_roi_analysis_run01/per_gt_coverage.csv)与[运行核验](outputs/full_val_roi_analysis_run01/verification.json)
