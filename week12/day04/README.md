# Week12 Day04 · Fusion 丢失队列批量重放

**批量重放已完成。** Fusion 新增 80 次离线重放，纳入 2 份历史重放和 1 份 Day02 试跑。83 个模型×图片项的完整保留索引和顺序都与原输出一致。124 个丢失目标中，122 个通过归因核验，2 个保留为未核验。新增模型前向和 checkpoint 加载均为 0。

## 做了什么，为什么

Day03 已用固定规则跑完 IR。今天对 Fusion 用同一个脚本、同一份冻结队列和相同的验收标准。这样后续可以区分两模型归因分布的差异与分析方法的差异。每张图只重放一次，再按该图不同 GT 的达标框分别统计。

## 怎么理解结果

| 分组 | 最高分框移除全部 | 移除部分 | 移除零个 | 未核验 |
|---|---:|---:|---:|---:|
| Fusion 全部 124 个 NMS 丢失目标 | 37 | 0 | 85 | 2 |
| NMS 前两模型都覆盖、后来 Fusion 丢失的 105 个目标 | 33 | 0 | 70 | 2 |

「零个」表示该目标的达标框由其他保留候选首次移除。Fusion 有 85 个目标属于这一类，因此 Week11 中最高分框直接移除全部达标框的案例只是其中一种情形。未核验的 2 个目标均来自 `030397`，不能归到「零个」。

`030397` 的完整保留顺序仍精确一致。重放里有一对预测框的 float32 IoU 为 `0.4999999403953552`，离类别 NMS 阈值 `0.5` 极近。按 Day02 预定的保守验收线，这张图的两个目标暂不分配首次抑制类别。临界检查用于标记归因缺口，实际重放仍用严格的 `IoU > 0.5`。这条记录尚未说明该临界比较确实改变了两目标的归因。[该图的临界检查](outputs/fusion_roi_nms_batch_run01/items/030397/attribution_checks.json) · [完整保留序列比较](outputs/fusion_roi_nms_batch_run01/items/030397/output_comparison.json)

两模型队列至此共 151 项，包含 148 次 Week12 新重放和 3 份历史重放。下一步可以用 297 个 NMS 前双方均覆盖的目标作配对，并在汇总中保留未核验项。Day04 复核了 323 个输出文件的 SHA256，同时从逐目标表复数各类计数。

```powershell
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week12/day03/replay_batch.py' --model Fusion --output 'week12/day04/outputs/fusion_roi_nms_batch_run01'
```

[Fusion 逐目标归因](outputs/fusion_roi_nms_batch_run01/target_attributions.csv) · [逐图重放索引](outputs/fusion_roi_nms_batch_run01/batch_index.json) · [核验记录](outputs/fusion_roi_nms_batch_run01/verification.json)
