# Week12 Day03 · IR 丢失队列批量重放

**已完成。** 处理 IR 剩余 66 个模型×图片项，纳入 1 份历史重放和 1 份 Day02 试跑。IR 队列的 68 个图片项完整的 NMS 保留索引及顺序全部一致，97 个丢失目标均通过归因核验。新增模型前向和 checkpoint 加载均为 0。

## 做了什么，为什么

Day02 已验证单图的重放和归因链。Day03 用同一规则覆盖冻结队列中的全部 IR 丢失目标。先形成 IR 自身的基线，之后才能在相同定义下比较 Fusion。每张图只重放一次，再将首次移除决定映射到图内的各个 GT。

## 怎么理解结果

统计图内最高分框首次移除了该目标的多少个原 NMS 前达标框。下表的「零个」表示该目标原有达标框均由其他保留候选首次移除。

| 分组 | 全部 | 部分 | 零个 | 未核验 |
|---|---:|---:|---:|---:|
| IR 全部 97 个 NMS 丢失目标 | 28 | 3 | 66 | 0 |
| NMS 前两模型都覆盖、后来 IR 丢失的 80 个目标 | 25 | 3 | 52 | 0 |

IR 队列有 66/97（约 68.0%）的目标落在「零个」。这种情况下，只看图内最高分框会错过真正首次移除好框的候选。这里的比例以本验证集上已经失去覆盖的 IR 目标为分母。等 Fusion 队列完成后，再用共同覆盖的目标做两模型对照。

本轮新增 66 次离线重放，读取 66 份原始候选捕获，逐项保留完整输出核验和决定表。重新校验了 267 个输出文件的 SHA256，并从逐目标表复数三类计数。

```powershell
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week12/day03/replay_batch.py' --model IR --output 'week12/day03/outputs/ir_roi_nms_batch_run01'
```

[IR 逐目标归因](outputs/ir_roi_nms_batch_run01/target_attributions.csv) · [逐图重放索引](outputs/ir_roi_nms_batch_run01/batch_index.json) · [核验记录](outputs/ir_roi_nms_batch_run01/verification.json)
