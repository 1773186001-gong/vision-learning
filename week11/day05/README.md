# Week11 Day05 · 全验证集 RoI 捕获

**已完成。** 依据 Day04 的冻结清单，在本地 CPU 上为 IR 和 Fusion 收集完整的 RoI 阶段记录。300 个模型×图片项均通过核验，包含 296 次新增前向与 4 份旧捕获。两个模型各有 150 张图和同一批 416 个标注目标。

## 做了什么、为什么

Day01–Day03 只能解释两个选定案例。要在全验证集区分「回归后才达标」和「达标框在筛选时丢失」，必须保存最终检测之前的候选框和去留索引。旧 Week09 捕获覆盖 RPN，Day05 将证据补到 RoI 和模型最终输出。

每张图保存从 RPN 候选标识、实际 RoI 输入行、`person` 分支，到分数、小框、类别 NMS 和预算切片的索引。逐图用原模型的实际输入和返回核对重建索引，确认可以把同一个框跨阶段追下去。与 Day04 的标注坐标及模型配置核对。已有的四份完整捕获保留来源路径和 SHA256，直接纳入同一索引。

## 执行结果

| 模型 | 新捕获 | 复用 | 严格恢复权重 | 恢复时结构探测 |
|---|---:|---:|---:|---:|
| IR | 148 | 2 | 1 | 0 |
| Fusion | 148 | 2 | 1 | 1 |
| 合计 | **296** | **4** | **2** | **1** |

正式运行退出码为 0，用时约 137 秒（脚本计时范围见[核验记录](outputs/full_val_roi_capture_run01/verification.json)）。新捕获文件总计 619,802,704 字节。读回时逐项检查 300 份捕获的 SHA256，并对 IR、Fusion 各抽查第一份和最后一份新捕获，重新载入张量后核对 RoI 身份链。原始 `.pth` 文件由项目的 Git 忽略规则保存在本地。

这些记录可用于下一步对 416 个目标逐个统计：哪里首次出现达标框、哪里失去全部达标框、IR 和 Fusion 对同一个目标的差异。Day05 的产物是可核验的候选数据，整体覆盖率与阶段损失将在统计阶段计算。

## 运行与数据入口

在项目根目录执行。正式命令要求新的输出目录，每个新目录会触发一次完整采集。

```powershell
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week11/day05/capture_full_val.py' --preflight-only
& 'D:\python\.venv\Scripts\python.exe' -B -X utf8 'week11/day05/capture_full_val.py' --output 'week11/day05/outputs/full_val_roi_capture_run01'
```

- [冻结清单](../day04/outputs/full_val_scope_run02/scope.json)
- [300 项捕获索引](outputs/full_val_roi_capture_run01/capture_index.json)和[逐图账本](outputs/full_val_roi_capture_run01/item_ledger.jsonl)
- [运行参数和源码摘要](outputs/full_val_roi_capture_run01/run_metadata.json)和[最终核验](outputs/full_val_roi_capture_run01/verification.json)
