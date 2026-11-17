# Week10 Day05：030129/IR最终类别NMS的完整离线重放

只读取Day04的`raw_capture.pth`，`map_location='cpu', weights_only=True`。这是候选捕获，不是模型checkpoint。唯一主输入取原`nms.args`中的boxes、分类scores、class labels及阈值，全部44项原样处理。RPN的flat及FPN层仅用于来源标识；主算法不接收GT、flat、好框标记、objectness、FPN层或原保留索引。

在项目根目录执行：

```powershell
& .\week10\day05\run_replay.ps1
```

默认新建`outputs/roi_nms_replay_run01`；已存在时拒绝覆盖。复跑必须明确指定新目录，例如`-Output week10/day05/outputs/roi_nms_replay_run02`，本任务不会自动重试或增加主重放。

新增代码：`replay_roi_nms.py`与`run_replay.ps1`。主函数`greedy_replay(boxes,scores,labels,threshold)`独立生成排序、逐次决定和保留序列。同分规则预先固定为分类分数降序、输入位置升序，不根据原输出选择。按本地实际小输入路径进行float32的max+1、类别偏移及坐标相加，IoU算式每步按binary32舍入，删除条件严格为IoU>0.5。即使类别全为1，也不省略坐标偏移。原完整返回索引只在重放完成后用于核验。

产物：

- `nms_inputs.csv`：全部44项完整身份、RoI行、来源flat、person分支、回归裁剪后坐标、分类分数、GT IoU、类别偏移。来源FPN层单列且不用于NMS分组。
- `replay_steps.csv`、`candidate_decisions.csv`、`replay_comparisons.csv`：每步选中及新删列表、每个输入唯一决定、实际进行过的成对比较。
- `output_comparison.csv`、`output_verification.json`：完整保留数量/索引序列/框/分数/类别/身份，原postprocess、100预算及model_output逐项对照。
- `focus_attributions.csv`：5个本阶段好框与8877参照的首次删除归因。
- `retained_focus_all_pairs.csv`、`possible_retained_relations.csv`：原保留框与6项的全部配对，以及所有满足同类别高分删除条件的可能关系，保留原坐标和偏移路径两种条件，区分首次归因。
- `all_pair_numeric_audit.csv`、`score_ties.csv`、`threshold_precision_audit.csv`：全部946个输入对的float32/float64、原坐标/偏移路径审计。1e-6仅标记，不放宽主阈值。浮点数值变化不等于判定变化，全部数值均保留。
- `case_report_zh.md`、`key_candidates.png`、`replay_process.png`、`image_context.png`、`figure_geometry.json`：中文报告与图；局部范围只用于展示，完整GT保留。
- `verification.json`、`input_provenance.json`、`implementation_evidence.json`、`boxes_source.py.txt`、`history_hashes.json`、`run_metadata.json`、`run.log`、`artifact_manifest.json`：核验、SHA、实现与环境、实际命令、次数、日志和退出码。外层命令和进程退出码另存同级`*_process.json/log`。

固定范围仅030129/IR/G0。主重放预计1次、完整输出核验1次；矩阵IoU和全输入对审计是额外几何核验，不是第二次NMS重放。实际次数以核验JSON为准。无模型创建/恢复、checkpoint加载、前向、训练、test、服务器连接、反事实或依赖安装下载；Week08、Week09及Week10 Day01—Day04做SHA前后检查。

若序列或数值核验不通过，保存差异并标记失败，不调整条件。即使全部匹配，也只能称“与Day04原CPU完整输出一致的独立重放解释”，不是直接记录的原CPU内核逐次抑制事件，不推广为全验证集或AP75结论。
