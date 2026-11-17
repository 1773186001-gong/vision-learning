# Day05 Step8：离线定位质量分析

使用固定、已核验SHA256的三份预测缓存，CPU运行，不构造检测器、不加载模型checkpoint、不调用forward、不连接GPU或服务器。

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day05\localization_analysis\analyze_localization.py
```

完整终端输出见run.log。已有results时直接拒绝覆盖；复现应在独立的新分析目录运行，并保持ROOT项目定位正确，不删除已有产物。

先以原ir_common.metrics与历史test_metrics.json逐值相等作停止门槛；原match负责不同IoU重匹配，AP使用与原metrics相同的101点插值表达式，并断言AP50和10阈值均值一致。AP使用全部缓存分数，P/R匹配score≥.5。高分≥.9仅筛原匹配完成后的标签，不重新匹配。trace复用Day05保存的Day04原追踪函数，逐图与原match事件一致。

tp_iou_statistics.csv统计原固定匹配的真实TP IoU，标准差为总体ddof=0；matching_by_iou.csv每个IoU重新匹配，不能将前者的阈值计数当作后者。

high_confidence_analysis.csv的FP比例分母是全部高分预测，IoU≥.75比例分母为高分TP；FP仅记录maxIoU，不作为真实匹配IoU。高分明细和全部预测追踪保存为JSON。相关系数只对固定匹配TP计算，不是校准指标或因果证据。

paired_gt_iou_comparison.csv含295个共同GT的两模型原预测索引、坐标、分数、实际匹配IoU和Fusion−IR差值。paired_summary.json以abs(delta)≤1e-6定义近似相等，另列仅IR/仅Fusion/共同漏检。top10按绝对差排序。

7张统计图和8张案例图，案例含完整640图与相同坐标局部放大。近.5/.75案例选TP侧最近者，不宣称覆盖边界两侧；分数和精确IoU置于图下以减少文字遮挡。类别可能重复选择同一目标，保留真实选择清单与明细。RGB仅作统计辅助，案例聚焦IR/Fusion。

阅读DAY05_LOCALIZATION_ANALYSIS_GPT_HANDOFF.md获取实际结果。run_metadata.json记录环境、缓存路径/哈希、匹配与追踪代码哈希。结果不等于经过人工验证的根因，不改变历史模型、评估或部署阈值，不追加训练。
