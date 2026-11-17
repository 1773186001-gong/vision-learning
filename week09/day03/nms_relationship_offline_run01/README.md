# Day03：固定六个删除候选的NMS保留框关系

本步骤已完成，只使用两份已有原始捕获作本地CPU几何计算。未连接服务器、加载checkpoint、执行模型前向、重新运行CPU/GPU NMS或改变阈值/预算。历史代码和结果不修改；新增代码与产物全部在本目录。

## 输入与实现

读取项目AGENTS.md、六例交付报告、捕获保存格式、verify_merged_capture.py，以及安装版TorchVision的batched_nms/nms说明和按层坐标偏移源码。通过shell检索研究记忆；当前事实采用真实捕获。没有使用研究记忆MCP。

- analyze_relationships.py：只加载030269/G1 Fusion及130126/G3 IR的raw_capture；根据真实原NMS输入、层、分数、保留索引形成关系表。
- verify_exports.py：独立读取CSV，按原候选索引取回双方坐标和分数，复算960对关系及全图GT重叠。
- results/：分析脚本的真实输出、中文报告、日志、环境和数值核验。
- independent_verification/：导出表独立复算的实际日志及退出状态。

保留框集合为原NMS全部输出：030269/Fusion共105个；130126/IR共171个。后者有21个随后被最终150预算截断，这21个仍完整保留在retained_inventory.csv及关系审计范围内。

关系须满足同层、保留框分数更高或完全同分、相互IoU严格大于原0.7阈值。保留原始分数，不重新排序或重跑NMS。另列CPU float64和按层偏移后float32几何诊断；分歧与阈值附近情况单独审计。本次没有同分或判定分歧，相关比较距离阈值均超过0.02，不存在1e-6范围内的边界关系。1e-6仅为审计标记，不是阈值放宽。

## 实际执行

在项目根目录执行：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day03/nms_relationship_offline_run01/analyze_relationships.py --output week09/day03/nms_relationship_offline_run01/results
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day03/nms_relationship_offline_run01/verify_exports.py --results week09/day03/nms_relationship_offline_run01/results --output week09/day03/nms_relationship_offline_run01/independent_verification
~~~

两条命令真实退出码均为0。分析日志末行为：

~~~text
OFFLINE_ANALYSIS_EXIT 0 historical_files_unchanged True
~~~

数值核验：960对比较、6条可能关系、276个原NMS保留框目录记录、39行全图GT重叠；全部复算通过。独立Python binary64面积公式与Torch float64最大差值为0，验收容差1e-12。104项原捕获清单SHA匹配，174项既有Day03文件前后及交付复查未变；独立核验前后12项本次分析输出未变。运行环境和源码SHA见run_metadata.json。

输出目录要求不存在。复跑必须使用新的本目录子目录，不覆盖当前日志；本步骤已完成，本轮无需复跑。

## 核心观察

以下都只是“可能相关的原保留框”，不是已证实的实际抑制者。完整候选ID、双侧坐标和实际保留位置见CSV。

|案例/模型|被删除达标ID|可能相关原保留ID|双方相互IoU（CPU float32）|保留框NMS位置（0起始）|
|---|---:|---:|---:|---:|
|030269/G1 Fusion|2201|2501|0.753271937370|2|
|130126/G3 IR|3216|3221|0.743190228939|0|
|130126/G3 IR|3516|3221|0.749724388123|0|
|130126/G3 IR|3521|3221|0.720879495144|0|
|130126/G3 IR|9216|9221|0.744562685490|1|
|130126/G3 IR|9516|9221|0.788245260715|1|

六条关系全部表现为：保留框的objectness更高，但对指定GT定位更差；三个相关保留框均通过最终预算截断。

- 030269/Fusion ID2501：score=0.9966658949851990，指定GT IoU=0.6052919030189514；删除ID2201为score=0.9004508256912231、IoU=0.7505373358726501。保留框宽/高约为GT的1.310/1.261倍，中心较GT向右0.650、向下4.696像素。
- 130126/IR ID3221：score=0.9997752308845520，指定GT IoU=0.6647025346755981；对应三个删除框IoU为0.818582356、0.875696659、0.868215621。该保留框中心较GT向左3.656、向上11.283像素，宽约为GT的0.899倍。
- 130126/IR ID9221：score=0.9995077848434448，指定GT IoU=0.5950712561607361；对应两个删除框IoU为0.791036189、0.760225832。该保留框中心较GT向左6.209、向上10.095像素，宽约为GT的0.910倍。

上述几何量由已有坐标计算，坐标范围为含padding的640×640。中心偏移方向为x向右、y向下；详细值在独立核验JSON。

全图其他GT的最大IoU：030269/Fusion ID2501为0；130126/IR ID3221为0.072479866445、ID9221为0.118666343391，最大值对应G4。这些重叠均低于它们与指定GT的重叠，不支持直接声称保留框转而定位了邻近GT。最大重叠索引始终是几何描述，不能确认为语义归属。

## 证据边界与下一项假设

原索引证实六个达标框在NMS阶段删除，并证实上述三个框属于原NMS输出。坐标证实它们之间满足同层、高分和重叠条件；分数与指定GT IoU的排序存在冲突。这些都是可重复计算的观察。

没有直接证据记录原GPU逐次抑制事件。即使每个删除ID只对应一条可能关系，也不能据此命名实际抑制者。candidate_summary.csv中的unexplained_by_requested_retained_relation仅表示“是否未找到满足本轮条件的保留框”，不表示实际删除机制已经解释完毕。

当前足以提出具体待验证假设：030269/Fusion的ID2501，以及130126/IR的ID3221、9221，可能直接参与了对应达标框的删除；其objectness更高、对指定GT定位更差的排序冲突，可能是这些个例高IoU候选流失的一环。仍需直接抑制记录或经过严格核验的独立归因来验证；本轮未实施贪心归因，也不能由此推广总体比例、证明融合特征根因或决定修改NMS/融合模块。

分析报告为results/analysis_report.md；实际关系表为results/possible_related_retained.csv。本步骤到此停止，等待用户反馈。
