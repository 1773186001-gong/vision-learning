# Week09 Day04：已有六图全GT离线试运行

这只是Day03已选六图的全GT试运行，不是完整validation结果。六图存在选择偏差，不能估计全val比例。未连接服务器、加载checkpoint、执行新模型前向或NMS，也没有修改阈值／预算。

## 实际样本

|图片|场景|GT数（每模型）|
|---|---|---:|
|030269|03|2|
|030308|03|2|
|030205|03|1|
|170100|17|2|
|130126|13|5|
|170455|17|3|

IR：6图，15个GT，空GT图0张。

Fusion：6图，15个GT，空GT图0张。

## CPU float32历史一致口径

|阶段|IR覆盖/GT|Fusion覆盖/GT|IR−Fusion净差|净差较前阶段变化|
|---|---:|---:|---:|---:|
|decoded|13/15|10/15|3|None|
|pre_nms_topn|13/15|10/15|3|0|
|clipped|13/15|10/15|3|0|
|small_box_filtered|13/15|10/15|3|0|
|score_filtered|12/15|10/15|2|-1|
|layer_nms|11/15|9/15|2|0|
|post_nms_topn|11/15|9/15|2|0|

净差变化=(Fusion丢失−IR丢失)+(IR新增覆盖−Fusion新增覆盖)，只是算术分解，不是因果贡献。总体覆盖率按GT汇总，不平均逐图比例。

## 覆盖丢失及条件率

|模型|阶段|丢失GT|前阶段已覆盖GT|条件率|部分好框减少但仍覆盖GT|
|---|---|---:|---:|---:|---:|
|IR|pre_nms_topn|0|13|0.0|0|
|IR|clipped|0|13|0.0|0|
|IR|small_box_filtered|0|13|0.0|0|
|IR|score_filtered|1|13|0.07692307692307693|1|
|IR|layer_nms|1|12|0.08333333333333333|8|
|IR|post_nms_topn|0|11|0.0|0|
|Fusion|pre_nms_topn|0|10|0.0|0|
|Fusion|clipped|0|10|0.0|0|
|Fusion|small_box_filtered|0|10|0.0|0|
|Fusion|score_filtered|0|10|0.0|0|
|Fusion|layer_nms|1|10|0.1|8|
|Fusion|post_nms_topn|0|9|0.0|0|

## 七阶段始终未覆盖与配对

- IR始终未覆盖：2/15。逐GT名单见never_covered_gt.csv。
- Fusion始终未覆盖：5/15。逐GT名单见never_covered_gt.csv。
- decoded：两者覆盖10、仅IR3、仅Fusion0、均无2，GT分母15。
- pre_nms_topn：两者覆盖10、仅IR3、仅Fusion0、均无2，GT分母15。
- clipped：两者覆盖10、仅IR3、仅Fusion0、均无2，GT分母15。
- small_box_filtered：两者覆盖10、仅IR3、仅Fusion0、均无2，GT分母15。
- score_filtered：两者覆盖9、仅IR3、仅Fusion1、均无2，GT分母15。
- layer_nms：两者覆盖7、仅IR4、仅Fusion2、均无2，GT分母15。
- post_nms_topn：两者覆盖7、仅IR4、仅Fusion2、均无2，GT分母15。

完整场景表、float64诊断表及逐GT完整覆盖序列另存CSV；两种精度不混用。裁剪后的首次达标与恢复不会被忽略，0.75阈值没有加容差。候选数量减少不等于GT覆盖丢失。

## 真实验收与范围

原六个指定GT共84行与Day03逐项一致（f32精确、f64 atol=1e-12，数量与最佳ID一致）；全部GT共30项末阶段与Day01逐GT核对，实际差值见verification.json，不放宽1e-6容差。
全val参考来自既有Day01：{'IR': {'images': 150, 'valid_gt': 416, 'covered_075': 302}, 'Fusion': {'images': 150, 'valid_gt': 416, 'covered_075': 285}}。当前没有执行全val捕获。满足未来预检后，尚需144张图、预计288次新增完整图片前向。
原核心只支持一个指定GT的捕获期诊断，但保存的候选与GT无关、全图GT齐全，因此本次离线扩展到所有GT，不修改原模型或旧核心。未来应适配空GT采集，不能为占位而伪造GT；具体分块与恢复方案见FULL_VAL_CAPTURE_PLAN.md。
本次只说明这些选定图片内的分布与流水线是否能工作，不解释融合根因，不扩展NMS抑制者配对。完成本地试运行后停止，等待反馈。
