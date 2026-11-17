# Week09 Day04：完整validation RPN逐阶段覆盖统计

固定原val、原best、原预处理及pre/post150、score0.05、NMS0.7。六图12份原捕获复用，剩余144图IR/Fusion各一次，共288次新增原eval前向；没有训练、test、调参或NMS抑制者配对。

主口径为CPU float32 IoU≥0.75；float64诊断另列CSV。覆盖以GT为单位，整体不平均逐图或逐场景比例。

## 实际范围与终点

{'IR': {'images': 150, 'valid_gt': 416, 'covered_075': 302}, 'Fusion': {'images': 150, 'valid_gt': 416, 'covered_075': 285}}
逐GT对照Day01共832项，最大绝对IoU差=0，atol=1e-6、rtol=0；覆盖判定不加容差。

## 全部阶段

|阶段|IR覆盖|Fusion覆盖|IR−Fusion净差|净差变化|
|---|---:|---:|---:|---:|
|decoded|327/416|321/416|6|None|
|pre_nms_topn|326/416|319/416|7|1|
|clipped|328/416|321/416|7|0|
|small_box_filtered|328/416|321/416|7|0|
|score_filtered|326/416|321/416|5|-2|
|layer_nms|302/416|287/416|15|10|
|post_nms_topn|302/416|285/416|17|2|

解码时IR−Fusion覆盖GT净差为6；最终净差为17。这描述达标覆盖差距，不直接归因为特征或回归。

净差变化=(Fusion丢失−IR丢失)+(IR新增覆盖−Fusion新增覆盖)，只是算术分解，不是因果贡献。净差相同也可能有双方丢失相互抵消。

## 丢失、出现与部分好框流失

|模型|阶段|丢失GT|前阶已覆盖GT|条件丢失率|新增覆盖GT|部分好框减少仍覆盖GT|
|---|---|---:|---:|---:|---:|---:|
|IR|pre_nms_topn|1|327|0.0030581039755351682|0|6|
|IR|clipped|0|326|0.0|2|0|
|IR|small_box_filtered|0|328|0.0|0|0|
|IR|score_filtered|2|328|0.006097560975609756|0|16|
|IR|layer_nms|24|326|0.0736196319018405|0|255|
|IR|post_nms_topn|0|302|0.0|0|1|
|Fusion|pre_nms_topn|2|321|0.006230529595015576|0|5|
|Fusion|clipped|0|319|0.0|2|0|
|Fusion|small_box_filtered|0|321|0.0|0|0|
|Fusion|score_filtered|0|321|0.0|0|10|
|Fusion|layer_nms|34|321|0.1059190031152648|0|241|
|Fusion|post_nms_topn|2|287|0.006968641114982578|0|0|

## 始终未覆盖与配对

- IR七阶段始终未覆盖：87/416。
- Fusion七阶段始终未覆盖：93/416。
- decoded：两者覆盖296、仅IR31、仅Fusion25、均无64；GT分母416。
- pre_nms_topn：两者覆盖295、仅IR31、仅Fusion24、均无66；GT分母416。
- clipped：两者覆盖298、仅IR30、仅Fusion23、均无65；GT分母416。
- small_box_filtered：两者覆盖298、仅IR30、仅Fusion23、均无65；GT分母416。
- score_filtered：两者覆盖297、仅IR29、仅Fusion24、均无66；GT分母416。
- layer_nms：两者覆盖256、仅IR46、仅Fusion31、均无83；GT分母416。
- post_nms_topn：两者覆盖255、仅IR47、仅Fusion30、均无84；GT分母416。

## 场景如何组成总体净差

|场景|GT分母|解码净差|最终净差|解码后净差变化|
|---|---:|---:|---:|---:|
|03|93|6|15|9|
|08|121|-2|0|2|
|13|131|0|-1|-1|
|17|71|2|3|1|

各场景净差相加组成总体净差，是GT计数的描述性分解。场景分母不同，不直接平均比例，也不能把场景净差称为某个模块的因果贡献。场景逐阶段变化见net_difference.csv。

## 证据与限制

原Top-N/NMS索引传播稳定候选ID；原方法调用并原样返回。新捕获在原设备精确重放NMS输入、最终坐标／顺序／分数，并与实际RPN输出一致；模型state、配置、权重和受保护文件前后校验及hooks恢复见逐项原捕获和run_metadata.json。
原NMS全部输出先保存，再计算最终150截断。裁剪后首次达标及恢复保留完整序列，不能只看decoded。空GT图单独记录，不伪造GT。
原操作事实与离线覆盖统计不等于根因；当前未做抑制者配对、独立贪心归因、融合模块或RoI因果分析。RPN覆盖不是最终检测Recall或AP75。
完整前向账本见forward_ledger.csv，六图来源见reuse_evidence.json，原始捕获逐项落盘。下载SHA和本地独立重算为单独检查记录。完成本轮后停止，等待反馈。
