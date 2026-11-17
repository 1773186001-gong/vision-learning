# Day03：030269/G1 单图RPN筛选阶段追踪

仅完整validation图片030269；指定GT索引1（0起始）。IR/Fusion各一次原eval前向，不训练、不使用test、不改预算。

## 各阶段实际观测

f32列：与Day02一致的CPU float32 torchvision.box_iou；f64列：核心模块float64诊断IoU，两者不可混用。
候选ID稳定对应原展平索引，表内最佳框每阶段重新选取，不表示始终是同一框。

|模型|阶段|候选数|最大IoU f32 CPU|≥0.75数 f32|最大IoU f64|≥0.75数 f64|
|---|---|---:|---:|---:|---:|---:|
|IR|decoded|13500|0.793667018|2|0.793667077|2|
|IR|pre_nms_topn|450|0.793667018|2|0.793667077|2|
|IR|clipped|450|0.793667018|2|0.793667077|2|
|IR|small_box_filtered|450|0.793667018|2|0.793667077|2|
|IR|score_filtered|109|0.793667018|2|0.793667077|2|
|IR|layer_nms|57|0.793667018|2|0.793667077|2|
|IR|post_nms_topn|57|0.793667018|2|0.793667077|2|
|Fusion|decoded|13500|0.750537336|1|0.750537370|1|
|Fusion|pre_nms_topn|450|0.750537336|1|0.750537370|1|
|Fusion|clipped|450|0.750537336|1|0.750537370|1|
|Fusion|small_box_filtered|450|0.750537336|1|0.750537370|1|
|Fusion|score_filtered|158|0.750537336|1|0.750537370|1|
|Fusion|layer_nms|105|0.605291903|0|0.605291866|0|
|Fusion|post_nms_topn|105|0.605291903|0|0.605291866|0|

## 高IoU框是否曾存在、何时全部消失

- IR（历史f32口径）：有高IoU候选的阶段为['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；由有到无的转变为[]。
  没有记录到由有到无的阶段转变；这不等于证明某种上游机制。
  f64诊断口径的由有到无转变：[]。
- Fusion（历史f32口径）：有高IoU候选的阶段为['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；由有到无的转变为[{'from': 'score_filtered', 'to': 'layer_nms'}]。
  第一次全部消失发生于layer_nms；需同时查看完整阶段表，不能忽略后续是否恢复。
  f64诊断口径的由有到无转变：[{'from': 'score_filtered', 'to': 'layer_nms'}]。

## 验收与边界

重放的NMS输入、最终框、顺序和分数与同次原方法精确一致；最终框与RPN forward实际输出精确一致。阶段表/固定ID轨迹、原始捕获均保存，未配对猜测NMS抑制者。
- IR：Day02 f32 IoU=0.7936670184135437，本次=0.7936670184135437，实际差值=0；atol=1e-6、rtol=0；包装/hooks恢复=True。
- Fusion：Day02 f32 IoU=0.6052919030189514，本次=0.6052919030189514，实际差值=0；atol=1e-6、rtol=0；包装/hooks恢复=True。

这是一个GT的阶段观察，不能推广到六图或整个验证集。阶段索引能证明某个候选在哪个操作被删除，不能证明融合特征或回归就是根因；原NMS只返回保留索引，未记录实际抑制配对。
本次从解码候选开始，不覆盖anchor回归前的机制。对于未通过Top-N的框没有作反事实裁剪冒充实际阶段。
受保护输入、权重、模型持久state及配置前后校验见run_metadata.json和各模型verification文件。

## 实际命令

~~~text
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day03\capture_single_val.py --device cuda:0 --output D:\vision-learning-server\week09\day03\outputs\single_030269_run01
~~~

原始捕获：IR_raw_capture.pth、Fusion_raw_capture.pth。阶段表、候选目录和相关ID轨迹使用对应模型前缀。
