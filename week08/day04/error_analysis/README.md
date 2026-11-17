# Day04 Step7：离线检测错误分析

仅使用已确认可信的IR/RGB预测缓存，在CPU执行匹配追踪、统计和图像绘制。导入既有公用模块不会构造模型；本脚本不调用build_model、checkpoint_model、infer或CUDA。没有服务器连接、训练或新模型预测。

在项目根目录执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day04\error_analysis\analyze_detection_errors.py
```

本次完整输出见`run.log`。已有`results`时脚本立即拒绝覆盖；重新分析应先将脚本复制到一个新的独立分析目录，并相应保持项目ROOT定位正确，不删除旧结果。

两缓存固定为Day03 IR run01与Day04 rgb_baseline_retrieved_20261030_213029_fa237fce。加载前对照历史保护清单/本地artifact清单验证SHA256，加载后检查150个同顺序stem、348个GT、坐标和Tensor类型。直接调用ir_common.match复算历史计数；不一致就抛出异常，在错误分类前停止。对原始150对图像通过既有Dataset和test预处理验证GT坐标，不能读取或不一致时停止可视化。

匹配协议：score≥0.5，IoU≥0.5，稳定分数降序；先屏蔽已用GT，再选剩余最大IoU。追踪实现与原函数逐图完整事件列表核对。prediction_id和GT编号为0起始，prediction_id是原缓存索引；image_id为文件stem，dataset_image_id是缓存/清单索引。

FP主类别互斥：duplicate优先（重合于最终已匹配GT，保存所有TP链接和双方IoU）；localization为0.1≤maxIoU<0.5；low_overlap为maxIoU<0.1；其余other。无GT的FP计入low_overlap，同时单独记录no_gt_fp且maxIoU=null，不重复计数。诊断分类不改变TP/FP结果。高分诊断预定义score≥0.9，不选择新的模型工作阈值。

结果说明：

- `ir/rgb_error_summary.json`：四类数量/比例、空GT、分数/IoU分位数、逐图FP分布、Top10及高分统计。
- `ir/rgb_matching_details.json`：每图GT及全部缓存预测、每个预测与全部GT的IoU、原预测索引、是否通过分数门槛、匹配GT、错误类型和重复链接。
- `per_image_errors.csv`：300行，两模态各150图，包含TP/FP/FN及四类计数。
- `per_prediction_errors.csv`：包含所有缓存框，selected=false的是低于阈值框，不进入FP分类；列表字段为JSON字符串。
- `fp_comparison.csv`、`paired_fp_difference.csv`：类别对照与同图差值。
- 三张统计图均包含绝对数量和占比；IoU图只统计存在GT的FP，缺失值数量在summary明确给出。
- `comparison_examples`：190001/190040/190474双模态同坐标对照及全部匹配明细。绿GT、蓝TP、橙重复、紫定位候选、红低重合、灰其他，底部有图例。
- `run_metadata.json`：实际环境、命令、缓存结构/哈希、匹配源码哈希及原图检查范围。

阅读`DAY04_ERROR_ANALYSIS_GPT_HANDOFF.md`获取真实结果与学习建议。几何分类只能提出诊断候选，不证明背景误识别、漏标或物理成因，也不证明RGB+IR融合有效。未进行逐像素跨传感器配准验证。
