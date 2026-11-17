# Day05 已完成：双分支融合基线

GPU冒烟、10轮、val选第6轮、一次test、出图/错误分析及本地核验均通过，退出码0。正式交接见[DAY05_FUSION_TRAINING_GPT_HANDOFF.md](DAY05_FUSION_TRAINING_GPT_HANDOFF.md)。

测试P=0.889855、R=0.882184、AP50=0.903234、mAP50:95=0.439998；TP/FP/FN=307/38/41。低重合5、定位24、重复9。融合并非全面优于IR。

本地结果`outputs/fusion_concat_run01`，本地核验`checks/final_review_1793608106351600200/review.json`。已完成任务，不再运行部署启动命令或重新test。历史准备说明保存在checks/README_PREPARATION_ARCHIVE.md，六通道早期方案DAY05_PLAN.md已被正式双分支任务取代。

实际代码：fusion_model、fusion_common、train_fusion、evaluate_fusion、predict_fusion、preflight_fusion、check_day05及error_trace；后台start_fusion/monitor_fusion；收尾review_fusion_results和download_results。复现命令、全部Loss/评估与限制见正式交接。
