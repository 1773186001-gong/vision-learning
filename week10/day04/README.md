# Week10 Day04：030129/IR RoI身份追踪

仅恢复一次IR best，CPU eval/inference_mode完整前向一次；无结构探测、Fusion、重试或独立NMS重放。

```powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week10/day04/trace_roi.py --output week10/day04/outputs/roi_trace_run01
```

输出必须新建。调用原postprocess_detections；窄范围捕获原decode、softmax、clip、remove_small_boxes、batched_nms的输入/返回值。去背景/展开及分数索引离线重建，与实际小框过滤输入逐位核对；NMS、small索引直接捕获。没有以自写后处理替换原函数，没有根据GT猜身份。

产物包括raw_capture.pth、proposal_branches.csv、stage_candidates.csv、stage_coverage.csv、trace_8877.csv、final_sources.csv、day03_flat_comparison.csv、day03_final_comparison.csv、day03_reproduction.json、verification.json、provenance.json、model_checks.json、history_hashes.json、run_metadata.json、run.log、case_report_zh.md、identity_flow.png、key_boxes.png和artifact_manifest.json。源码快照roi_heads_source.py.txt用于审阅实际实现。空阶段最大IoU为null、覆盖False。

身份链为RPN保存的flat序列→实际RoI输入行→类别分支→实际筛选索引→最终输出，不作最近框匹配。GT只用于核查G0几何覆盖；全部GT保留作参照。原始1280×1024→Letterbox 640×640（上下padding64），内部transform与输出均640×640，不重复去padding。
