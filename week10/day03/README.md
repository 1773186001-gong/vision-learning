# Week10 Day03：同次RPN与最终检测终点对照

仅030078/G0、030129/G0，各IR/Fusion一次CPU eval/inference_mode完整前向；各模型只恢复一次。严格复用ir_common.checkpoint_model和fusion_model.restore，不下载权重。Fusion恢复自带的零张量backbone探测单列，不能算validation图片前向。

从项目根目录运行：

```powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week10/day03/compare_endpoints.py --output week10/day03/outputs/rpn_final_comparison_run01
```

输出目录必须不存在。原始捕获在每次前向后立即保存；发生错误不自动重复前向。完整输出、RPN flat映射、配置、状态SHA、hook恢复证据都保留。报告只作指定GT的几何覆盖诊断，不是AP75/一对一TP；不把端点最大IoU匹配当作内部身份追踪。任何CPU/GPU差异原样导出，不调参或用容差强求复现。final_boxes在RPN记录中仍指RPN候选，最终检测另存model_output。

产物：summary.csv、rpn_candidates.csv、final_detections.csv、historical_differences.csv、每项raw_capture.pth、两例PNG、case_report_zh.md、verification.json、input_provenance.json、model_checks.json、history_hashes.json、run_metadata.json、run.log、artifact_manifest.json。空集合最大IoU记null，覆盖False，框索引/分数记null。最终检测不加显示阈值，所有返回框均导出；图中按GT选框仅为诊断展示。
