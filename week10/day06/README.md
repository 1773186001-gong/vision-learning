# Week10 Day06：030078/IR成功覆盖案例的RoI身份与两例对照

仅恢复一次IR best、strict=True、CPU eval/inference_mode，threads=2、seed=42。仅030078一张validation图片的一次完整模型前向；030129只读取Day04/Day05真实表格，不重新推理。不增加独立NMS重放、阈值扫描或反事实。

项目根目录执行：

```powershell
& .\week10\day06\run_trace.ps1
```

默认新建`outputs/success_case_trace_run01`，已有目录及外层进程记录拒绝覆盖。可通过`-Output`明确另一个新目录，本任务不自动重试。原始捕获在一次前向后立即保存；若后续核验失败，保留捕获和失败记录，不追加前向。

`trace_success.py`将Day04已核对的捕获思路复制到本目录，没有运行历史分析脚本作前置步骤。只导入原IR恢复/数据组件及只读RPN捕获类。原postprocess_detections照常调用并原样返回；窄范围捕获decode、softmax、clip、remove_small_boxes、batched_nms输入及结果，实际索引构成身份链。去背景/展开、严格score>0.001索引离线重建，必须精确等于下一实际小框过滤输入；预算切片精确核对原postprocess输出。不添加argmax筛选。

RPN候选flat→实际RoI输入行→person类别分支→过滤索引→实际最终输出。GT不参与身份匹配或推理筛选；不采用最近框或最大IoU猜来源。配置从实际模型读取，逐项比较Day03。36个proposal全部保存，背景/person分支分别导出。对“保住IoU≥0.75覆盖”与“保住NMS前最高IoU框”分别核查。

产物：

- `raw_capture.pth`：同次RPN阶段、实际RoI输入、predictor logits/regression、概率、decode/clip、小框/NMS索引、原postprocess和完整模型输出、完整GT、输入图像及运行身份。
- `proposal_branches.csv`：全部proposal与背景/person分支；完整身份、原objectness、GT IoU、logits、regression和分支概率/坐标。
- `stage_candidates.csv`、`stage_indices.json`、`stage_coverage.csv`：全阶段候选、索引与G0覆盖。
- `trace_9141.csv`、`final_good_source_traces.csv`、`focus_traces.csv`、`good_branch_outcomes.csv`：9141、全部最终达标分支及所有阶段曾达标分支的实际经历。
- `final_sources.csv`：全部实际最终输出的来源及其RPN/解码/clip定位。
- `day03_flat_comparison.csv`、`day03_final_comparison.csv`、`day03_reproduction.json`：按flat对齐的原RPN差异，顺序另外核对；最终完整输出按原顺序逐项比较，无数值容差。
- `success_summary.json`、`two_case_comparison.csv`、`comparison_sources.json`：来自真实表格的两例对照，跨图片使用完整IR/图片/flat身份。
- `case_report_zh.md`、`key_candidates.png`、`two_case_key_boxes.png`、`stage_flow.png`、`figure_geometry.json`：初学者报告与图。完整GT保留；640×640含padding，局部裁剪仅展示。
- `verification.json`、`model_checks.json`、`input_provenance.json`、源码快照、`history_hashes.json`、`run_metadata.json`、`run.log`、`artifact_manifest.json`；同级`*_process.json/log`保存外层实际命令、日志及进程退出码。

成功仅指G0最终几何覆盖，不代表图片或模型整体性能。两例不是控制变量实验；不估计总体比例、AP75或因果效应。若Day03未精确复现，身份链仍只解释本次新运行，核验JSON单独标记，不当作Day03内部过程。
