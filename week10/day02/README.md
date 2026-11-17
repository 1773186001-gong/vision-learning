# Week10 Day02：两个IR案例的上游关系与离线NMS重放

仅分析完整IR/030129（80候选）和IR/030078（65候选），追查8872和8841的删除关系。不运行Fusion重放，不新增研究指标，不做反事实删除实验，不进入RoI。

运行命令（项目根目录）：

```powershell
& .\week10\day02\run_replay.ps1
```

默认输出`outputs/upstream_replay_run01`。已存在的输出拒绝覆盖，可用`-Output week10/day02/outputs/upstream_replay_run02`明确指定新目录。

`replay_upstream.py`中`greedy_replay(boxes,scores,levels,threshold)`只接收原算法输入；先构造float32坐标偏移，再按原分数降序贪心处理。每一步显式记录选中框和新删除候选；不调用现成nms/batched_nms。GT、Day01标签和原保留索引均不参与决策。排序预先固定为分数降序、同分按输入位置；同分情况完整记录，有影响解释的歧义则不作无歧义归因。没有根据原输出挑选排序方案。

必须区分两种证据：保存的Python源码快照来自Day03本地CPU只读分析，不是原GPU二进制源码；原Day04 CUDA环境记录另存于环境证据。快照描述的小输入坐标偏移路径用于本次主重放，精确核对原完整输出后才用于解释。没有原CUDA逐次事件或编译内核源码，匹配不能证明原设备内部运算和事件轨迹逐位相同。

主路径对offset和坐标相加逐步float32舍入，IoU面积算式同样逐步float32舍入；严格`> float32(0.7)`删除。原坐标float32、原坐标float64、偏移坐标float64均为数值审计，不能替换主判定。1e-6仅用于标记阈值附近。审计完整输入所有候选对，包括跨层与同分关系。

输出包括：

- `case_report_zh.md`：先解释过程，再列身份、分数、GT/相互IoU、原事实与重放归因。
- `all_inputs.csv`、`replay_steps.csv`、`replay_events.csv`、`replay_pair_checks.csv`：全部输入、每步保留/新删除列表、逐候选首次决定、实际比较过程。
- `output_comparison.csv`、`case_verification.csv`：完整输出顺序、身份、框、分数、layer_nms与150截断逐项核对；不匹配仍导出差异。
- `case_relations.csv`、`good_survival_checks.csv`：全部可能上游保留候选关系与好框存活的每步比较，不排除好框之间的关系。
- `all_pair_numeric_audit.csv`、`score_ties.csv`、`threshold_precision_audit.csv`：全输入同分、阈值及精度记录。
- 每例`*_key_candidates.png`和`*_process.png`，以及`figure_geometry.json`：原Letterbox、GT与局部展示范围。
- `verification.json`、`input_provenance.json`、`history_hashes.json`、`implementation_evidence.json`、`run_metadata.json`、`run.log`、`artifact_manifest.json`：核验、SHA、环境、命令、次数、日志与退出码。外层进程日志和退出码保存为同级`*_process.log/json`。

新增主重放按模型×图片计数，完整输出核验也按项计数；两者都预计为2，实际值以verification.json为准。没有checkpoint加载、模型前向、服务器连接、训练、test、安装或升级。只读原图片并复用原数据读取及PairedLetterbox，精确核对完整GT。Week08、Week09、Week10 Day01及所用数据在运行前后做SHA检查。
