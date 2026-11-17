# Week10 Day01：两个固定GT的候选关系与原NMS去留

本步骤只读已有捕获，分析030078/G0、030129/G0的IR与Fusion，共4项。GT索引从0开始。没有新增研究指标，没有连接服务器、加载checkpoint、执行模型前向、重跑NMS、训练、调参或运行test。

## 入口与输出

在项目根目录运行：

```powershell
& .\week10\day01\run_analysis.ps1
```

默认输出为`outputs/candidate_review_run01`。输出目录已存在时拒绝覆盖；再次执行须显式指定新的目录，不能覆盖已完成的核查。

```powershell
& .\week10\day01\run_analysis.ps1 -Output week10/day01/outputs/candidate_review_run02
```

`analyze_candidates.py`是离线分析入口，`run_analysis.ps1`保存实际Python命令、进程日志及退出码。输出含：

- `case_report_zh.md`：中文案例报告，回答4个固定问题，附全部选中候选和直接关系。
- `all_score_filtered_candidates.csv`：4项全部原NMS输入，包含其他候选，不据结局删选。
- `selected_candidates.csv`：全部好框及其全部直接竞争者，图中编号映射完整ID。
- `good_candidate_checks.csv`、`case_summary.csv`：逐好框C、全部最小值对应ID及4项L/T，与Day07精确比较。
- `competition_relations.csv`：29条原定义关系，双方身份、坐标、分数、GT/相互IoU及原去留。
- `original_nms_outputs.csv`：原NMS完整输出顺序与最终150预算标记。
- `equal_score_relations.csv`、`precision_audit.csv`：同分及阈值/精度敏感项单列，空表保留表头。
- `figures/`：两例IR/Fusion对照、分层关系图、逐候选图册与640×640几何核对图。
- `input_provenance.json`、`history_hashes.json`、`verification.json`、`run_metadata.json`、`run.log`、`artifact_manifest.json`：来源SHA、历史文件前后哈希、核验、环境、实际命令与退出码。

## 固定判定与证据边界

好框对指定GT的CPU float32 IoU≥0.75。直接竞争者必须来自完整`score_filtered`输入，同实际层、对指定GT的IoU<0.75、原分数严格更高，且相互IoU严格>0.7。C和T沿用Day07，T为全部好框C的最小值；并列最小值全部保留。最小值选择不使用NMS结局。

只读`raw_capture.pth`中的保存张量（`weights_only=True`、CPU），它们是候选捕获，不是模型checkpoint。去留直接取`records[0].nms_indices`；逐项核对保存的`layer_nms`与`post_nms_topn`，不调用NMS算法或独立贪心重放。所有位置从0开始；空输出位置表示原NMS未保留，最终预算另有独立列，不能称为预算删框。

图中GT为黄色，好框为青色，直接竞争者为粉红色；实线为原NMS保留，虚线为原NMS删除。关系边只称“同层高分重叠关系”。保留索引证明最终集合和顺序，不包含GPU逐次抑制事件；本轮不追查被删竞争者的上游关系，不依外观判断误差原因。

图像复用原`week08/day02/paired_letterbox.py`，只导入该几何模块和原数据读取模块，不导入模型恢复入口。核对原始图片/标注/manifest/读取与Letterbox源码SHA；原图1280×1024按0.5缩放为640×512，上下各64像素padding。XML按原顺序解析，转换后的完整GT张量须精确等于原捕获。两模型使用相同IR底图和共同放大范围，Fusion标签表示模型来源，不表示把IR当成其唯一输入。

程序仅对4项进行数值核验；历史Week09文件做SHA前后检查，不重跑完整validation统计。禁止写入Week09，禁止自动覆盖输出，运行期间拦截模型调用与NMS入口。未安装或升级依赖，也未运行单元测试或test推理。完成本步骤后停止。
