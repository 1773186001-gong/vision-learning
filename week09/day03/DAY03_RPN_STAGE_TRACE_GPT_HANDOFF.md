# Week09 Day03：RPN候选筛选阶段追踪交接

整理日期：2026-11-08（Asia/Shanghai）。状态：固定六例捕获、阶段验收和六条NMS离线关系分析均已完成并经用户审阅。本次收尾只读取已有报告、CSV和验收记录，新增本文件；没有重新核验数值、连接服务器、加载模型或启动实验。

## 1. Day02的问题与Day03的回答

Day01已在完整validation的150张图、416个GT上发现：使用原RPN全部实际保留框时，IR的IoU≥0.75覆盖为302/416，Fusion为285/416。这是最终RPN候选的覆盖差异，不是最终检测AP75，也没有证明差异来自哪一筛选阶段。

Day02选取6个有代表性的GT，观察其最终最佳RPN框的坐标、分数和定位偏差。选例规则是3个IR优势例、2个Fusion优势例、1个临界例；它们不是随机样本。Day02只保存最终后处理候选，不能还原已删除的框，因此提出：

> 高IoU候选是否曾经存在？如果存在，在哪一步丢失？

Day03保持原模型、权重和筛选参数，记录解码后至最终预算截断的真实索引和阶段候选，实际回答了这6个指定GT的上述问题。结果并非只有一种模式：

- 部分Fusion例在全部七阶段都没有IoU≥0.75候选；
- 有的达标候选存在，但在原score阈值处全部丢失；
- 有的达标候选进入NMS后全部丢失，而且IR和Fusion都有这样的个例；
- 本次指定GT的覆盖没有在最终post-NMS预算截断处进一步降低。

随后离线计算得到6条“被删除达标候选—可能相关原保留框”关系，均存在分数与指定GT定位质量的排序冲突。尚未记录原GPU实际抑制配对，不能把这些关系命名为实际抑制者。

## 2. 实际范围与固定条件

本地项目：`D:\python\test\vision-learning`。服务器项目：`D:\vision-learning-server`。本地Python：`D:\python\.venv\Scripts\python.exe`；历史GPU捕获使用`D:\cv-server-env\Scripts\python.exe`。本文件中的服务器信息只用于说明既有记录，不代表当前SSH连接存在或获准重新登录。

六例顺序、GT索引、image_index按Day02 selected_cases.csv：

|案例|场景|image_index|Day03真实图片前向|
|---|---|---:|---|
|030269/G1|03|21|先前单图IR/Fusion各1次；后续五图步骤直接复用|
|030308/G1|03|24|IR/Fusion各1次|
|030205/G0|03|16|IR/Fusion各1次|
|170100/G1|17|120|IR/Fusion各1次|
|130126/G3|13|85|IR/Fusion各1次|
|170455/G1|17|145|IR/Fusion各1次|

GT索引从0开始。Day03真实图片前向总计12次：单图步骤2次，后续五图步骤10次。推理始终使用完整val配对图片，只分析指定GT，没有按GT裁图。Fusion原恢复入口中的CPU零张量backbone结构探测属于既有构造行为，不是额外真实图片前向。

IR复用`week08/day03/ir_common.py::checkpoint_model`；Fusion复用`week08/day05/fusion_model.py::restore`，配对数据入口为`week08/day05/fusion_common.py::loader`。同步Letterbox和原内部transform保持不变，IR与配对加载的红外张量和GT已在捕获时核对一致。

|模型|原服务器best路径|已有记录中的SHA256|
|---|---|---|
|IR|D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth|9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2|
|Fusion|D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth|9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a|

实际特征键顺序为`0、1、pool`，不套用P2–P5；Fusion RoI使用实际存在的`0、1`。逐层pre-NMS上限150，三个层合计Top-N后450；逐图post-NMS总上限150；score阈值0.05、NMS阈值0.7，均未修改。使用model.eval()和torch.inference_mode()，没有训练、优化器更新或test运行。

## 3. 阶段、坐标与指标口径

七阶段依次为：

1. decoded：原RPN解码后的全部候选；
2. pre_nms_topn：逐层原Top-N索引保留；
3. clipped：裁剪到完整图像边界；
4. small_box_filtered：原最小尺寸过滤；
5. score_filtered：原sigmoid objectness≥0.05过滤；
6. layer_nms：按层原NMS的全部输出；
7. post_nms_topn：原NMS输出按顺序截到每图最多150。

本次每项decoded为13500框，层0、1、pool分别为6000、6000、1500；Top-N后为450框。候选ID来自原展平索引，并随实际Top-N/NMS索引传播，不靠坐标匹配或重新编号。同flat索引在不同模型中不代表相同预测框；引用ID必须带图片和模型上下文。

GT与候选使用同次transform后的640×640坐标，包含Letterbox padding。GT完整位于裁剪范围内。对有限有效框，精确几何下同ID裁剪不降低IoU；已记录真实裁剪前后坐标与IoU，并用float64容差核查。未通过Top-N的候选没有被反事实裁剪后冒充实际经历。裁剪后退化框由原小框规则删除；空候选最大IoU按约定为0。

本交接表的IoU主口径为与Day02一致的CPU float32 torchvision.box_iou。float64诊断IoU另列在原CSV/JSON中，不能混用。达标定义始终是IoU≥0.75，不在阈值上附加容差。阶段最佳框是每阶段重新选择的最大IoU框；固定候选轨迹则始终追同一原ID，两者要分开读。

## 4. 六例12项：解码质量、筛选损失与最终IoU

下表摘录既有汇总。筛选损失定义为“decoded最大IoU减最终最大IoU”，不是损失函数。对本次12项，Top-N、裁剪、小框过滤均未改变指定GT最大IoU，所以非零差值可结合真实阶段表定位到score或NMS。一般情况下裁剪可能提高IoU，不能把任意解码—最终差值都直接当作纯删框损失。

|案例／模型|decoded最大IoU|最终最大IoU|筛选损失|观测到的最大IoU下降阶段|
|---|---:|---:|---:|---|
|030269/G1 IR|0.793667018414|0.793667018414|0|无|
|030269/G1 Fusion|0.750537335873|0.605291903019|0.145245432854|NMS|
|030308/G1 IR|0.878793776035|0.878793776035|0|无|
|030308/G1 Fusion|0.712219297886|0.712219297886|0|无；七阶段均未达标|
|030205/G0 IR|0.851542413235|0.851542413235|0|无|
|030205/G0 Fusion|0.729380309582|0.696796298027|0.032584011555|NMS；始终未达标|
|170100/G1 IR|0.769244015217|0.578785657883|0.190458357334|score过滤|
|170100/G1 Fusion|0.845996797085|0.845996797085|0|无|
|130126/G3 IR|0.875696659088|0.664702534676|0.210994124413|NMS|
|130126/G3 Fusion|0.846598207951|0.846598207951|0|无|
|170455/G1 IR|0.750012934208|0.750012934208|0|无|
|170455/G1 Fusion|0.674085140228|0.674085140228|0|无；七阶段均未达标|

这些值说明解码差距和筛选损失可以并存。例如030205/Fusion解码最佳已低于IR，随后NMS又进一步降低最佳IoU；130126/IR的解码最佳反而高于Fusion，却因筛选丢失达标框而最终更差。不能从最终好坏直接推断解码阶段谁更好。

12项最终IoU与Day02实际差值全部为0，比较起始atol=1e-6、rtol=0未放宽。170455/G1 IR完整值为`0.7500129342079163`，0.75覆盖保持True；对应float64诊断值为`0.7500129389053078`。不能把四舍五入后的0.750代替完整精度判断。

## 5. 达标候选的存在与消失

下表数量按七阶段顺序排列，统计的是每个指定GT的IoU≥0.75候选数，不是全图候选总数：

|案例|IR|Fusion|
|---|---|---|
|030269/G1|2→2→2→2→2→2→2|1→1→1→1→1→0→0|
|030308/G1|5→5→5→5→4→2→2|0→0→0→0→0→0→0|
|030205/G0|8→8→8→8→8→2→2|0→0→0→0→0→0→0|
|170100/G1|1→1→1→1→0→0→0|3→3→3→3→3→3→3|
|130126/G3|5→5→5→5→5→0→0|2→2→2→2→2→1→1|
|170455/G1|1→1→1→1→1→1→1|0→0→0→0→0→0→0|

### 始终没有达标框

030308/G1、030205/G0、170455/G1的Fusion在七阶段均无达标框。该判断检查了真实裁剪后阶段，而不是仅因decoded不达标就跳过后续过程。不能把它直接解释为某个融合特征或回归分支出了问题。

### score过滤丢失

170100/G1 IR的唯一达标候选ID13256属于pool层，IoU为0.7692440152168274，objectness为0.0320378914475441，低于原0.05阈值。它通过逐层Top-N和小框过滤，随后被score过滤删除，指定GT最大IoU下降到0.5787856578826904。它在小框过滤后全图分数排名165、层内21，说明“每层前150”与“全图前150”不能混为一谈。

### NMS丢失

- 030269/G1 Fusion：ID2201是唯一达标候选，进入NMS后消失，最大IoU由0.7505373358726501降至0.6052919030189514。
- 130126/G3 IR：5个达标ID3216、3516、3521、9216、9516均进入NMS，均不在原NMS输出中，最大IoU由0.8756966590881348降至0.6647025346755981。
- 030205/G0 Fusion虽从未达标，NMS仍删除了阶段最佳ID9506，使最佳切换为9492，IoU由0.7293803095817566降至0.6967962980270386。

另有达标候选数量减少但指定GT覆盖保持的例子，如030308/IR、030205/IR和130126/Fusion。因此“删除一个好框”与“这个GT全部好框消失”是不同统计事件。

### 最终预算没有进一步降低这些指定GT的覆盖

130126/IR原NMS输出171个框，最终截为150；Fusion为201截为150。指定G3的最大IoU在这一步未变，达标数量分别保持0和1。其他例原NMS输出已小于150，最终截断不再删框。本次没有证据将这些指定GT的覆盖失败归于post-NMS总预算。

这不等于证明全部val GT都不受预算影响，也不等于130126整张图的其他GT覆盖已被逐一分析。

## 6. 六条离线可能关系与分数／定位排序冲突

仅对030269/G1 Fusion的ID2201、130126/G3 IR的5个达标删除ID进行关系分析。原保留框集合取原NMS全部输出：分别为105、171个，未将后者提前裁到150。

可能关系条件：同一特征层，原objectness更高或完全同分，相互IoU严格大于原NMS阈值0.7。真实保留索引决定保留集合；没有重新运行CPU NMS来替代原GPU决策。CPU float32、float64及按层偏移后的float32敏感性诊断分别记录；本次没有同分或同层阈值判定分歧，也无1e-6范围内的边界比较。诊断窗口不是阈值容差。

|案例／模型|删除ID／层|删除分数|删除框对GT IoU|可能相关保留ID／层|保留分数|保留框对GT IoU|相互IoU|原NMS输出位置（0起始）|
|---|---|---:|---:|---|---:|---:|---:|---:|
|030269/G1 Fusion|2201／0|0.900450826|0.750537336|2501／0|0.996665895|0.605291903|0.753271937|2|
|130126/G3 IR|3216／0|0.988013268|0.818582356|3221／0|0.999775231|0.664702535|0.743190229|0|
|130126/G3 IR|3516／0|0.964920402|0.875696659|3221／0|0.999775231|0.664702535|0.749724388|0|
|130126/G3 IR|3521／0|0.999112666|0.868215621|3221／0|0.999775231|0.664702535|0.720879495|0|
|130126/G3 IR|9216／1|0.982528687|0.791036189|9221／1|0.999507785|0.595071256|0.744562685|1|
|130126/G3 IR|9516／1|0.959691882|0.760225832|9221／1|0.999507785|0.595071256|0.788245261|1|

表内每个ID均限定于所在行的图片和模型。完整带前缀ID、双方坐标、分数、原输入／输出位置、阈值余量和最终预算保留标志在关系CSV中。上述三个相关保留框均通过最终预算截断。

六条关系都表现为“保留框分数更高，但对指定GT定位更差”。几何上，030269/Fusion ID2501偏大：宽、高约为GT的1.310、1.261倍；130126/IR ID3221、9221中心分别向上约11.283、10.095像素，且偏窄。这些是坐标描述，不是特征机制解释。

已有全图GT被用于其他GT重叠检查：030269/Fusion ID2501与其他GT最大IoU为0；130126/IR ID3221、9221与其他GT最大IoU分别为0.072479866445、0.118666343391，最大值对应G4。该索引只是几何最大重叠对象，不能当作已证实语义归属，也不能据此认定邻近目标混淆。

每个删除ID恰好找到一条符合条件的关系，但这不构成原GPU实际抑制配对的直接证据。记录中的unexplained_by_requested_retained_relation=False只表示找到了本轮条件下的可能关系，不表示实际删除机制已经完全解释。

## 7. 三种证据必须分开

|证据层次|当前可以说什么|当前不能推出什么|
|---|---|---|
|原操作事实|真实索引证明哪个ID在哪一步删除，哪些框属于原NMS全部输出及最终输出；真实阶段张量给出候选数和指定GT覆盖|不能仅凭保留索引认定具体哪个框实际抑制了哪个框|
|离线几何关系|坐标、层和分数证明六条可能关系满足指定条件，且存在更高分但指定GT定位更差的排序冲突|不能把几何相容关系或唯一可能关系命名为原GPU实际抑制事件；最大GT重叠不等于语义归属|
|机制假设|高分但较差定位的重叠框可能参与达标候选流失；上游候选生成差异也值得研究|尚未证明融合特征、anchor／框回归或某个模块是根因，不能解释全部AP75差异或推广总体比例|

原GPU逐次抑制事件没有保存，本轮也没有实现独立贪心NMS归因。离线按层坐标偏移只是同版本本地源码下的数值敏感性诊断，不是原CUDA内核内部输入和执行轨迹的直接记录。即使没有同分或边界争议，也不能省略这个证据区别。

## 8. 已有验收证据与运行记录

以下均摘自历史文件；本次收尾没有复跑这些检查。

|步骤|已有结果与证据|
|---|---|
|核心、锁修复与适配器人工测试|完整37项通过、exit_code=0，含27项核心、4项单图、6项五图适配；是人工单元核验，不是LLVIP指标|
|030269单图捕获|IR/Fusion各一次原eval前向，状态completed、exit_code=0|
|新增五图并合并|10次新增前向、复用030269新增前向0次；process_status.json为0|
|同次推理一致性|重放NMS输入、最终框／顺序／分数与原方法精确一致；最终框与实际RPN forward输出一致；原方法正常调用并原样返回|
|稳定性与恢复|原checkpoint、模型持久state、RPN/transform配置及493项服务器受保护文件前后不变；正常路径hooks／包装恢复；异常恢复由人工测试覆盖，未在真实GPU前向中人为制造异常|
|六例离线验收|84行阶段统计一致，12项最终IoU与Day02差值均为0；本地独立核验exit_code=0；104项传输产物SHA匹配，19项复用文件与原单图一致，429项本地历史文件未变|
|NMS离线关系分析|960对比较、6条可能关系、276项原保留框记录、39行全图GT重叠；分析及独立复算exit_code=0；独立binary64公式与Torch float64最大误差0，验收atol=1e-12|
|离线阶段文件保护|既有174项Day03文件未变；独立复算前后12项当次分析输出未变；原GPU实际抑制配对已证实数仍为0|

历史GPU环境：Windows10，Python3.14.6，Torch2.14.1+cu126，TorchVision0.29.1+cu126，RTX3060 Laptop GPU，cuDNN91002；seed=42、threads=2、cudnn_deterministic=True、benchmark=False。离线关系环境：Windows11，本地Python3.14.7，Torch2.14.1+cpu、TorchVision0.29.1+cpu，仅CPU几何运算。

历史关键命令（供追溯，不是本轮执行指令）：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day03\capture_remaining_val.py --device cuda:0 --reuse D:\vision-learning-server\week09\day03\outputs\single_030269_run01 --output D:\vision-learning-server\week09\day03\outputs\remaining_five_merged_run01

& D:\python\.venv\Scripts\python.exe -B week09/day03/verify_merged_capture.py --capture-dir week09/day03/outputs/remaining_five_merged_run01 --output week09/day03/checks/independent_six_case_run01

& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day03/nms_relationship_offline_run01/analyze_relationships.py --output week09/day03/nms_relationship_offline_run01/results

& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day03/nms_relationship_offline_run01/verify_exports.py --results week09/day03/nms_relationship_offline_run01/results --output week09/day03/nms_relationship_offline_run01/independent_verification
~~~

源码／输入SHA、原权重SHA及完整命令保存在各run_metadata.json。六例压缩包服务器／本地SHA256为`7b8728e791e0a46b085c6fad75f517556a2e3ed7ec2918e8f68b8ec79788a220`。旧日志与失败／首版检查记录保留，未用成功文件覆盖历史。

## 9. 接手应阅读的主要文件

以下为本地真实路径，服务器同名产物不代替已下载并验收的本地副本：

- Day01全val背景：`D:\python\test\vision-learning\week09\day01\outputs\val_rpn_run01\rpn_diagnostic_report.md`。
- Day02问题和选例：`D:\python\test\vision-learning\week09\day02\outputs\proposal_visualization_run01\day02_report.md`、同目录`selected_cases.csv`。
- Day03原方案：`D:\python\test\vision-learning\week09\day03\DAY03_RPN_STAGE_TRACE_PLAN_DELIVERY.md`；六例交付：同目录`DAY03_SIX_CASE_CAPTURE_DELIVERY.md`。旧方案描述的是当时计划，当前完成状态以本交接及真实产物为准。
- 已验收实现：`D:\python\test\vision-learning\week09\day03\trace_rpn_stages.py`、`capture_single_val.py`、`capture_remaining_val.py`、`verify_merged_capture.py`。
- 原单图保留目录：`D:\python\test\vision-learning\week09\day03\outputs\single_030269_run01`。
- 六例合并目录：`D:\python\test\vision-learning\week09\day03\outputs\remaining_five_merged_run01`。主要为`merged_stage_summary.csv`、`merged_case_summary.csv`、`merged_candidate_trajectory.csv`、`merged_verification.json`、`six_case_report.md`、`run.log`、`run_metadata.json`、`process_status.json`。
- 原始捕获：上述合并目录下`reused_030269_G1`、`new_030308_G1`、`new_030205_G0`、`new_170100_G1`、`new_130126_G3`、`new_170455_G1`中的`IR_raw_capture.pth`和`Fusion_raw_capture.pth`。仅为自生成张量／字典，不是模型checkpoint。
- 六例独立验收：`D:\python\test\vision-learning\week09\day03\checks\independent_six_case_run01\verification.json`；人工测试：同checks下`remaining_adapter_unit_run02\verification.json`；传输：`remaining_transfer_run01\verification.json`。
- NMS离线代码及说明：`D:\python\test\vision-learning\week09\day03\nms_relationship_offline_run01\analyze_relationships.py`、同目录`verify_exports.py`和`README.md`。
- NMS真实报告与关系：`D:\python\test\vision-learning\week09\day03\nms_relationship_offline_run01\results\analysis_report.md`、`possible_related_retained.csv`、`pair_audit.csv`、`candidate_summary.csv`、`retained_inventory.csv`、`gt_overlaps.csv`、`numeric_verification.json`、`run.log`、`run_metadata.json`、`execution_status.json`。
- NMS导出表独立验收：`D:\python\test\vision-learning\week09\day03\nms_relationship_offline_run01\independent_verification\verification.json`。

## 10. 给研一初学者的简明解释与未完成部分

可以把RPN看作先提出很多框，再逐步筛选的过程。定位准确的框可能一开始就没有，也可能已经有了，却因为分数低或NMS去重没有留下。objectness表示模型对候选的打分，不是定位IoU的估计，所以更高分并不保证框更准。

Day03用原始索引追踪框的身份，回答了选中GT的“好框曾否存在、在哪一关消失”。在NMS关系分析里，我们知道两个框位置很近、保留框分数更高，但没有原GPU逐次删除记录，所以仍不能把可能关系写成已证实的直接抑制。

尚未完成：完整val的逐阶段／逐GT分布；原GPU实际抑制事件归因；anchor回归前后的机制分析；融合特征或回归模块的因果验证；RoI阶段及最终AP75差异的机制解释。上述均不是本轮遗漏的必做实验，也没有在本轮启动。

现有六例经过优势差值与临界规则主动选择，不能用它们估计总体比例。全图GT虽保存在原始捕获中，Day03当前阶段统计仍只针对6个指定GT，不应写成已完成6张图所有GT的阶段诊断。

## 11. 下一阶段唯一研究问题与最小方案（未实施）

> 这些现象在完整validation中是否仍存在、分布如何？

目标是描述分布，而不是马上改融合模块或调NMS。以下只列方案，未来实施仍需用户明确授权。

### 最小验证范围

1. 固定原validation、原best、原预处理和全部RPN参数，核对manifest、输入、源码和环境。按确定性顺序覆盖全部150图、预计每模型416个有效GT；实际图数／有效GT数必须以运行时核对为准。
2. 每张完整图对每个模型最多一次原eval前向，保存真实阶段候选、索引、分数及全图GT；随后离线对该图所有GT计算，不能为了每个GT重复前向。若现有6图捕获的源码／输入／配置与未来条件一致且所需字段完整，可复用这些完整图片的候选和全图GT，避免重复前向；其余图才作新增捕获。当前只提出这一复用条件，没有实施或重新验收。
3. 仍用真实Top-N/NMS索引，原调用正常执行并原样返回；沿用原输出精确重放、最终RPN输出一致、state／配置／权重保护及恢复标准。发现不一致保留记录并停止归因，不自动放宽容差。无需为这项分布问题新增NMS抑制者配对或贪心归因。

### GT级统计定义

唯一键为`模型＋stem＋gt_index`，另保留真实场景ID。不同模型分别统计；IR与Fusion通过`stem＋gt_index`配对。同一个GT可以被多个候选覆盖，但在覆盖分子中只计一次。

对GT g和阶段s记录最大IoU M(g,s)、达标候选数量 C(g,s)、是否覆盖 H(g,s)=1[M(g,s)≥0.75]，以及阶段最佳ID和相关固定ID轨迹。CPU float32为历史一致主口径；float64另列诊断，不用数值容差改变0.75覆盖阈值。空候选M=0、C=0。

按模型分别输出整体及各场景：

- **逐阶段覆盖率**：该阶段H=1的GT数／该组全部有效GT数；整体按GT汇总，不平均逐图或逐场景比例。每组同时给分子与分母。
- **始终无达标候选率**：七阶段H全为0的GT数／全部有效GT数。不能只以decoded未达标代替，因为真实裁剪可能使候选达标。
- **各阶段覆盖丢失事件**：前一阶段H=1、当前H=0。既报告该GT数／全部有效GT数，也报告／前一阶段已覆盖GT数的条件率，并明确两个分母。条件分母为0时记为空，不制造除零结果。
- **出现、首次全部消失与恢复**：保存完整H序列、首次达标、首次1→0及后续0→1。不同阶段事件率不能直接相加当作互斥根因比例；若需要互斥汇总，须另行明确分组规则并保留恢复信息。
- **部分候选流失与几何变化**：区分“C减少但H仍为1”与“覆盖全部丢失”；记录相邻阶段最大IoU变化，裁剪变化与纯删框损失分开。最终预算比较必须从全部原NMS输出到最终截断输出，不能提前只看150框。
- **分数／定位冲突的GT级描述**：对发生NMS达标覆盖丢失的GT，使用原NMS全部输出检查是否存在本轮定义的高分、同层、重叠且对该GT定位更差的可能保留框；每个GT只计一个“存在／不存在”标志。主条件分母明确为发生该NMS覆盖丢失的GT数，也可另报其占全部有效GT的比例；不以关系条数或候选数充当GT比例，不声称实际抑制配对。
- **配对与场景差异**：对同一GT比较IR/Fusion在各阶段是否覆盖，报告两者都覆盖、仅IR、仅Fusion、两者均无；按真实场景分组给GT数和率。必要时描述场景对总体净差的贡献，不能拿六例或某一个场景替代整体结论。

空GT图片单独记录数量和候选情况，不进入GT分母；无效GT若出现需明确原因与实际有效分母，不能静默丢弃。每模型分母独立，不能把IR和Fusion的GT合并后当作一个模型的覆盖率。当前Day01全val终点记录可作为未来末阶段口径核对参照，但不代表中间阶段分布已知。

未来最小交付仅需逐GT阶段表、模型／场景汇总、配对表、验收和中文分布报告，继续区分操作事实、可能关系和机制假设。此方案未实施，本次交接写完读回后即停止，等待用户反馈。
