# Week09 Day03：RPN候选筛选阶段追踪——核查与捕获方案交付

日期：2026-11-08（北京时间）  
交付性质：本地只读核查结果与待实施方案，可交给ChatGPT接续。  
修订状态（2026-11-08）：已获准实现本地独立模块及人工数据单元核验；尚未连接服务器，尚未执行Day03真实模型前向。测试是否通过以本轮测试日志为准。

实施进展：第13节本地最小模块现已实现，最终21项人工张量单元测试通过、退出码0；429个历史文件SHA校验未变。日志见checks/synthetic_unit_run02/run.log，范围与限制见README.md。这些是单元核验结果，不是Day03的LLVIP实测结果。首轮18项出现1项包装签名错误，修复后19项通过；补充两项核验后最终为21项通过。

本文件最初交付时只有方案；当前授权仅推进第13节的本地实现与人工数据测试。服务器连接和真实图片捕获仍需用户反馈后再推进。

## 1. 唯一研究问题与执行边界

固定Day02的六张validation图片、六个指定GT、原IR/Fusion best权重，回答：

**高IoU候选是否曾经存在？如果存在，在哪一步丢失？**

本轮已完成本地文件与源码核对、候选身份设计、捕获方法及验收条件设计。没有训练、没有新模型前向、没有test评测，也没有修改融合模块或任何旧代码/权重。

后续保持原pre/post-NMS预算、score/NMS阈值和预处理，不通过扩大Top-N制造改善。不要假定旧SSH仍然在线，不保存或携带凭据，不安装或升级依赖。

## 2. 项目位置与协作规则

~~~text
本地项目：D:\python\test\vision-learning
本地Python：D:\python\.venv\Scripts\python.exe
服务器项目：D:\vision-learning-server
服务器Python：D:\cv-server-env\Scripts\python.exe
~~~

已检查根目录AGENTS.md，未发现更具体的下级AGENTS.md。按约定全程中文，每次只推进一个明确步骤，解释后等待用户反馈；修改前先读文件，不覆盖已完成实验，不安排测验。

已用shell检索：

~~~text
C:\Users\GXBBBXB\.codex\memory\research-notes.md
~~~

没有使用研究记忆MCP。记忆仍包含较早的“RPN实际诊断尚待确认”描述，与当前已存在的Day01/Day02产物相比滞后，本次以真实产物为准；本轮未更新记忆。

用户是研一初学者。解释时可以把候选ID理解为“每个框的固定身份证”，把筛选阶段理解为“逐关检查”；不要把相关现象写成已证明的机制。

## 3. 实际读取范围

以下相对路径均以本地项目根目录为基准。

| 文件 | 本轮读取目的 |
| --- | --- |
| week09/day02/DAY02_RPN_VISUALIZATION_GPT_HANDOFF.md | Day02历史结论、运行与限制 |
| week09/day02/visualize_proposals.py | 原临时包装、hooks、最佳框选择及恢复逻辑 |
| week09/day01/rpn_diagnostic.py | transform/RPN捕获、坐标/状态检查、原配置记录 |
| week09/day02/outputs/proposal_visualization_run01/selected_cases.csv | 六例实际stem、0起始GT索引、原image_index |
| 同目录proposal_details.csv | 12项最佳IoU、框、分数及最终列表索引 |
| 同目录geometry_analysis.csv | 12项中心/宽高/边界误差 |
| 同目录rpn_proposals.json | 两模型六图的全部最终后处理候选及对应分数 |
| 同目录run_metadata.json | 历史环境、权重、配置和保存的服务器筛选源码 |
| 同目录independent_verification.json | 历史独立复算范围及误差 |

补查原文件：

- week08/day03/ir_common.py：checkpoint_model()及原IR数据入口。
- week08/day05/fusion_model.py：restore()、PairedDetector和双分支骨干。
- week08/day05/fusion_common.py：原paired loader。
- week08/day01/llvip_dataset.py：配对读取、target及原索引。
- week08/day02/paired_letterbox.py、paired_horizontal_flip.py：同步几何变换及val不翻转。
- 本地TorchVision的models/detection/rpn.py、ops/boxes.py：解码、展平、逐层Top-N和NMS实现。

本轮通过标准库解析CSV/JSON和源码，没有导入项目模块来构造模型。常规shell工具初始化曾因本机权限错误而未启动，随后只读重试成功；没有为此修改系统设置。

## 4. 核查确认的事实与已有数据边界

### 4.1 六个指定GT

以selected_cases.csv为准，GT索引从0开始：

| 案例 | 场景 | validation原image_index |
| --- | --- | ---: |
| 030269/G1 | 03 | 21 |
| 030308/G1 | 03 | 24 |
| 030205/G0 | 03 | 16 |
| 170100/G1 | 17 | 120 |
| 130126/G3 | 13 | 85 |
| 170455/G1 | 17 | 145 |

selected_cases.csv有6行，proposal_details.csv和geometry_analysis.csv各12行。两模型原始JSON的每图字段是：

~~~text
image_index
image_size_hw
gt_boxes
proposal_boxes
objectness_sigmoid
~~~

proposal_details中的best_proposal_index_0based仅是最终列表索引；objectness_rank_1based是最终列表中的1起始排名。它们都不是筛选前的anchor/候选原始编号。

### 4.2 现有文件能回答什么

可以确认最终保留候选的坐标、分数、数量，以及指定GT的最大IoU和几何误差。

不能还原被删除的候选、原始层归属或每一步删除原因。仅凭最终候选偏宽、偏移或objectness高低，不能判断高IoU框是否曾经存在，也不能断言它被NMS或Top-N删掉。

Day02历史独立复核记录：12项最大IoU与Day01差值为0；从保存框以Python float64重算最大IoU，与Torch float32最大误差约1.2630×10⁻⁷。这些是历史核验记录，不是Day03捕获结果。

### 4.3 以实际运行源码为依据

Day02的run_metadata.json保存了当时服务器filter_proposals的完整源码。将其与本地源码正确规范化后，文本一致；整个rpn.py的SHA256也一致：

~~~text
00ff7ea77fe9ba37038ad6b6c1289185b2e753ca124c50cf64d8f0d30edecd43
~~~

因此本方案依据的是已核对的Day02实际实现，不只是模型名称或版本号。但服务器当前状态尚未检查，未来执行前仍需重新记录当前源码及环境；如已变化，先停止复核。

记录中的服务器TorchVision为0.29.1+cu126。本轮未重新查询服务器版本。

## 5. 原推理路径与参数

继续使用原模型恢复方法：

- IR：ir_common.checkpoint_model()。
- Fusion：fusion_model.restore()，实际detector为model.detector。

原best路径：

~~~text
IR：
D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth
SHA256=9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2

Fusion：
D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth
SHA256=9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a
~~~

这些是Day02记录的实际路径与校验值，未来运行仍应重新计算校验。

输入保持原RGB/IR配对、同步640 Letterbox和原内部transform。val不翻转。Fusion仍以visible/infrared字典输入，不改成六通道卷积。GT与所有候选在同一次内部transform后的坐标中计算IoU。

实际特征键为0、1、pool，不能套用P2–P5。每次捕获须读取实际键顺序，不能只用三个层号代替键名。

| 参数 | 原值与含义 |
| --- | --- |
| eval pre-NMS Top-N | 每层最多150个，不是全图共150 |
| eval post-NMS Top-N | 每张图片最终总上限150 |
| score阈值 | sigmoid(objectness) ≥ 0.05 |
| NMS IoU阈值 | 0.7，按层独立抑制 |
| 最小框尺寸 | 0.001，按原库宽高条件保留 |

## 6. 候选身份证设计

在进入filter_proposals时，输入已是box_coder.decode产生的解码框。为每个框建立不可变身份：

~~~text
candidate_id = run_id / model / stem / flat_index

辅助字段：
level_key
level_index
index_within_level
flat_index
~~~

以上是命名结构，不是实验中已经捕获的ID。

flat_index为筛选前展平列表中的原始位置；用实际num_anchors_per_level的累计区间确定所属层和层内索引。层键由同次RPN输入特征字典顺序对应，不按坐标或框尺寸猜测。

源码确认当前展平顺序为逐层拼接，层内按H、W、anchor顺序组织。若额外保存网格行列及anchor类型，必须读取并校验实际H/W及每位置anchor数量；不能凭经验硬编码候选总数或形状。

后续选取、过滤、排序和裁剪都携带原ID，不重新编号。坐标被裁剪后ID保持不变，另存新坐标。两个坐标相同的候选也不能合并成一个ID。

ID用于同一模型、同一运行内部追踪。不能因为IR/Fusion索引相同，就认定两模型是同一个语义候选；跨设备或版本也不承诺完全相同的保留身份。

## 7. 最小捕获设计：原调用不变，记录与计算分开

计划仅用原完整eval前向，对IR/Fusion各处理选中6张图片，固定种子与数据顺序。保留原transform/RPN只读hooks，不使用最终RoI检测框作本次指标。

### 7.1 捕获原调用的输入、索引和输出

1. 临时包装filter_proposals，复制解码候选、objectness logits、图像尺寸和各层数量。
2. 临时包装RPN实例的_get_top_n_idx，调用原方法，保存其实际返回的Top-N索引，原样返回。
3. 在该次filter_proposals内部调用期间，临时记录原batched_nms的实际输入及保留索引，调用原方法并原样返回。
4. 保留同次原filter_proposals最终输出和RPN forward输出，作为核对基准。

真实索引比自行重新排序更重要：同分情况下，新排序可能选出不同的框。图像顺序或候选ID只用于记录，不能用它们重排原筛选操作。

NMS记录会临时涉及共享库模块的函数引用，作用范围必须限制在该次筛选调用内，采用单进程串行捕获、禁止该进程内并发模型调用，并在离开筛选时恢复，不能影响后续RoI。若不能保证这个隔离条件，不实施这种包装，先调整方案。

### 7.2 在副本上重放，绝不替代模型输出

使用复制的同设备张量、捕获到的实际Top-N索引和NMS保留索引，按原源码顺序计算裁剪、小框/分数过滤等中间快照。

重放的NMS输入必须与实际捕获的NMS输入一致；重放终点必须与原方法最终框、顺序和分数一致。只在核验通过后，才解释中间阶段的身份去留。

保留原始logits。实际筛选的sigmoid计算位置沿用源码：先逐层Top-N，再对选中的logits计算sigmoid。解码阶段若为展示补算全部候选的sigmoid，应标为诊断分数，不把它用于替代原筛选计算。

原方法的返回值始终原样交还模型，不能把重放输出传回模型。这样不需要第二次网络前向。

所有包装、hooks和原实例属性存在状态都应在finally中恢复，包括异常路径。异常时保留已捕获原始信息和失败日志，不继续声称阶段归因成功。

## 8. 覆盖的筛选阶段及记录内容

| 阶段 | 操作与记录 |
| --- | --- |
| decoded | 全部解码候选、原始身份、层号、logits；暂时越界不应额外删除 |
| pre_nms_topn | 逐层原Top-N实际索引，每层最多150 |
| clipped | 原图像边界裁剪前后坐标及同ID的IoU |
| small_box_filtered | 按原min_size返回的保留索引 |
| score_filtered | 原sigmoid分数≥0.05的保留索引 |
| layer_nms | 原按层NMS的输入、返回保留索引及去留 |
| post_nms_topn | NMS输出顺序中的每图前150 |
| final_verified | 与同次原方法和RPN实际输出核对 |

对每一阶段、每个指定GT，记录：

- 候选总数量和逐层数量。
- 最大IoU及对应候选ID；并列时保留并列信息，展示ID采用明确规则。
- IoU≥0.75候选数量。
- 相关候选的坐标、logit、分数、层内/阶段排名及同分信息。
- 固定候选ID的首次删除阶段和直接操作原因。

空集合计数为0，最佳ID为空；最大IoU按明确的空候选约定记0，并保留空集合标志。发现非有限坐标/分数时先记录并停止解释，不额外清洗输入来改变原操作。

### 两个不同的视角

**阶段最佳框**：每一关重新找最大IoU，最佳ID可能换人。

**固定候选追踪**：给定一个ID，逐关查看它是否存活、坐标和IoU怎样变化。至少保留全部初始候选的身份与状态，便于追踪各阶段最佳框、高IoU框及最终最佳框。

不能用阶段最佳IoU下降，直接推断前一关的最佳框被哪一个候选抑制。

### 裁剪与“丢失”的区分

裁剪不删除候选，只改变坐标。应单独记录：

- 首次被删除的阶段。
- 同ID裁剪前后的坐标、IoU及容差内变化；不把裁剪记为候选删除。
- 某阶段高IoU候选从有到无的转变，以及之后是否重新出现。

本实验GT必须是有限、正面积、完整位于[0,width]×[0,height]的框。裁剪范围为包含Letterbox padding的整张640×640图像，不是去掉padding后的有效内容区域。在此前提下，对于有限、正面积的解码候选，精确几何下裁剪不改变其与GT的交集面积，只会使候选面积不增，因此同ID的IoU不能降低。原先“裁剪可能降低IoU”的表述不适用于本实验，现予修正。

实现使用float64诊断IoU，明确核验iou_after + atol >= iou_before，默认atol=1e-6、rtol=0；这不是修改0.75覆盖阈值。保存每个实际进入裁剪阶段ID的前后坐标、IoU、差值与裁剪后退化标志，异常超差时停止核验，不自动放宽容差。GT越界、GT退化、非有限输入及初始无效候选应明确拒绝，不静默清洗。有效候选裁剪后可能变为零面积，此时IoU记0，仍保留ID及退化标志，随后由原小框规则删除。由于该候选与范围内GT的交集原本为0，不违背上述结论。

空候选集合数量为0、最佳ID为空、最大IoU按约定记0；裁剪不降断言为空集上的无违反，不制造虚构候选。裁剪可能提高IoU、高IoU候选数可能增加；所有后续纯删框阶段，以及裁剪前的Top-N子集选择，最大IoU和高IoU候选数不应增加。0.75判定保留原比较，阈值附近的容差内数值变化需单独记录，不能当作机制结论。

对未通过pre-NMS Top-N的框，不能把另外计算的“如果裁剪会怎样”当作它实际经历过的阶段。若以后需要这种反事实分析，必须另外标注。

## 9. NMS抑制关系的证据边界

最小实验只确认哪些ID被原NMS保留或删除，不直接给出“框A实际抑制框B”的断言。

当前库返回保留索引，没有返回原内核逐次抑制配对。找到一个同层、分数较高且IoU>0.7的保留框，只能说明它是可能相关的框，不能自动确定原内核的实际抑制者。

若后续独立实现贪心重放以构建抑制配对，需要额外记录使用的排序、同分情况、计算设备和数值误差，并与原NMS保留结果核对。即使保留集合相同，配对解释仍应注明是重放归因，不冒充原内核直接证据；同分、阈值附近及多重重叠尤其要保留不确定性。

首次最小实验不以识别抑制者为完成条件。

## 10. 未来结果表字段示例

**以下仅为字段设计，不是实验结果。当前没有任何Day03阶段数量或IoU数值。**

| 文件 | 主要字段 |
| --- | --- |
| stage_summary.csv | run_id, model, stem, gt_index, stage, level_key, candidate_count, max_iou, best_candidate_id, count_iou_ge_075 |
| candidate_trace.csv | candidate_id, flat_index, level_key, index_within_level, stage, xyxy, logit, score, iou, kept, first_removed_stage |
| 排名字段 | rank_in_level, stage_score_rank, nms_keep_rank, equal_score_group |
| 裁剪字段 | candidate_id, xyxy_before_clip, xyxy_after_clip, iou_before_clip, iou_after_clip |
| gt_trace_summary.csv | ever_high_iou, coverage_loss_transitions, final_max_iou, day02_iou, difference, trace_verified |

可另记录score阈值、small-box条件、Top-N截断和NMS删除等直接操作原因。NMS_suppressor_id默认留空；如果没有足够证据，不能为了填满表格而猜测。

## 11. 拟新增文件清单

建议全部位于week09/day03，新运行目录禁止覆盖：

~~~text
week09/day03/
  DAY03_RPN_STAGE_TRACE_PLAN_DELIVERY.md   # 已修订的方案
  trace_rpn_stages.py                     # 已实现本地核心，未接入真实模型
  test_trace_rpn_stages.py                # 已实现人工张量测试
  verify_trace.py                         # 已实现本地单元测试运行入口
  README.md                              # 已完成说明
  checks/synthetic_unit_run02/            # 已完成的人工测试日志/结果
  outputs/<新的运行目录>/                 # 待实施
    raw_capture.pth
    stage_summary.csv
    candidate_trace.csv
    gt_trace_summary.csv
    verification.json
    day03_report.md
    run_metadata.json
    run.log
    execution_status.json
~~~

raw_capture保存原调用输入/输出、实际索引和必要的阶段快照，不保存凭据。最终表格从原始捕获生成，可离线重复核验。阶段候选数量以实际张量为准，不在方案中预填。

## 12. 未来真实运行验收条件

1. 在新运行开始时重新检查实际环境、源码与配置，记录Torch/TorchVision、设备、源文件SHA、原checkpoint路径与SHA，以及完整实际命令。
2. 捕获/重放终点与同次原filter_proposals输出，在数量、坐标、顺序、对应分数上相符；同时核对RPN forward实际框。
3. 同次同设备重放优先要求精确一致，身份索引及顺序应精确一致。若不一致，保存原输出、差异位置和误差，不自动放宽容差或继续归因。
4. 六例12项最终最大IoU与Day02逐项比较，报告实际差值。起始比较容差为atol=1e-6、rtol=0，并另查0.75覆盖判定是否变化。
5. 临界案例170455/G1的IR历史IoU为0.750012934，必须保留足够精度，不能只用四舍五入后的0.750判断。
6. 原权重文件、模型持久state和RPN/transform配置前后不变，原Week08/Day01/Day02输入与产物校验不变。
7. 正常结束及人为异常测试后，所有临时hooks、方法/库函数引用及实例属性状态可靠恢复。
8. 记录运行范围、时间、退出码、容差、核验结果和未完成部分。结果异常时保留新目录，不覆盖或伪造完成状态。

不承诺跨设备、跨版本或同分情况下绝对逐位一致。发生差异时先区分数值、索引身份和筛选判定变化，再排查；不能把差异直接当作新的研究结论。

## 13. 第一项实施步骤与验收

收到用户反馈后，先只实现Day03独立模块的候选ID、阶段记录和重放核验，用人工小数据测试，不加载真实模型、不读取测试集、不连接服务器。

最低验收：

- 三层候选身份正确，筛选和排序后仍能追溯原ID。
- 每层pre-NMS预算与每图post-NMS预算严格区分。
- 裁剪前后同ID坐标/IoU变化正确，候选删除与阈值跌落分开。
- score边界、小框条件、同分和空候选处理有明确记录。
- 同分不通过额外排序改变原输出。
- 正常及人为异常退出后包装全部恢复，重放输出核验能发现故意制造的差异。

人工数据只是单元核验，不是LLVIP实验结果。通过并解释后，再由用户决定是否推进单张选中val图的GPU捕获，最后扩展到六图。

## 14. 当前结论与接续提醒

已证实的历史事实：Day02最终RPN候选在这些个例中存在不同的几何质量，且历史复核通过。

合理研究假设：某些高IoU候选可能在原Top-N、score或NMS等阶段丢失；也可能解码结果本身就缺乏足够好的框。

尚未证实：高IoU候选究竟曾否存在、在哪一关丢失、是否由融合特征或RPN回归导致，以及NMS实际抑制配对。

本轮没有Day03新实验结果。接手ChatGPT应先解释本方案、等待用户反馈，不直接开始训练、修改融合模块、扩大预算或重复test评测。

