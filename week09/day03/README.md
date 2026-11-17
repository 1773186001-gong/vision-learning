# Week09 Day03：本地RPN阶段追踪模块与人工数据核验

状态：本地模块及锁修复的27项人工测试已通过；随后仅030269/G1的IR/Fusion单图真实捕获已完成并验收。人工测试与真实单图结果分别保存，不将前者写成LLVIP实验结果。其余五张图尚未捕获。

## 文件与职责

- trace_rpn_stages.py：稳定ID、阶段快照、原索引副本重放、原输出核对、临时包装和hooks清理。
- test_trace_rpn_stages.py：人工数据测试。NumericRPN是普通Python对象，只复用已安装TorchVision原filter_proposals和_get_top_n_idx数值方法，不实例化真实RegionProposalNetwork或检测模型。
- verify_trace.py：运行人工测试，保存真实run.log、verification.json及execution_status.json，核对历史文件。
- checks/local_unit_protected_before.json：本轮测试前对Week08、Day01、Day02的429项文件SHA256快照（排除__pycache__）；仅哈希读取，未加载checkpoint。
- checks/synthetic_unit_run01/、synthetic_unit_run02/：原19项/21项测试日志和结果，保持原样。
- checks/lock_release_regression_run01/：锁释放修复后的完整27项测试实际日志和结果。
- checks/lock_fix_preservation_before.json：本轮修改前对run01/run02的6个原始测试产物建立的SHA256快照。
- DAY03_RPN_STAGE_TRACE_PLAN_DELIVERY.md：已修正裁剪规则的方案。

## 调用与数据结构

核心入口为TraceCapture(rpn, layer_keys, gt, prefix=...)上下文。rpn必须提供原数值API和配置；当前最小适配器仅支持batch_size=1、单个指定GT。调用方需给出本次实际层键顺序、完整位于图像内的GT和唯一的运行/模型/stem前缀。

进入上下文时临时包装实例filter_proposals；只在原filter调用期间包装_get_top_n_idx和库batched_nms，调用原方法并原样返回。捕获实际Top-N/NMS索引和原始输入输出。库函数引用属于进程共享状态，必须串行、禁止同进程并发推理；捕获锁会拒绝同时进入的捕获，跨线程误调用会报错。退出和异常均恢复原实例属性状态、库函数与注册hooks。

上下文退出后：
- capture.records：包含原解码框、logits、层数量、实际Top-N/NMS索引、实际NMS输入、原最终输出及配置的CaptureRecord列表。
- capture.traces：对应的Trace列表，只有重放全部核验通过才产生。
- trace.catalogue：候选ID、原展平索引、所属层和层内索引。
- trace.stages：decoded、pre_nms_topn、clipped、small_box_filtered、score_filtered、layer_nms、post_nms_topn。
- 每阶段保存原ID、框、分数、float64诊断IoU，以及数量、逐层数量、最佳ID/并列最佳、高IoU数量、最佳框分数/同分排名范围。
- trace.history(flat_index)：同一个候选的逐关记录，含存活、首次删除阶段、坐标、IoU、分数、当前序位和排名范围。
- trace.clip_deltas：仅实际通过Top-N后进入裁剪的ID，含裁剪前后坐标/IoU、差值和退化标志。
- trace.first_removed：固定ID第一次被原操作删除的位置。
- trace.verified：与原NMS输入、原最终框/顺序/分数精确一致后才返回True。

核心模块本身不负责模型/数据加载。新适配器capture_single_val.py已实现固定单图加载、CUDA前向及CSV/原始捕获导出，独立核验入口为verify_single_capture.py。仍未实现六图捕获、NMS抑制者配对、anchor回归分析或本次阶段可视化。人工测试的hooks清理同时覆盖普通注册对象和PyTorch Identity的真实句柄，Identity仅输入人工一维张量，无学习参数或真实图片。

## 裁剪规则

检查GT有限、正面积、完整位于[0,width]×[0,height]。真实实验范围为含Letterbox padding的整张640×640图像，不是仅裁有效内容。对有限、正面积候选，精确几何下裁剪不改变与该GT的交集面积，候选面积不增，故IoU不降。

诊断IoU使用float64，断言after + atol >= before，默认atol=1e-6、rtol=0；不把容差加到0.75覆盖阈值。记录同ID裁剪前后值。GT越界、GT/初始候选退化、非有限输入会明确拒绝；不静默清洗。有效候选完全落在范围外时，裁剪后可退化为零面积，IoU=0并保留ID，然后由原小框规则删除。空集合max_iou按约定为0、best ID为空、高IoU数量为0，保留empty标志。

裁剪可以提高IoU；Top-N及裁剪后的各纯删框阶段，最大IoU和高IoU数量不能增加。未过Top-N的框不出现在clip_deltas，不能把反事实裁剪写成实际阶段。

## 保持原顺序与核验范围

不自行排序替代Top-K/NMS。使用原调用返回的Top-N和NMS索引传播ID；同坐标框保持不同身份，同分按实际索引顺序记录。decoded阶段全体sigmoid仅为诊断展示，真正重放分数仍在Top-N之后按原张量形状计算。

重放使用同版本TorchVision clip_boxes_to_image、remove_small_boxes及原score比较；直接应用捕获到的NMS索引，不重新猜测抑制者。重放的NMS输入、最终框和对应分数要求dtype/device/shape/数值精确一致，发现错误即抛VerificationError，不把重放框替换给原推理。

同值、同坐标框无法单靠最终数值区分身份；身份依据是原操作捕获的索引，不靠坐标匹配。不能从当前单元测试推断不同设备或版本必定逐位一致。

## 实际执行与复现

本地解释器为D:\python\.venv\Scripts\python.exe。最终实际运行：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B week09/day03/verify_trace.py --output week09/day03/checks/lock_release_regression_run01
~~~

最新真实输出：Ran 27 tests，OK，exit_code=0；429个历史文件changed=0。环境Python3.14.7、Torch2.14.1+cpu、TorchVision0.29.1+cpu。源码SHA与数值库rpn.py校验见新目录中的verification.json。

输出目录必须不存在，重跑应使用新名字，例如lock_release_regression_run02；不要覆盖已有日志。直接运行单元测试也可：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -m unittest discover -s week09/day03 -p test_trace_rpn_stages.py -v
~~~

首轮18项中1项报错：已有通用参数实例包装使参数名绑定失败（KeyError: proposals），已修正为按已安装原API签名绑定，同时继续调用原实例方法。修复后19项通过，记录在synthetic_unit_run01。再补充完整640边界与真实hook句柄检查后，最终21项通过。首轮错误不计为通过。

测试覆盖三层身份/预算、同坐标不同ID、裁剪/退化/GT前提、原min_size和score边界、同分、空候选、阶段最佳切换、裁剪不降及纯删框单调性；故意破坏框、顺序、分数/NMS输入/索引均能发现；正常、上下文异常、原NMS异常、重放异常、部分hook注册失败、已有实例覆盖及并发捕获拒绝都检查恢复。

## 异常锁释放修复

之前锁成功获取后，配置对象构造、pre/post预算读取、配置属性和原方法引用读取仍处于释放锁的try/finally之外；这些位置抛错可能留下已持有的锁。旧21项测试没有覆盖该缺口，不能将旧通过记录理解为这条路径已被验证。

本轮仅将释放锁的try保护范围前移到acquire成功之后，配置读取、函数引用读取和后续原有操作全部在保护内。acquire失败时直接报错，不进入释放分支，因此不会释放其他捕获持有的锁。原方法调用、返回对象、索引传播、数值筛选和重放代码保持不变。

新增6项测试方法（含参数化子情况）：

- pre_nms_top_n()、post_nms_top_n()调用失败；min_size、score_thresh、nms_thresh属性读取失败。
- Config构造失败。
- _get_top_n_idx引用读取失败；batched_nms引用读取失败。
- 上下文进入时原filter_proposals引用读取失败（此时尚未获取锁）。

每种失败均核对锁未占用、实例原属性状态、库函数及hooks恢复；移除人工故障后，在同一对象上再完成正常捕获。既有并发拒绝测试增加“失败尝试之后，原持有者的锁仍然锁定”的断言。

本轮完整测试27项通过、退出码0；原run01/run02未覆盖。历史日志SHA的保留核验另记在新检查目录preservation_verification.json。仍未验证真实检测模型集成、CUDA/跨线程实际推理或LLVIP捕获；这次修复与测试不推进这些范围。

## 单图接入的原计划（现已仅完成030269/G1）

建议先选030269/G1，保持原best和原val。未来在单张图片适配器中由原transform/RPN前置hook读取同次GT、图像尺寸、实际特征键顺序，再把这个数值模块接入。分别运行IR/Fusion各一次，原pre/post150、score0.05、NMS0.7不变。

验收需同时包含本模块精确重放、RPN实际输出一致、原权重/模型state/配置前后一致，以及与Day02的IR=0.7936670184135437、Fusion=0.6052919030189514比较（记录实际差值，起始atol=1e-6、rtol=0）。本次已按新增单图授权执行并通过，实际差值均为0；后续仍需用户反馈才考虑其他图片。

## 030269/G1 单图适配器（本步骤新增）

capture_single_val.py固定stem=030269、gt_index=1，不提供切换到其他案例的参数。先核对Day02环境/源码/权重及已审阅锁修复核心的校验，再用完整原val图片推理；只读一张配对图片，IR/Fusion各一次原eval前向。

适配器从同次原transform输出及RPN前置hook取得GT、尺寸、实际特征键；RPN后置hook核对实际框。使用原模型恢复入口和原预算，在finally退出所有hooks/方法包装，核对权重、模型state、配置和受保护文件。

stage_summary中的float32_cpu列使用与Day02相同的CPU float32 torchvision.box_iou；float64_diagnostic列来自核心诊断IoU。不得混用。导出阶段最佳ID及所有阶段最佳/高IoU候选的固定身份轨迹，并保存全部原解码输入、实际索引、原输出和阶段张量。核验失败保留raw_capture和failure.json，不自动重跑或放宽容差。

执行入口（仅在已获准且预检通过的服务器上）：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day03\capture_single_val.py --device cuda:0 --output D:\vision-learning-server\week09\day03\outputs\single_030269_run01
~~~

输出目录须不存在。是否已经真实执行及验收通过，只以该新目录中的日志、execution_status.json和run_metadata.json为准。test_capture_single_val.py为人工张量适配测试，不执行真实模型。

## 030269/G1 本次真实结果与验收

新目录outputs/single_030269_run01：IR/Fusion各1次完整val图片原eval前向，进程退出码0。运行环境与Day02一致，原始19项文件下载后SHA逐项通过；451项服务器受保护文件前后未变。新增适配器4项人工测试通过，核心模块与已审阅锁修复版SHA一致，未修改核心代码。

历史一致口径为CPU float32 torchvision.box_iou：IR=0.7936670184135437，Fusion=0.6052919030189514，两者与Day02实际差值均为0。核心float64诊断的最终值分别为0.7936670774730248、0.6052918661582033，导出列明确区分。

IR在所有7阶段均有2个IoU≥0.75候选，最终57个框。Fusion在解码时有1个高IoU框（索引2201，层0，IoU float32=0.7505373358726501，score=0.9004508256912231），通过Top-N、裁剪、小框和score后仍在，按层NMS时删除；NMS后105个框、高IoU数量0，最终预算截断不再删框。原NMS返回的索引证明删除阶段，不等于证明哪个框实际抑制了它。

真实GPU日志及原始捕获在outputs/single_030269_run01；本地独立验收在checks/independent_single_030269_run02，追加核对当前运行记录的推理前RGB/IR/XML及共同受保护文件哈希与Day02一致，没有再次模型前向。第一次独立验收记录也保留在run01。当前结果只适用于这一个GT，不推广到整个验证集，单图后停止等待反馈。

原始capture文件为自己生成的张量/字典，不包含凭据。单图适配器的真实范围不扩展到其余五图；在报告解释后停止。

## 后续授权步骤：其余五图及六例合并（已完成）

用户审阅单图后另行授权固定五图：030308/G1、030205/G0、170100/G1、130126/G3、170455/G1，顺序及image_index按Day02 selected_cases.csv。capture_remaining_val.py复用原单图捕获函数的独立模块命名空间；原单图入口与trace_rpn_stages.py没有修改。030269入口在新适配器中明确拒绝实时捕获，只复制已验收19项产物并校验SHA。

新增verify_merged_capture.py、test_remaining_capture.py。完整37项人工测试（27核心+4单图+6新适配）在checks/remaining_adapter_unit_run02通过、退出码0；run01保留。这是单元核验，不是LLVIP结果。

真实服务器命令：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day03\capture_remaining_val.py --device cuda:0 --reuse D:\vision-learning-server\week09\day03\outputs\single_030269_run01 --output D:\vision-learning-server\week09\day03\outputs\remaining_five_merged_run01
~~~

已完成10次新增原eval前向，进程退出码0，未重新前向030269。493项服务器受保护文件、原checkpoint、模型持久state、RPN/transform配置前后未变，全部hooks/包装恢复。原逐层pre150、逐图post150、score0.05、NMS0.7未改。正常与异常恢复的人工回归全部通过；真实前向只验证正常路径，没有人为制造GPU异常。

outputs/remaining_five_merged_run01包含新五图原始捕获、复用单图、84行merged_stage_summary.csv、固定ID轨迹、merged_verification.json、six_case_report.md和run.log。12项最终CPU float32 IoU与Day02实际差值全部为0；float64诊断列明确分开。170455/G1 IR=0.7500129342079163，0.75覆盖判定保持True。

下载压缩包SHA为7b8728e791e0a46b085c6fad75f517556a2e3ed7ec2918e8f68b8ec79788a220；104项文件逐项校验通过，见checks/remaining_transfer_run01。本地独立离线复算12项全部通过、退出码0，见checks/independent_six_case_run01；未加载checkpoint或执行模型。命令：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B week09/day03/verify_merged_capture.py --capture-dir week09/day03/outputs/remaining_five_merged_run01 --output week09/day03/checks/independent_six_case_run01
~~~

主要观察：030269/Fusion和130126/IR在NMS丢失全部达标框；170100/IR在score过滤丢失；030308、030205、170455的Fusion七阶段均无达标框。030205/Fusion虽未达标，NMS仍进一步降低最佳IoU。Top-N/裁剪/小框过滤没有改变本次12项的最大IoU或达标数量；130126两模型最终预算截断实际生效，但指定GT最佳IoU及达标数量未下降。这些是选定六例的操作事实，不是融合特征/回归根因或全val频率。

详细中文交付见DAY03_SIX_CASE_CAPTURE_DELIVERY.md，含真实命令、完整精度、固定ID证据及未验证范围。本步骤完成后停止，后续需用户反馈；不调参、训练或启动新模型实验。


