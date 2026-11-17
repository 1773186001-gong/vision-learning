# Week09 Day01：Faster R-CNN RPN Proposal Recall 诊断交接报告

日期：2026-11-05（北京时间）  
接收对象：ChatGPT，用于理解本次真实实验并接续 Week09 Day02。  
状态：Day01 已完成，本报告仅整理已执行证据，没有启动新实验。

## 1. 请先理解本次结论

在 LLVIP_1000 的 validation 集上，使用原 IR、Fusion 的 best checkpoint、原预处理和原 RPN 配置，已经在 **RPN 后处理输出** 中观察到高 IoU 覆盖差异。

使用全部实际保留候选时：

- IR 的 IoU≥0.75 覆盖为 302/416，Fusion 为 285/416，Fusion 少覆盖 17 个 GT，低约 4.09 个百分点。
- IoU≥0.50 时方向相反：IR 为 407/416，Fusion 为 410/416。
- GT 最大 IoU 均值：IR=0.786114，Fusion=0.778792。

因此，可以说“本验证集上，Fusion 的 RPN 输出在较严格 IoU 下覆盖较弱”；不能说“Fusion 的 RPN 召回全面下降”，也不能直接认定 RPN 模块自身就是根因。

历史 AP75 来自 test，本次 RPN 来自 validation。两者不能作一一对应的因果解释。RPN Proposal Recall 不是最终检测 Recall，也不是 AP75。

## 2. 我的背景和协作方式

我是软件工程硕士研一，正在从基础学习计算机视觉和深度学习。已完成 Week01–Week07 的 CNN、ResNet18、迁移学习、模型评估、注意力机制、FPN/PAN、目标检测框、IoU 和检测损失；Week08 完成了 LLVIP_1000 的单模态与双分支融合检测实验。

请全程使用中文，每次只推进一个明确步骤，解释清楚后等待我的反馈。不要主动安排测验。优先给完整可运行代码、运行命令、预期输出及结果解释。

修改代码前先阅读相关文件并说明修改范围；不要覆盖已完成实验、日志、数据或权重。未知参数应读代码确认，不要编造结果或论文机制。不要擅自安装依赖、升级环境、重新训练或在 test 上反复调参。后续连接服务器或启动新的计算，应确认当前任务授权范围，不要把本次已结束的 SSH 会话视为仍然在线。

如果当前 ChatGPT 不能访问本机路径，不要声称已读取文件；本报告包含核心实测数字。需要看图或分析代码时，可由我补充相应附件。

## 3. 路径与实际环境

~~~text
本地项目：
D:\python\test\vision-learning

本地 Python：
D:\python\.venv\Scripts\python.exe

GPU 服务器项目：
D:\vision-learning-server

GPU 服务器 Python：
D:\cv-server-env\Scripts\python.exe

服务器主机名：
DESKTOP-BFIO0RA
~~~

本次实际核对的服务器环境：Windows 10，Python 3.14.6，PyTorch 2.14.1+cu126，TorchVision 0.29.1+cu126，CUDA build 12.6，cuDNN 91002，NumPy 2.5.4，RTX 3060 Laptop GPU，6GB 显存。

本地 CPU 冒烟环境：Windows 11，Python 3.14.7，PyTorch 2.14.1+cpu，TorchVision 0.29.1+cpu，NumPy 2.5.3。本机 AMD 显卡未用于本次推理。

本次连接时使用的局域网地址为 192.168.0.149，可能变化。结束前检查到诊断 Python 进程数为 0，随后 SSH 正常退出，退出码 0；没有下发服务器关机命令。这是当时状态，不代表下一次对话时服务器仍然在线。

## 4. 已有 Week08 背景

LLVIP_1000：train/val/test=700/150/150 对，person GT 分别为 1991/416/348。

| split | 场景前缀 |
| --- | --- |
| train | 01、02、04、05、06、07、09、10、11、12、14、15、16、18、25 |
| val | 03、08、13、17 |
| test | 19、20、21、22、23、24、26 |

val 来自官方 train 的独立场景，test 来自官方 test；不是本次临时随机切分。数据按 manifest 原顺序读取。本地检查 val 的 150 对图像完整、尺寸一致，有 416 个有效 person 框，空 GT 图片为 0。

历史测试结果如下，本次没有重跑 test：

| 模型 | AP50 | AP75 | mAP50:95 |
| --- | ---: | ---: | ---: |
| IR | 0.898907 | 0.432512 | 0.471626 |
| Fusion | 0.903234 | 0.338772 | 0.439998 |

已有离线分析支持 Fusion 的最终检测高 IoU 定位表现弱于 IR，但此前尚未检查真实 RPN proposals。本次 Day01 正是补这一步。

历史材料实际位于：

~~~text
week08/day03/DAY03_IR_TRAINING_GPT_HANDOFF.md
week08/day05/DAY05_FUSION_TRAINING_GPT_HANDOFF.md
week08/day05/localization_analysis/DAY05_LOCALIZATION_ANALYSIS_GPT_HANDOFF.md
~~~

这些相对路径均以项目根目录为基准，不能假设文件位于根目录。

## 5. 实际复用的模型、代码和 checkpoint

检查并复用了以下 Week08 文件，未修改它们：

| 文件 | 用途 |
| --- | --- |
| week08/day01/llvip_dataset.py | manifest 顺序、RGB/IR/XML 配对、person 标注、像素 xyxy |
| week08/day01/prepare_llvip_subset.py | 场景分组及划分规则 |
| week08/day02/paired_letterbox.py | 同步 Letterbox |
| week08/day02/paired_horizontal_flip.py | 训练翻转、验证不翻转 |
| week08/day03/ir_common.py | IR 模型构造、checkpoint_model()、原 loader |
| week08/day03/train_ir.py、evaluate_ir.py | IR 保存、恢复及评估路径 |
| week08/day05/fusion_model.py | DualFPN、PairedDetector、restore() |
| week08/day05/fusion_common.py | paired loader 及设备搬运 |
| week08/day05/train_fusion.py、evaluate_fusion.py | Fusion 保存、恢复及评估路径 |

IR：Faster R-CNN MobileNetV3 Large 320 FPN 构造器，实际 min_size=max_size=640，person+background 两类。恢复时调用原 checkpoint_model()，构造完整恢复骨架并 strict=True 加载。

Fusion：两个独立 MobileNetV3+FPN 分支，对应实际尺度 Concat+1×1 Conv，然后共享 RPN 和 RoI Head。恢复调用原 restore()，真正的 Faster R-CNN 位于 model.detector。

实际特征键为 **0、1、pool**，不是标准 P2/P3/P4/P5。Fusion 的 RoI 配置已过滤为 0、1，不要擅自改结构。

一个措辞澄清：自动诊断报告中 IR 的“RoI实际特征键”记录为 0、1、2、3，这实际上是 box_roi_pool.featmap_names 的配置列表。TorchVision RoI pooling 只取输入特征字典中存在且名字命中的项；结合该骨干结构，参与 RoI 的是 0、1。Fusion 则在构造时显式把配置列表过滤为 0、1。RPN 使用包括 pool 在内的全部实际尺度。不要把 IR 的配置列表误当作存在四个骨干尺度，也不要为统一显示而修改原模型。

服务器实际加载的文件：

~~~text
IR：
D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth
epoch=9
bytes=151529396
SHA256=9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2

Fusion：
D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth
epoch=6
bytes=190083098
SHA256=9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a
~~~

本地相同文件分别位于项目根下相同的 week08 相对路径。本次服务器与本地的两份权重、六个核心源文件和 manifest 的 SHA256 均一致。

checkpoint 是包含 model、optimizer、epoch、history、config、trainable_names、RNG 等字段的字典，不是单独的 state_dict。诊断只恢复模型，不创建优化器、不恢复训练流程、不更新权重。RPN 的非张量配置并未完整存入 checkpoint，必须从恢复后的真实模型读取，不能仅凭权重文件推断。

manifest SHA256：

~~~text
8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb
~~~

## 6. 本次实现如何保证提取正确

新增文件：

~~~text
D:\python\test\vision-learning\week09\day01\rpn_diagnostic.py
D:\python\test\vision-learning\week09\day01\README.md
~~~

只使用 val。复用原 Fusion paired loader，并对 IR 输入逐样本与原 IR dataset 的图像和 target 字段做严格一致性检查。

输入及几何过程：

1. RGB 转三通道；IR 转灰度并复制为三通道；float32，像素范围 [0,1]。
2. 两路与 GT 同步 Letterbox 到 640×640，fill=114/255，val 翻转概率为 0。
3. 本 val 原图均为 1280×1024，缩至 640×512，上下各补 64；GT 为 x'=0.5x、y'=0.5y+64。
4. Fusion 输入字典为 visible、infrared。wrapper 暂时拼成六通道，只为共享内部 transform，进入独立骨干前拆回两路，不是六通道卷积。
5. 正常执行原 torchvision transform；原 min/max 均为 640，因此此处几何尺寸不再改变，归一化仍正常执行。
6. model.eval() 与 torch.inference_mode() 下执行原模型完整前向，同时传入 targets。
7. 通过临时只读 transform/RPN forward hooks，捕获同次内部 transform 后的 GT 和 RPN 后处理 proposals；最终 RoI 检测输出不用于指标。hooks 退出时移除。
8. 实际检查变换尺寸、GT 一致性、有限数、有效框和原候选上限。IoU 始终在相同的内部像素 xyxy 坐标中计算。

两模型实际 RPN 配置一致：

| 参数 | 原实际值 |
| --- | ---: |
| train pre/post NMS Top-N | 2000/2000 |
| eval pre-NMS Top-N | 150（逐层筛选） |
| eval post-NMS Top-N | 150（每图输出上限） |
| RPN objectness 阈值 | 0.05 |
| RPN NMS IoU | 0.7 |

TopK=100、300、1000，取原 RPN 完成解码、裁边、小框/分数过滤、按层 NMS 和最终截断后的前 K 个框。原 batched_nms 返回 objectness 降序索引，RPN 保留该顺序；没有用最终分类分数重新排序，也没有提高原预算。

**Top300 和 Top1000 实际使用同一批最多150个候选，不是两个独立预算实验。** 每图不足 K 时使用全部实际候选并记录。

对每个 GT 计算与所取候选的最大 IoU，再统计 IoU≥0.50/0.75 的 GT 数。全局 Recall 分母为全部有效 GT 数，不是逐图 Recall 平均。一个 proposal 可以覆盖多个 GT，不做 AP 所用的贪心一对一匹配。空 GT 的逐图 Recall 留空，不进入分母；无 proposals 时该图每个 GT 的最大 IoU 为 0。本次真实 val 没有空 GT，不能宣称实际覆盖过空 GT 分支。

## 7. 实际运行及版本记录

固定 seed=42、threads=2、batch_size=1、num_workers=0、shuffle=False，按 manifest 确定顺序运行。cudnn deterministic=True、benchmark=False。没有声称 CPU/GPU 逐位一致。

三次运行都成功：

| 运行 | 设备 | 每模型图片数 | 每模型 GT | 状态 |
| --- | --- | ---: | ---: | --- |
| cpu_smoke_run01 | 本地 CPU | 3 | 7 | 退出码0 |
| gpu_smoke_run01 | 服务器 cuda:0 | 3 | 7 | 退出码0 |
| val_rpn_run01 | 服务器 cuda:0 | 150 | 416 | 退出码0 |

实际执行命令如下，仅用于记录；这些目录已经有结果，不要直接重跑覆盖。

本地 CPU 冒烟，在本地项目根目录执行：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B week09/day01/rpn_diagnostic.py --device cpu --limit 3 --seed 42 --threads 2 --output week09/day01/outputs/cpu_smoke_run01
~~~

服务器 GPU 冒烟：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day01\rpn_diagnostic.py --device cuda:0 --limit 3 --seed 42 --threads 2 --output D:\vision-learning-server\week09\day01\outputs\gpu_smoke_run01
~~~

服务器全 validation：

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day01\rpn_diagnostic.py --device cuda:0 --seed 42 --threads 2 --output D:\vision-learning-server\week09\day01\outputs\val_rpn_run01
~~~

正式运行时间为北京时间 2026-11-05 13:37:40 至 13:38:36。退出码另存 execution_status.json，run.log 包含完整逐图记录，末尾实际输出包括：

~~~text
MODEL_DONE model=IR images=150 gt=416 state_unchanged=True checkpoint_unchanged=True
MODEL_DONE model=Fusion images=150 gt=416 state_unchanged=True checkpoint_unchanged=True
COMPLETED D:\vision-learning-server\week09\day01\outputs\val_rpn_run01
~~~

版本说明：CPU 冒烟使用初版脚本；随后只修正命令记录以保留解释器 -B、增加 NMS 源码校验信息，并在图和报告中明确原候选上限，没有改变指标计算和模型提取逻辑。CPU 初版日志的 command 字段漏记了 -B，不代表实际没有使用 -B。GPU 冒烟及全量使用相同最终脚本。

~~~text
CPU初版脚本SHA256：
618a64ffb047cb66927f2e26bcea326d2e907ffc8074f4ac1c26f4fb5735d650

GPU最终版及当前本地脚本SHA256：
06dee2f5ad1194d1eedee6bc8e437e79606a14ddbc38f16e61aca5adb3bd866a
~~~

CPU 与 GPU 冒烟有少量 IoU/候选数差异，不能混合作为正式指标；例如第三张 Fusion 的候选数，CPU=65，GPU=66。正式结论只引用服务器 GPU 全量结果。

曾遇到的执行问题如实记录：受限连接首先被拒绝，之后免密认证失败，再通过用户授权的交互式密码认证完成连接。结果取回时，首次解包校验因 Windows 反斜杠与 ZIP 正斜杠不一致而失败，发生在写入前；统一路径表示后成功校验解包，没有因此重跑模型，也没有改变原结果。

## 8. 正式 GPU 全量结果

以下每项的 GT 分母均为 416：

| 模型 | TopK | Recall@0.50（命中数） | Recall@0.75（命中数） | 平均最大 IoU |
| --- | ---: | ---: | ---: | ---: |
| IR | 100 | 0.978365（407/416） | 0.718750（299/416） | 0.784562 |
| IR | 300 | 0.978365（407/416） | 0.725962（302/416） | 0.786114 |
| IR | 1000 | 0.978365（407/416） | 0.725962（302/416） | 0.786114 |
| Fusion | 100 | 0.983173（409/416） | 0.682692（284/416） | 0.776859 |
| Fusion | 300 | 0.985577（410/416） | 0.685096（285/416） | 0.778792 |
| Fusion | 1000 | 0.985577（410/416） | 0.685096（285/416） | 0.778792 |

| TopK | Recall@0.50差值 | Recall@0.75差值 | 平均最大IoU差值 |
| --- | ---: | ---: | ---: |
| 100 | +0.004808 | -0.036058 | -0.007703 |
| 300 | +0.007212 | -0.040865 | -0.007322 |
| 1000 | +0.007212 | -0.040865 | -0.007322 |

差值均为 Fusion−IR。Recall 差值换成百分数后应写“百分点”；IoU 均值差是绝对 IoU 差，不是准确率下降百分比。

每图原 proposals：IR 为 13–150 个，均值 87.086667；Fusion 为 17–150 个，均值 102.220000。候选不足100的图片：IR=92，Fusion=75。两模型均无空 GT 图片、无零候选图片。Fusion 候选数量更多，并不意味着高 IoU 覆盖更好。

使用全部实际候选时，GT 最大 IoU 分布：

| 模型 | Q10 | Q25 | 中位数 | Q75 | Q90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IR | 0.660255 | 0.740698 | 0.803346 | 0.860516 | 0.900047 |
| Fusion | 0.640699 | 0.730137 | 0.795405 | 0.852288 | 0.892598 |

Fusion 的这些分位数和均值均较低，但 IoU≥0.50 覆盖更高，不能说整个分布严格整体左移或所有 GT 都更差。

## 9. 逐 GT 配对与场景差异

在全部原候选、IoU=0.75 下，按相同 stem+gt_index 配对：

| GT 覆盖关系 | 数量 |
| --- | ---: |
| 两模型都覆盖 | 255 |
| 仅 IR 覆盖 | 47 |
| 仅 Fusion 覆盖 | 30 |
| 两模型都未覆盖 | 84 |

共 416 个 GT。净差为 47−30=17，不能只看 IR 优势案例而忽略 Fusion 优势反例。

同 GT 的最大 IoU：IR 更高229个，Fusion更高186个，相等1个。Fusion−IR 的配对 IoU 差均值为 -0.007321534，中位数为 -0.012813926。

按场景查看全部原候选的 IoU≥0.75 覆盖：

| 场景 | 图片 | GT | IR 覆盖数 | Fusion 覆盖数 | Fusion−IR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 03 | 38 | 93 | 78 | 63 | -15 |
| 08 | 38 | 121 | 90 | 90 | 0 |
| 13 | 37 | 131 | 91 | 92 | +1 |
| 17 | 37 | 71 | 43 | 40 | -3 |

总净差17个中，场景03贡献15个。因此后续宜优先看场景03，同时保留其他场景和 Fusion 优势反例。场景差异只是描述性观察，尚不能解释为光照、配准、背景或其他具体机制。

从 Top100 扩至全部原候选，IR 的高 IoU 覆盖只增加3个GT，Fusion增加1个GT，差距仍存在。因此“仅仅被前100排序截断”不足以单独解释差距；但本次仍没有检验原 pre/post150、score阈值及NMS之前是否存在更优框。

## 10. 输出文件及独立核验

本地正式输出目录：

~~~text
D:\python\test\vision-learning\week09\day01\outputs\val_rpn_run01
~~~

服务器同名正式输出目录：

~~~text
D:\vision-learning-server\week09\day01\outputs\val_rpn_run01
~~~

本地已取回七项主产物：

- rpn_metrics.json：真实环境、命令、权重路径与校验、源代码校验、数据顺序、模型配置及全局统计。
- rpn_per_image.csv：每图一行的宽表，含 IR/Fusion proposal 数、各K使用数、命中数、Recall和Fusion−IR差异。
- rpn_per_gt.csv：IR/Fusion 长表，每模型416行，共832行；含 GT 坐标、proposal 数、各K最大IoU和覆盖标记。
- proposal_recall_comparison.png：两模型的候选覆盖对比。
- max_iou_distribution.png：GT最大IoU分布对比。
- rpn_diagnostic_report.md：运行自动生成的中文报告。
- run.log：真实完整终端输出。

另有 execution_status.json，以及在本地独占新建的 independent_verification.json / independent_verification.log。独立核验文件是在下载后本地生成的，没有把它们当作服务器原始输出，也没有重新上传或改写原报告。

GPU冒烟和全量结果包SHA256：

~~~text
60ce18a794cdac34ae8cfce6e67ad3dbf4c49246edabeae21ac5cd64c0f3fc6f
~~~

已验证 ZIP 与服务器一致，16个原始文件逐一校验通过。独立核验用 CSV 的逐GT最大IoU重新统计全局 hits、Recall 和均值，与 JSON 一致；逐图计数与差值、TopK单调性、Top300=Top1000、两模型GT几何、原150上限均通过。

GPU冒烟与全量前3图再次比较：proposal数量最大差=0，42项最大IoU比较值最大差=0.0（容差1e-6）。这是同GPU、同样本的一致性检查，不是再次跑完整验证集。

独立核验不运行模型，也未从原proposals坐标重新计算IoU；它检查已输出表格与汇总的一致性。原proposals坐标和objectness分数未保存，不能声称对它们做了独立数值重算。所有被审计的11个输入文件前后SHA一致。未做显著性检验、因果干预、多seed或重训练。

两张主图已实际打开检查，图中明确标注 IR/Fusion 原 post-NMS 上限为150。

![Proposal Recall 对比](D:/python/test/vision-learning/week09/day01/outputs/val_rpn_run01/proposal_recall_comparison.png)

![GT最大IoU分布](D:/python/test/vision-learning/week09/day01/outputs/val_rpn_run01/max_iou_distribution.png)

## 11. 可以确认、可以假设、尚待验证的内容

**真实观测**

Fusion 在本 validation 的 RPN 后处理输出中，高 IoU 覆盖和平均最大IoU较低；低 IoU 覆盖反而略高。差异随场景和具体GT变化，场景03贡献主要净差。改变只读诊断的前100切片到全部原候选后，高IoU差距仍在。

**合理假设**

这一现象与既有“最终检测精细定位较弱”的方向一致，值得进一步追踪候选几何质量及筛选过程。Fusion 的特征、RPN框回归或候选过滤/排序都可能参与，但目前不能区分各自作用。

**仍需验证**

- 差异是否主要来自融合特征、RPN解码/框回归，还是原score/NMS/预算筛选。
- 对相同GT，RoI框回归是改善还是损害候选的IoU。
- 最终分类分数、排序和NMS是否进一步放大或缩小差异。
- 场景03的具体失败几何形态，以及Fusion优势反例。
- 在同一validation集合上，候选质量与最终检测之间的联系。

本次不是因果证明，不能把全部历史test AP75下降归因于RPN，也没有证明换anchor、扩大Top-N、加注意力或重新训练就会改善。

## 12. Week09 Day02 建议与下一轮起点

下一步先做一个明确任务：**固定本次best权重、validation和原配置，可视化高IoU覆盖不一致的GT。**

建议先从现有逐GT表按 stem+gt_index 配对，选取少量“IR≥0.75而Fusion<0.75”的案例，并选取“Fusion≥0.75而IR<0.75”的对照。优先查看场景03，按最大IoU差值选择典型案例，同时保留接近0.75边界的案例，避免只挑极端图。

一个实现限制必须记住：当前CSV保存的是 **GT坐标** 和最大IoU，未保存proposals坐标、最佳proposal索引或objectness分数。不能直接从CSV画出原候选框。先用CSV确定少量图像，再在新的 Day02 目录中对这些val样本执行同best的只读前向，保存所需候选信息及绘图；这只是建议，尚未执行。

个例可视化完成后，再决定是否追踪同GT在RoI回归前后、最终score排序及NMS后的变化。当前不要直接开始训练、换融合模块、扩大RPN预算或重新评测test。

请接手的ChatGPT先向我解释：

1. 为什么本次“高IoU覆盖更差、低IoU覆盖更好”不矛盾。
2. 为什么能确认RPN输出有差异，却还不能锁定RPN模块自身为根因。
3. 为什么Day02应同时看IR优势与Fusion优势案例。

解释后等待我反馈，再推进这一个Day02步骤。若要查看图像或代码，优先使用上面列出的真实产物；不要假设能从本地路径自动读取，也不要编造尚未完成的Day02结果。

