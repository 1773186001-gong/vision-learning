# Week09 Day02：RPN Proposal 定位误差可视化交接报告

日期：2026-11-05（北京时间）  
接收对象：ChatGPT，用于接续 Week09 Day03。  
当前状态：Day02 已完成；本文件只整理真实证据，没有再次推理或启动 Day03。

## 1. 先读本次结论

这次不是训练，也不是完整验证集重评。我们从 Day01 正式 GPU 结果中按明确规则选出6个GT，对对应6张validation图片，用原IR/Fusion best各执行一次只读前向，提取真实RPN后处理框和对应objectness，并完成全部6张对比图。

12项“GT×模型”的最大IoU与Day01逐值相同。三个场景03的IR优势案例中，Fusion最佳框都偏宽；两个Fusion优势反例中，IR主要表现为向上偏移。临界案例IR的IoU为0.750012934，仍有可见位置误差，只是略高于0.75。

当前证据支持继续检查RPN候选的框回归与筛选过程，不足以直接决定修改融合模块。这是经过差值和阈值挑选的个例，不能据此估计整个验证集的偏差比例或证明根因。

## 2. 我的背景与协作约定

我是软件工程硕士研一，正在从基础学习计算机视觉和深度学习，已完成Week01–Week07基础实验及Week08的LLVIP单模态/双分支检测训练。请用中文，每次只推进一个明确步骤，解释后等待我反馈，不主动安排测验。

修改代码前先读真实文件并说明范围。不要覆盖Week08、Week09 Day01/Day02的代码、数据、日志、结果或权重。不要擅自安装依赖、升级环境、开始训练、扩大RPN Top-N或在test上反复调参。后续连接服务器或执行新计算，应确认当前任务授权；不能把已经退出的旧SSH会话当作在线会话。本报告不携带任何凭据。

若当前ChatGPT不能访问本机路径，不要声称已经读过文件。本报告内给出了核心数值；需要直接查看图像时，可另外提供PNG附件。

## 3. 路径与已有模型

~~~text
本地项目：D:\python\test\vision-learning
本地Python：D:\python\.venv\Scripts\python.exe
服务器项目：D:\vision-learning-server
服务器Python：D:\cv-server-env\Scripts\python.exe
服务器主机：DESKTOP-BFIO0RA
~~~

本次服务器实际环境：Windows10、Python3.14.6、PyTorch2.14.1+cu126、TorchVision0.29.1+cu126、NumPy2.5.4、RTX3060 Laptop GPU 6GB，cuda:0。固定seed=42、threads=2，cudnn deterministic=True、benchmark=False。

两份实际checkpoint：

~~~text
IR：
D:\vision-learning-server\week08\day03\outputs\ir_baseline_run01\best.pth
SHA256=9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2

Fusion：
D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01\best.pth
SHA256=9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a
~~~

本地有对应相同week08相对路径，权重SHA一致。IR为Faster R-CNN MobileNetV3 FPN；Fusion为独立RGB/IR MobileNetV3+FPN，对应尺度Concat+1×1 Conv，共享RPN与RoI Head。

真实骨干特征键为0、1、pool，不要改成P2–P5。Fusion的RoI配置为0、1。IR的RoI配置列表可能显示0、1、2、3，但库只使用输入字典实际存在且匹配的特征，不能把配置列表当作四个实际骨干尺度。

## 4. Day01承接背景

LLVIP_1000的train/val/test为700/150/150对，person GT为1991/416/348；val场景03、08、13、17，与其他split场景隔离。Day01使用全部150张validation图、416个GT，原RPN eval pre/post NMS上限都是150。

|模型|全部原候选Recall@0.50|Recall@0.75|平均最大IoU|
|---|---:|---:|---:|
|IR|407/416|302/416|0.786114|
|Fusion|410/416|285/416|0.778792|

IoU≥0.75：两模型都覆盖255、仅IR47、仅Fusion30、两者都未覆盖84。场景03对高IoU覆盖净差贡献15个GT。

Day01的Top300与Top1000都是全部实际保留的最多150个候选，是同一实际集合，不是两个独立预算实验。Day01 CSV只有GT坐标和最大IoU等统计，没有proposal坐标和objectness，因此Day02才对选中少量图片重新只读前向。

历史test AP75为IR=0.432512、Fusion=0.338772；本次没有重跑test。validation RPN指标不是test AP75，也不是最终检测Recall，不能据此直接作因果归因。

Day01正式产物：week09/day01/outputs/val_rpn_run01/。完整背景另见week09/day01/DAY01_RPN_DIAGNOSTIC_GPT_HANDOFF.md。

## 5. Day02选择规则与真实索引

先检查实际CSV列名，再按stem+gt_index配对，gt_index从0开始。使用max_iou_top300，并确认与max_iou_top1000相同，表示全部实际后处理候选。

1. IR优势3个：IR≥0.75且Fusion<0.75；优先场景03，再按IR−Fusion IoU降序。
2. Fusion优势2个：Fusion≥0.75且IR<0.75；按Fusion−IR IoU降序。
3. 临界1个：排除前面已选GT，至少一侧IoU在[0.70,0.80]；优先跨越0.75，再按任一侧距0.75的最小距离升序。

平局按stem、gt_index升序。候选池IR优势47个、Fusion优势30个；未放宽规则。最终选例顺序为030269/G1、030308/G1、030205/G0、170100/G1、130126/G3、170455/G1，恰好6张不同图片。实际前向按原image_index升序，模型先IR后Fusion。

## 6. 原模型与提取方法

新增本地文件：week09/day02/visualize_proposals.py、README.md。没有修改Week08或Day01。

主要复用：

- week09/day01/rpn_diagnostic.py：原恢复入口、paired loader、CaptureRPN、坐标/状态检查。
- week08/day03/ir_common.py：IR checkpoint_model()及原dataset。
- week08/day05/fusion_model.py：Fusion restore()和PairedDetector。
- week08/day05/fusion_common.py：原paired loader。
- week08/day01/llvip_dataset.py及week08/day02两个paired预处理模块。

IR输入与原IR dataset逐图逐字段严格核对。Fusion仍是visible/infrared字典；临时打包只为共享内部transform，骨干前拆回两路，不是六通道卷积。

model.eval()+torch.inference_mode()下正常执行原完整forward，最终RoI检测输出不用于计算。当前TorchVision的RegionProposalNetwork.filter_proposals实际返回(final_boxes,final_scores)，分数是sigmoid(objectness)，已经与原分数过滤/NMS/post-NMS截断后的框对应。

临时包装调用原绑定方法、复制输出并原样返回，与RPN forward真实框逐位核对；finally中恢复方法原始查找状态并移除hooks。没有替换筛选规则、扩大预算或修改持久模型结构。

每个GT从该模型全部实际保留框中取最大IoU框，objectness取同一框对应的分数，绝不是取最高分框代替最大IoU框。并列时argmax取原顺序第一个。GT索引和保存的proposal索引从0开始，objectness排名从1开始。

原RPN仍为eval pre/post=150/150、score阈值0.05、NMS IoU=0.7。模型配置与Day01逐项比较，权重及模型持久state前后SHA一致。

几何坐标使用640×640 Letterbox与同次模型transform后的xyxy。原图1280×1024缩为640×512，上下各pad64，GT同步变换。dx=proposal中心x−GT中心x，正值向右；dy正值向下；归一化分别除GT宽高；宽高比例为proposal/GT。顶部原始RGB/IR不叠加640框，底部两个模型使用相同红外底图及相同放大范围，Fusion推理输入仍保留RGB+IR。

## 7. 运行记录、文件和版本差异

GPU正式运行为北京时间2026-11-05 14:04:59至14:05:11；两模型各6图，退出码0。真实命令和末尾输出见后面的逐图结果及run.log。

本地正式输出：

~~~text
D:\python\test\vision-learning\week09\day02\outputs\proposal_visualization_run01
~~~

服务器对应输出：

~~~text
D:\vision-learning-server\week09\day02\outputs\proposal_visualization_run01
~~~

文件含selected_cases.csv、proposal_details.csv、geometry_analysis.csv、全部6张case_*.png、day02_report.md、run.log、execution_status.json、process_exit_status.json、run_metadata.json和rpn_proposals.json。

**Day02新增保存了rpn_proposals.json**：里面是两模型六张图的全部真实后处理框、对应sigmoid objectness和GT。Day03可直接读取做后处理候选的进一步离线分析；但本次没有保存NMS前候选、anchor级原始回归量或每一步被剔除原因。研究这些中间阶段仍需要对少量固定val图做新的只读捕获。

原始GPU图有局部标题/图例拥挤，之后仅调整绘图代码，在本地从保存的CSV重绘到visual_review，未重新加载模型或执行forward。原始结果完整保留。最终优先看visual_review_report.md及visual_review/下全部6张图。

~~~text
GPU运行时visualize_proposals.py SHA256：
fa6b60d5b722db2ff85aae55cb4abb9f01dca8926ece03567ea33a6a3b86fdf7

当前本地脚本SHA256（加了仅重绘入口和排版调整）：
12422fccc8449f26c19c1e8d49472e79922ce3ce3edba66307ec39d3515fb6f4

原始结果ZIP SHA256：
5b254dcfb6b898428439a4805b67b8753a189fc9611d55863e6206e85e9bb481
~~~

不能把两版脚本不同的SHA误认为权重变化或传输损坏。原始15项服务器产物下载后逐一SHA通过，重绘后再次核对仍未改变。独立复核从保存的全部框用Python float64重算最大IoU，与Torch float32的最大绝对误差为1.2630032719673778e-07；最佳框索引、分数和几何量均通过，几何量最大误差0。12项IoU与Day01差值为0，38个本地历史文件未改变。

独立核验为本地离线复算，无模型前向，记录在independent_verification.json/.log。仅重绘命令、原输入校验及无forward记录在visual_review/render_metadata.json。原始6张及重绘6张都已逐张打开检查。

实际问题记录：本次免密认证被拒绝后使用此前获准的交互认证完成连接；没有把凭据写入文件。会话结束前诊断进程数检查为0，随后SSH正常退出；这只是当时状态，不代表下次仍在线。本次生成交接文件不连接服务器。

## 8. 已完成和未完成的边界

已完成：真实选例、两模型六图只读GPU推理、后处理框与objectness保存、最大IoU框选择、中心/宽高/边界误差分析、全部案例图、独立数值与文件校验。

未完成：RPN各筛选阶段的候选去留追踪、anchor/框回归中间量分析、RoI前后变化、因果检验、显著性检验及新模型消融。未训练、未运行test、未扩大原预算。本次个例不具备随机抽样代表性。

## 9. 给接手ChatGPT的建议

先解释两个现象：为什么高objectness不保证高IoU，以及为什么Fusion部分偏宽与IR在反例中上移可以同时存在。区分实际观察与机制猜测，不能把车灯、树木、光照或配准直接写成已证实原因。

Day03先不改融合模块。建议固定这六张val图和原权重，按GT追踪原解码候选、逐层pre-NMS筛选、score过滤、NMS和post-NMS预算各步的最佳IoU、分数排名与去留。如果高IoU框曾存在但被删，查排序/筛选；如果解码结果始终缺乏好框，再查回归及特征。保持原预算，不靠扩大Top-N制造改善。

下一轮先读现有rpn_proposals.json及对应案例，提出一个明确的中间阶段捕获步骤，解释后等待我反馈，再实施。Day03尚未执行，不要把建议写成已经取得的结果。

---

## 10. 六例完整坐标、分数、几何量与逐图审阅

下面嵌入本次已保存的逐图结果，便于只上传本交接文件也能阅读完整数值。图像链接指向本机文件；ChatGPT无法访问时需要另外提供图片附件。

## Week09 Day02：逐图审阅与最终结果说明

状态：六张验证图片已在RTX3060上分别完成IR/Fusion只读前向，进程退出码0；六张最终图已逐张打开检查。未训练、未运行test、未改原RPN预算或任何历史权重。

### 运行与复核证据

真实GPU命令：

~~~powershell
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day02\visualize_proposals.py --device cuda:0 --seed 42 --threads 2 --output D:\vision-learning-server\week09\day02\outputs\proposal_visualization_run01
~~~

原始GPU输出保留在本目录。初版部分图例和放大标签拥挤，随后仅用保存的CSV在本地重绘到visual_review，模型没有再次前向。当前脚本多了该重绘入口，服务器运行时源版本SHA仍保存在run_metadata.json，不将两版源文件混作同一哈希。
本次原始结果包SHA256：5b254dcfb6b898428439a4805b67b8753a189fc9611d55863e6206e85e9bb481；下载后15项原始文件逐一SHA256核验通过。
独立使用Python float64从全部原RPN候选重算最大IoU，最大误差1.2630032719673778e-07；最佳索引、对应分数、几何量通过；12项本次IoU与Day01完全一致。38个本地历史文件校验未变，服务器也记录了原输入和模型state前后校验。
证据：[GPU运行日志](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/run.log)、[退出状态](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/execution_status.json)、[独立复核日志](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/independent_verification.log)、[完整环境与校验](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/run_metadata.json)、[仅重绘记录](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/render_metadata.json)。

### 六个案例

坐标均为同次原Letterbox/模型transform后的640×640像素xyxy。dx正值向右，dy正值向下。绿色实线GT、蓝色虚线最大IoU的原后处理候选。原始RGB/IR在上方作参考，下方两格使用相同红外底图及相同放大区域；Fusion仍输入RGB+IR。

#### 01_IR_advantage_030269_G1

场景03；gt_index从0开始。选择原因：IR>=0.75>Fusion; scene03 first, then descending IR-Fusion IoU。
GT xyxy=[166.5, 203.0, 208.0, 290.5]。

|模型|最大IoU|objectness|最佳proposal xyxy|分数排名/候选数|
|---|---:|---:|---|---|
|IR|0.793667018|0.176903009|[163.464096, 195.282715, 205.988846, 287.749756]|30/57|
|Fusion|0.605291903|0.996665895|[160.723663, 196.25853, 215.076019, 306.634094]|3/105|

|模型|GT中心|Proposal中心|dx,dy(px)|dx/GT宽,dy/GT高|GT宽高|Proposal宽高|宽/高比例|
|---|---|---|---|---|---|---|---|
|IR|(187.250000, 246.750000)|(184.726471, 241.516235)|(-2.523529, -5.233765)|(-0.060808, -0.059814)|(41.500000, 87.500000)|(42.524750, 92.467041)|(1.024693, 1.056766)|
|Fusion|(187.250000, 246.750000)|(187.899841, 251.446312)|(0.649841, 4.696312)|(0.015659, 0.053672)|(41.500000, 87.500000)|(54.352356, 110.375565)|(1.309695, 1.261435)|

放大图中Fusion框在四周扩张，尤其下边界偏下；宽/高比例为1.310/1.261，IR为1.025/1.057。Fusion的中心偏移不大，但框尺寸明显偏大。Fusion objectness约0.996666、IoU仅0.605292，而IR最佳几何框objectness约0.176903、IoU为0.793667：高objectness不保证高IoU。

![01_IR_advantage_030269_G1](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/case_01_IR_advantage_030269_G1.png)

#### 02_IR_advantage_030308_G1

场景03；gt_index从0开始。选择原因：IR>=0.75>Fusion; scene03 first, then descending IR-Fusion IoU。
GT xyxy=[50.5, 64.5, 84.5, 98.0]。

|模型|最大IoU|objectness|最佳proposal xyxy|分数排名/候选数|
|---|---:|---:|---|---|
|IR|0.878793776|0.148354620|[49.192108, 66.806946, 84.083038, 98.350891]|26/58|
|Fusion|0.712219298|0.178818986|[43.039501, 60.2519, 83.751404, 98.052574]|36/105|

|模型|GT中心|Proposal中心|dx,dy(px)|dx/GT宽,dy/GT高|GT宽高|Proposal宽高|宽/高比例|
|---|---|---|---|---|---|---|---|
|IR|(67.500000, 81.250000)|(66.637573, 82.578918)|(-0.862427, 1.328918)|(-0.025365, 0.039669)|(34.000000, 33.500000)|(34.890930, 31.543945)|(1.026204, 0.941610)|
|Fusion|(67.500000, 81.250000)|(63.395452, 79.152237)|(-4.104548, -2.097763)|(-0.120722, -0.062620)|(34.000000, 33.500000)|(40.711903, 37.800674)|(1.197409, 1.128378)|

目标GT贴近有效图像上边缘，尺寸为34×33.5像素；放大图显示Fusion候选向左、向上多扩了一部分。其中心左移4.105px、宽比例1.197；IR中心左移0.862px、宽比例1.026。不能仅据原图亮区判断它造成了偏差，也不能将该边界GT简单当作完整直立行人。

![02_IR_advantage_030308_G1](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/case_02_IR_advantage_030308_G1.png)

#### 03_IR_advantage_030205_G0

场景03；gt_index从0开始。选择原因：IR>=0.75>Fusion; scene03 first, then descending IR-Fusion IoU。
GT xyxy=[383.5, 306.5, 425.0, 439.0]。

|模型|最大IoU|objectness|最佳proposal xyxy|分数排名/候选数|
|---|---:|---:|---|---|
|IR|0.851542413|0.998569250|[383.28241, 306.590332, 431.32782, 436.924316]|1/31|
|Fusion|0.696796298|0.597035766|[371.226837, 299.399841, 426.258881, 434.45929]|11/43|

|模型|GT中心|Proposal中心|dx,dy(px)|dx/GT宽,dy/GT高|GT宽高|Proposal宽高|宽/高比例|
|---|---|---|---|---|---|---|---|
|IR|(404.250000, 372.750000)|(407.305115, 371.757324)|(3.055115, -0.992676)|(0.073617, -0.007492)|(41.500000, 132.500000)|(48.045410, 130.333984)|(1.157721, 0.983653)|
|Fusion|(404.250000, 372.750000)|(398.742859, 366.929565)|(-5.507141, -5.820435)|(-0.132702, -0.043928)|(41.500000, 132.500000)|(55.032043, 135.059448)|(1.326073, 1.019317)|

两模型的候选在高度方向都比较接近GT，Fusion主要表现为左侧外扩和偏宽：左边界误差约−12.273px，宽比例1.326，中心左移5.507px。IR宽比例1.158，中心右移3.055px，整体重合更高。图中存在车灯亮区，但本次未验证它与误差的因果联系。

![03_IR_advantage_030205_G0](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/case_03_IR_advantage_030205_G0.png)

#### 04_Fusion_advantage_170100_G1

场景17；gt_index从0开始。选择原因：Fusion>=0.75>IR; descending Fusion-IR IoU。
GT xyxy=[167.5, 440.5, 211.5, 541.5]。

|模型|最大IoU|objectness|最佳proposal xyxy|分数排名/候选数|
|---|---:|---:|---|---|
|IR|0.578785658|0.999094367|[164.777603, 413.053833, 211.601273, 518.769409]|1/67|
|Fusion|0.845996797|0.145264372|[170.099991, 435.344849, 210.65538, 537.564331]|19/57|

|模型|GT中心|Proposal中心|dx,dy(px)|dx/GT宽,dy/GT高|GT宽高|Proposal宽高|宽/高比例|
|---|---|---|---|---|---|---|---|
|IR|(189.500000, 491.000000)|(188.189438, 465.911621)|(-1.310562, -25.088379)|(-0.029786, -0.248400)|(44.000000, 101.000000)|(46.823669, 105.715576)|(1.064174, 1.046689)|
|Fusion|(189.500000, 491.000000)|(190.377686, 486.454590)|(0.877686, -4.545410)|(0.019947, -0.045004)|(44.000000, 101.000000)|(40.555389, 102.219482)|(0.921713, 1.012074)|

这是Fusion优势反例。IR框整体明显上移，中心dy=−25.088px，约为GT高度的−24.84%；Fusion为−4.545px，约−4.50%。IR最高分候选objectness约0.999094、IoU仅0.578786；Fusion最大IoU框objectness约0.145264，分数排名19，IoU为0.845997。几何最佳框不能以最高分框替代。

![04_Fusion_advantage_170100_G1](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/case_04_Fusion_advantage_170100_G1.png)

#### 05_Fusion_advantage_130126_G3

场景13；gt_index从0开始。选择原因：Fusion>=0.75>IR; descending Fusion-IR IoU。
GT xyxy=[436.0, 287.0, 474.0, 389.5]。

|模型|最大IoU|objectness|最佳proposal xyxy|分数排名/候选数|
|---|---:|---:|---|---|
|IR|0.664702535|0.999775231|[434.25647, 276.569824, 468.432068, 377.363647]|1/150|
|Fusion|0.846598208|0.997969091|[437.833832, 283.73526, 471.461639, 387.822662]|8/150|

|模型|GT中心|Proposal中心|dx,dy(px)|dx/GT宽,dy/GT高|GT宽高|Proposal宽高|宽/高比例|
|---|---|---|---|---|---|---|---|
|IR|(455.000000, 338.250000)|(451.344269, 326.966736)|(-3.655731, -11.283264)|(-0.096203, -0.110081)|(38.000000, 102.500000)|(34.175598, 100.793823)|(0.899358, 0.983354)|
|Fusion|(455.000000, 338.250000)|(454.647736, 335.778961)|(-0.352264, -2.471039)|(-0.009270, -0.024108)|(38.000000, 102.500000)|(33.627808, 104.087402)|(0.884942, 1.015487)|

IR框向上且略向左偏，中心dy=−11.283px，底边相对GT偏上约12.136px；Fusion中心dy=−2.471px、底边偏上约1.677px，垂直贴合更好。Fusion本例也偏窄（宽比例0.885），却因中心和上下边界更准确而取得更高IoU，说明不能只用单一宽比例判断框质量。

![05_Fusion_advantage_130126_G3](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/case_05_Fusion_advantage_130126_G3.png)

#### 06_boundary_170455_G1

场景17；gt_index从0开始。选择原因：unused GT; one side in [0.70,0.80]; crossing first, then min distance to 0.75。
GT xyxy=[266.0, 425.5, 317.0, 543.0]。

|模型|最大IoU|objectness|最佳proposal xyxy|分数排名/候选数|
|---|---:|---:|---|---|
|IR|0.750012934|0.996604443|[274.387024, 427.206909, 321.433899, 544.552124]|1/92|
|Fusion|0.674085140|0.646484852|[260.770538, 425.520813, 335.187653, 544.944824]|13/104|

|模型|GT中心|Proposal中心|dx,dy(px)|dx/GT宽,dy/GT高|GT宽高|Proposal宽高|宽/高比例|
|---|---|---|---|---|---|---|---|
|IR|(291.500000, 484.250000)|(297.910461, 485.879517)|(6.410461, 1.629517)|(0.125695, 0.013868)|(51.000000, 117.500000)|(47.046875, 117.345215)|(0.922488, 0.998683)|
|Fusion|(291.500000, 484.250000)|(297.979095, 485.232819)|(6.479095, 0.982819)|(0.127041, 0.008364)|(51.000000, 117.500000)|(74.417114, 119.424011)|(1.459159, 1.016375)|

临界IR IoU为0.750012934，仅高于0.75约1.2934×10⁻⁵；可视上仍存在约6.410px的向右中心偏移。Fusion中心向右偏6.479px，与IR接近，但宽比例1.459，明显宽于GT，IoU降至0.674085。跨过0.75阈值不代表几何误差突然消失。

![06_boundary_170455_G1](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/visual_review/case_06_boundary_170455_G1.png)

### 重复现象与机制边界

本次三个场景03的IR优势案例，Fusion最佳候选的宽比例分别约1.310、1.197、1.326，均比GT宽；临界案例为1.459。这里存在可重复的“部分Fusion候选偏宽”现象，但这六例经过优势差值选择，不能据此估计全验证集比例，更不能认定所有Fusion候选都偏宽。
两个Fusion优势反例中，IR候选都明显上移，Fusion的垂直位置更接近GT。Fusion六例高度比例均大于1，但其中四例仅高约1%–2%，不能把符号相同写成同等严重的系统误差。
个别高objectness候选定位较差，而低objectness候选最大IoU更高；这在两模型中都能观察到，不能据此只指责Fusion。分数不等于IoU，也不是最终person分类置信度。

### Day03建议：先查原RPN筛选与框回归

目前不建议直接修改融合模块。优先固定这六张val图片和原权重，记录解码后的候选及逐层pre-NMS、score过滤、NMS、post-NMS各步去留，按GT追踪最佳IoU和分数排名：若高IoU框在筛选前存在却被丢弃，继续检查排序/筛选；若解码前后始终缺乏高IoU框，再检查框回归和融合特征。先保持原预算不变，不用扩大Top-N制造改善。
这些步骤尚未执行。本次是RPN后处理个例的几何观察，尚不能把原因归到融合特征、RPN回归、score/NMS中的某一个，也不能直接解释全部历史test AP75差异。

完整数据：[选例表](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/selected_cases.csv)、[最佳候选坐标与objectness](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/proposal_details.csv)、[几何量](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/geometry_analysis.csv)、[全部实际后处理候选及分数](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/rpn_proposals.json)、[原自动中文报告](D:/python/test/vision-learning/week09/day02/outputs/proposal_visualization_run01/day02_report.md)。
