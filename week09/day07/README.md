# Week09 Day07：NMS前高分重叠候选竞争

本轮已完成唯一指标T的离线检验。23个“L相同而NMS结局不同”的GT中，失败侧T更高22项、相同1项、反向0项，提供了与预先固定方向一致的描述性支持。不是显著性检验、因果证明或新的模型实验。

## 范围与定义

读取根目录AGENTS.md、Day05/Day06 README和独立验收JSON、Day06完整297项及31项配对表、Day04来源账本与280份原始捕获。已用shell检索研究记忆；记忆滞后，实验事实以产物为准。Day07原先不存在。本轮只新增Day07文件。

固定母集297个共同风险GT，涉及140张val图片、594项模型×GT。主比较由真实配对表筛得L相同且结局不同的23项，与31项不对称表交叉核对，没有手填名单。

只使用score_filtered全部候选定义竞争者：好框a对GT的CPU float32 IoU≥0.75；b须同实际层、对该GT IoU<0.75、objectness严格更高，且相互IoU严格>0.7。C(a)数这样的不同ID候选，T取全部好框C的最小值。相同坐标不同ID分别计数；重复真实ID报错。空好框集合报错，不设为0。

原NMS保留集合没有参与指标定义。T计算完才核对原NMS丢失状态。实际层键0、1、pool通过原展平索引、counts、层标识及目录核对，没有按大小猜层。不达标仅针对这个GT，不等于背景。

## 主配对结果

ΔT=失败T−保住T，正值符合假设，0无法区分，负值为反向。

|范围|GT分母|正向|相同|反向|
|---|---:|---:|---:|---:|
|全部|23|22|1|0|
|IR失败|9|9|0|0|
|Fusion失败|14|13|1|0|
|场景03|7|6|1|0|
|场景08|7|7|0|0|
|场景13|4|4|0|0|
|场景17|5|5|0|0|

22/23（95.65%）符合方向，两模型失败时的方向均以正向为主。全部名单及T见outputs/competition_run01/paired_23.csv和report.md，未过滤相同或反向结果。

场景03的Fusion失败6项中5项正向、1项相同；IR失败1项正向。唯一相同项030078/G0，Fusion失败、IR保住，但双方T=1。T在这一个GT上不能区分结局。其余场景本次均为正向，小分母仅4至7项，不能宣称稳定的场景差异。

## 全母集背景

直接列实际T值，不作事后分箱。分母是各模型该T值的共同风险GT数，不能平均逐图比例。

|模型|T|风险GT|NMS丢失GT|
|---|---:|---:|---:|
|IR|0|269|0|
|IR|1|24|17|
|IR|2|3|3|
|IR|4|1|1|
|Fusion|0|260|0|
|Fusion|1|27|21|
|Fusion|2|3|3|
|Fusion|3|5|4|
|Fusion|4|1|1|
|Fusion|5|1|1|

各模型分别合计297GT、丢失21/30项，与Day05共同风险状态逐项一致。全部场景的计数及条件率见model_scene_T_summary.csv，含小分母，不省略不利结果。

本样本T=0项未出现丢失。这是观测，不能把T=0作为通用存活保证，尤其本指标不计同分竞争。T>0也不是必然失败：例如IR的T=1组24项只丢失17项；030129/G0的IR存活侧T=1，说明面对直接竞争仍可保住。指标不模拟逐次NMS，因此不能判断竞争框是否先被其他框删掉。

## 核验、审计及限制

- 人工数据13项检查通过，包含同层、严格分数、同分、0.75/0.7边界、多好框取最小、无竞争者、相同坐标不同ID、重复ID拒绝、空好框报错、失败方向与配对计数。它们是单元核验，不是LLVIP结果。
- 主运算采用TorchVision CPU float32 IoU。另用独立标量面积公式，每一步舍入binary32，从原始坐标精确核对全部候选GT IoU、逐好框关系、C、T、实际层及丢失状态。没有重跑NMS。
- 独立的verify_exports.py再按导出的关系重建C、按好框重建T，核对23项来源、方向、模型/场景汇总及历史SHA。它不替代主运行中对原始坐标的独立几何核验。
- 4046条好候选×GT记录、2047条严格竞争关系；不是不同物理候选总数。同分重叠关系0条，精度审计0条，主与float64判定分歧0项。
- 1e-6仅用于审计阈值附近，不改变主阈值。GT阈值审计遍历全部NMS前候选；两框阈值审计覆盖主好框面对的同层、主不达标、分数不低候选。诊断从不扩大主竞争集合。敏感项若出现会原样保留。
- 1559个历史文件SHA前后相同。来源账本、配对表、原始捕获SHA及实际命令、CPU环境、脚本SHA保存在run_metadata.json及capture_provenance.json。

直观理解：T寻找一个GT所有好框中“直接高分竞争最少”的那个。多数失败侧连这个最少值都比存活侧高，这有助于描述L相同之后仍存在的候选竞争差异。但NMS规则本就依赖分数和几何重叠，本观察与规则存在联系；不能当成独立的融合根因证据，也不能把统计关系命名为原GPU实际抑制配对。

没有额外检验目标大小、分数间隔等主指标。没有显著性或因果检验，不能据此直接改网络。未解释实际抑制链、相同T如何产生不同结局、候选和分数如何形成，以及RoI和最终AP75差异。

## 新增代码、产物和实际命令

新增analyze_competition.py（指标、人工检查、原始坐标独立核验、导出与报告）、verify_exports.py（导出独立审计）及本README。未修改历史文件。主运行输出目录为outputs/competition_run01：

- per_gt_metrics.csv：594项T、L、丢失状态、最小值对应ID。
- good_candidates.csv：4046项好候选ID、真实层、分数、C、同分数量及最小值标记。
- competition_relations.csv、equal_score_relations.csv：严格竞争与同分关系，坐标、分数、GT/相互IoU和真实索引。
- precision_audit.csv：独立诊断敏感项，本次为空。
- paired_23.csv、paired_summary.csv、model_scene_T_summary.csv：完整配对与实际T值背景。
- verification.json、capture_provenance.json、run_metadata.json、run.log、report.md：核验、来源SHA、命令、环境、退出码和完整中文表格。

工作目录D:\python\test\vision-learning，实际执行：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day07/analyze_competition.py --tests-only --output week09/day07/checks/synthetic_run01
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day07/analyze_competition.py --output week09/day07/outputs/competition_run01
& D:\python\.venv\Scripts\python.exe -B -X utf8 week09/day07/verify_exports.py --input week09/day07/outputs/competition_run01 --output week09/day07/checks/export_verification_run01
~~~

三项退出码均为0，日志全部保留。本次无失败运行；程序遇错会保存failure.json且不自动重跑，已有目录拒绝覆盖。主运行环境版本见run_metadata.json。

本轮没有连接服务器、加载checkpoint、新模型前向、NMS执行、训练、调参、test或安装升级依赖。离线检验及交付完成后停止，等待反馈。
