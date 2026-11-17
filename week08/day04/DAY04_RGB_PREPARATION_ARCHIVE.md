# Week08 Day04：RGB基线交接（准备完成，GPU训练待授权）

## 当前真实状态

已阅读AGENTS、PROJECT_HISTORY、Day01/Day02预处理和交接、Day03模型/训练/评估/预测及真实结果。在`week08/day04`新增完整RGB流程，未修改Day01–Day03代码、IR结果或原数据。

**本次尚未连接服务器、上传代码或运行RGB CUDA冒烟/正式训练。没有RGB指标、best/last或训练曲线。** 历史IR最佳第9轮、test AP50=0.898907、mAP50:95=0.471626，只作为后续对照。

本地真实检查已经执行：

```powershell
& D:\python\.venv\Scripts\python.exe -B .\week08\day04\check_day04.py
```

退出码0，`status=passed`，解释器`D:\python\.venv\Scripts\python.exe`、Torch `2.14.1+cpu`。结果：`checks/local_checks_1793334271617412000.json`。

## 本地真实检查结果

- 完整解码1,000张RGB和1,000张IR；1,000份XML可解析，与同名图像尺寸及清单一致。
- train/val/test=700/150/150，person框=1991/416/348，场景前缀不跨集合。
- 真实RGB：train/010001 `(3,640,640)`、框`(6,4)`；train/100030空样本框`(0,4)`；val/030001框`(2,4)`；test/190001框`(2,4)`。
- 各样本返回Tensor与同种子paired模式的visible逐元素一致，实际与其IR Tensor不同；没有把IR当成RGB。
- 三集合数据顺序/seed42采样与IR loader一致，train翻转p=.5，val/test p=0；batch=1、自定义collate和workers0正确。
- 仅构造模型结构、执行模型transform检查，两类头、640尺寸、框不再resize；没有模型前向、反向或参数更新。
- 四项Loss更新与冒烟函数和Day03是同一函数对象，评估直接使用原`ir_common.metrics`。
- 40个历史代码及IR原始产物SHA256检查一致，拒绝把训练输出指向IR目录。
- 案例分类/优先排序合成断言通过；这不是RGB检测指标。
- 后台PowerShell仅本地语法检查通过，尚未验证服务器WMI权限及脱离SSH会话后的持久性。

补充检查：`checks/safety_cli_1793334685398549900.json`，程序退出码0；训练/评估/预测/报告写向IR目录的4个CLI负例均正确拒绝，CPU正式训练也正确拒绝，没有创建训练输出。11项Day04运行文件SHA校验通过，40项历史保护仍一致。负例返回1或2为预期结果，不代表执行过GPU训练。三个PowerShell脚本均通过Parser语法检查。

## 实现及固定实验方案

代码清单与命令见[README](README.md)，一次性远程授权范围见[REMOTE_PLAN](REMOTE_PLAN.md)。主要新增`rgb_common.py`、`train_rgb.py`、`evaluate_rgb.py`、`predict_rgb.py`、`preflight_rgb.py`、`report_rgb.py`、`check_day04.py`及后台启动/监控脚本、40文件保护清单。

模型、COCO预训练、两类头、640 Letterbox和内部resize、训练p=.5翻转、10轮、batch1、workers0、seed42、SGD(.001,.9,.0005)、无AMP/调度、val mAP选模与一次test协议沿用IR；只换模型输入为visible RGB。没有从IR微调checkpoint初始化。

预计服务器输出：`D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01`，存在旧目录则选下一个全新run。预期本地对应目录相同；目前尚未创建真实RGB训练结果目录。

## 尚待实测，不能填入历史IR数据

服务器当前连通性、CUDA/依赖/缓存、数据与保护哈希、电源状态、真实RGB前向/四项Loss/反向/更新/显存、10轮训练记录、best epoch、test全部指标、耗时、图片分类、后台持久性、退出码及RGB–IR差值均待本次授权后实测。

当前远程授权仍未获得，必须按AGENTS及本次用户要求先一次性列清单等待确认。后续若获准、环境正确且冒烟通过，脚本会连续完成完整流程，不逐轮询问。异常立即停止，不无限重试、改参或覆盖历史结果。

当前仅可据此学习设计与实现，不能称RGB基线实验已完成。成功后服务器将在该run内生成含真实10轮表格、环境、best/test、耗时/显存、案例和IR对照的同名报告；安全取回后再整理本地最终交接。请ChatGPT全程中文、一次一个明确步骤，不主动测验，不把代码通过当成已掌握。
