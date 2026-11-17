# Day05 双分支融合交接：本地准备通过，正式训练待执行

# Day05：双分支特征融合 Faster R-CNN

当前：本地真实COCO加载、双分支FPN及完整检测器CPU评估接口通过；GPU冒烟、10轮训练、一次test和结果取回待执行。服务器SSH免密认证被拒绝，未改变认证方式。用户已授权Day05全流程及本地约78MB官方权重下载，密码仅由用户在自己的SSH提示符输入。

新正式任务采用独立RGB/IR MobileNetV3+FPN、逐尺度Concat+1×1Conv、共享RPN/RoI。DAY05_PLAN.md保留为早期方案历史文档（六通道输入融合），已由本README和实际双分支代码取代。内部六通道打包仅为了共用检测器几何transform，Backbone前拆成两路3通道；不是输入级六通道卷积。

## 已运行本地检查

```powershell
& D:\python\.venv\Scripts\python.exe -B .\week08\day05\check_day05.py --weights-path .\week08\day05\cache\checkpoints\fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth
```

结果见local_check_final.log和checks/local_*.json。check_day05只运行真实训练/验证/测试样本的数据预处理及Backbone/FPN，未执行检测器forward；另一次独立检查在真实train/010001执行了CPU检测器eval forward，boxes输出(46,4)，没有算检测指标、反向、更新或测试集推理，记录checks/full_interface_cpu.json。随机两类头的46个框不表示有效检测成绩。

实际结构：```json
{
  "feature_shapes": {
    "0": [
      1,
      256,
      20,
      20
    ],
    "1": [
      1,
      256,
      20,
      20
    ],
    "pool": [
      1,
      256,
      10,
      10
    ]
  },
  "roi_keys": [
    "0",
    "1"
  ],
  "anchor_sizes": [
    [
      32,
      64,
      128,
      256,
      512
    ],
    [
      32,
      64,
      128,
      256,
      512
    ],
    [
      32,
      64,
      128,
      256,
      512
    ]
  ],
  "anchor_ratios": [
    [
      0.5,
      1.0,
      2.0
    ],
    [
      0.5,
      1.0,
      2.0
    ],
    [
      0.5,
      1.0,
      2.0
    ]
  ],
  "out_channels": 256,
  "fusion_initialization": "[0.5 I, 0.5 I], zero bias",
  "pretrained_loaded": true,
  "source": "official COCO only"
}
```

总参数23739157，可训练参数23621365。两分支权重同源但存储独立、冻结范围一致；融合层[0.5I,0.5I]初始化，后续可训练。完整参数来源及requires_grad将在服务器model_metadata.json保存。

## 文件

fusion_model.py：显式paired输入接口、双FPN、动态特征键检查、共享检测器及strict恢复。
fusion_common.py：沿用Dataset/同步预处理/collate，双模态设备传输、四项Loss更新与缓存推理；复用原匹配、AP、checkpoint原子保存。
check_day05.py、preflight_fusion.py：真实COCO结构检查、全数据/环境/电源/保护检查。
train_fusion.py：自动检查→GPU正/空样本冒烟与状态恢复→10轮→val选模→一次test→缓存分析及报告；异常立即保留现场停止。
evaluate_fusion.py：独立val及受test_started保护的一次最终test。
predict_fusion.py、error_trace.py：从缓存出图、真实案例分类、三模型FP归因与报告；trace为Day04匹配追踪的独立原样副本。
start_fusion.ps1、monitor_fusion.ps1：从Day04已实测方案新建，WMI独立监控及唯一job/run，不改旧管理器。
deploy_and_start.ps1：用户本地命令行执行，先SSH只读检查再上传包，远端全量预检现有冲突后只新增文件，再启动后台。不会存密码、不安装依赖。
references：历史IR/RGB缓存与指标的只读副本，包清单校验；不上传旧best/last或完整数据。
protected_inputs.json：204项本地Day01–Day04成果快照。服务器检查实际存在的映射历史文件（RGB服务器仍叫rgb_baseline_run01）；缺少的非必需历史本地文件单独列出，不声称服务器核验了全部204项。原代码和两模态best/last/cache必须存在。

## 用户本地CMD的一次启动交接

```cmd
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File D:\python\test\vision-learning\week08\day05\deploy_and_start.ps1
```

仅在已开机、插电、盖子打开且没有其他训练时运行一次。SSH/scp可能多次提示密码；只在终端输入，不发聊天。遇到认证或文件冲突立即停止，不自动重试。若已经启动过，先读job状态，不重复运行该启动命令。

启动返回精确JobDir后，把输出交回本会话，通过独立连接检查state/stdout的后续推进，才能确认脱离原SSH后继续运行。目前尚未验证Day05后台持久性。不会安装后台服务或改电源策略；保持唤醒不能阻止合盖、强制睡眠或断电。

## 直接训练命令（服务器，当前尚未执行）

```cmd
D:\cv-server-env\Scripts\python.exe -u -B D:\vision-learning-server\week08\day05\train_fusion.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day05\outputs\fusion_concat_run01 --device cuda:0 --keep-awake
```

10轮、batch1、workers0、seed42、SGD(.001,.9,.0005)、640、无AMP/scheduler。已有输出会选独立后缀，后台启动器也选首个不存在的run。OOM、NaN/Inf均停止，不自动改参数或重试。

续训仅人工核对同run未启动test、epoch<10、无冲突快照/tmp后，在直接命令后添加 `--resume D:\vision-learning-server\week08\day05\outputs\实际run\last.pth`。保存optimizer、Python/Torch/CUDA/DataLoader RNG和history；本地只验证了模型state及冻结标记的strict恢复，没有实测中断训练后的续训。

独立val命令：

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day05\evaluate_fusion.py --checkpoint D:\vision-learning-server\week08\day05\outputs\实际run\best.pth --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day05\outputs\实际run\val_review.json
```

不要单独重跑test。正式流程在best确定后一次test，保存预测缓存；predict_fusion.render_and_report只消费缓存生成9张代表图（不足如实记录）及5组三模型对照，不再次模型预测。

## 已知限制与待完成

没有融合训练指标、GPU显存实测或最终成功结论。待远程认证交接、GPU冒烟、10轮、test、取回核验后才能完成正式报告。相同数据/超参数/协议不控制模型容量，不能将变化完全归因为模态融合；测试集已用于研究启发，只能报告探索性对照。未授权自动关机，完成后不追加其他架构/实验。
