# Day04远程操作一次性授权清单

当前只有本地准备工作获准；**本次尚未建立远程连接或开始RGB训练**。历史Day03的“启动/保存”不是Day04授权。请求依据是项目AGENTS.md中的“不要未经我确认就连接服务器、启动训练、停止进程或修改服务器环境”，以及本次任务明确要求对连接、上传、运行和取回先确认。需要用户批准下面同一流程后才连续执行，不逐Epoch再次询问。

## 拟授权范围

1. 对`gxbb@192.168.0.149`尝试必要的SSH/SFTP连接，并以返回的hostname/IP核实是`DESKTOP-BFIO0RA`；之后允许为状态查询、持久性检查和取回建立必要的新连接。保留已有主机密钥校验，不绕过身份认证，不索取或保存密码。
2. 只读检查服务器解释器、Torch/TorchVision/CUDA/GPU、运行中的Python任务、LLVIP_1000的RGB/IR/XML/清单、Day03依赖和IR结果、COCO权重缓存、Day04是否已有文件以及电源/睡眠/合盖设置。
3. 仅将本地Day04新增代码、保护清单、说明文档上传到`D:\vision-learning-server\week08\day04`。只新增不存在的文件；若已经存在，先比对，同内容跳过，不同则停止，不覆盖。必要时先用唯一暂存目录上传并在确认目标不存在后移入Day04，保留暂存记录，不删除旧文件。不上传数据、.venv、IR权重、旧日志或完整LLVIP。
4. 在服务器用`D:\cv-server-env\Scripts\python.exe`执行完整预检、真实RGB CUDA冒烟，并在通过后**自动连续完成10轮训练、val选模、一次test、缓存出图、RGB/IR对照和真实报告**；不等待每轮反馈。固定Day03参数，错误立即停止并保留记录，不重试或调参。
5. 使用Day04独立后台启动器，允许WMI创建隐藏监控进程及其Python子进程、写入Day04新job/run目录的日志与状态，并在新的SSH会话检查PID和日志推进。不会创建系统计划任务、服务或修改已有训练管理器。如果无法核实持久运行，则明确报告，不能宣称已实现无人值守。
6. 允许`--keep-awake`临时调用Windows线程级保持唤醒请求，结束时释放；**不修改电源计划/睡眠/合盖配置**。该请求不能保证合盖、强制睡眠、关机或断电后继续运行。服务器应插电且盖子保持打开。
7. 成功或失败后，允许将本次新RGB run的日志、CSV、JSON、曲线、预测图、测试缓存、best.pth和last.pth以及对应job状态/日志取回本地Day04新目录，核对并完成本地交接及RGB/IR对照；不下载10份逐轮权重、不修改服务器原文件、不覆盖本地既有目录。授权还涵盖完成后必要的只读服务器哈希复核。

上述范围**不包括**安装/升级依赖、下载权重或数据、停止其他进程、修改系统配置、重训IR、增加RGB+IR融合实验或删除任何已有文件。若缺少依赖/权重/数据或遇到权限问题，停止并报告。

## 具体命令

首次连接检查目标身份（此处仅展示，尚未执行）：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 gxbb@192.168.0.149 hostname
```

通过身份核对后，把本地只读检查脚本文本送入远端PowerShell执行，不在远端保存此初查脚本：

```powershell
Get-Content -LiteralPath .\week08\day04\REMOTE_READONLY_CHECK.ps1 -Raw | ssh -o BatchMode=yes -o ConnectTimeout=10 gxbb@192.168.0.149 "powershell.exe -NoProfile -NonInteractive -Command -"
```

仅在缺少的Day04文件安全上传、哈希核对和检查通过之后，执行后台完整流程：

```cmd
powershell.exe -NoProfile -File D:\vision-learning-server\week08\day04\start_rgb_background.ps1 -Action start -KeepAwake
```

此命令为未来**服务器端执行命令**，不是本地直接运行命令。实际远程调用会通过已授权SSH承载。启动器第一次应选择`D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01`；如果已有任何同名目录，选择下一个不存在的runNN并记录真实名称。

启动后必须保留打印的JobDir，用README中的status命令从新SSH会话核实监控器/Python仍存在、日志实际推进，再报告后台持续运行。完成必须核对退出码、completed、完整10轮历史、best/test标记、缓存指标和受保护IR哈希；启动命令返回0本身不等于训练完成。

## 现实限制和时间估计

上次命令行SSH曾返回`Permission denied (publickey,password,keyboard-interactive)`；PyCharm已保存的SFTP可用，但它不保证命令行SSH能自动执行。本次授权后可以尝试已有认证；若仍失败，不绕过认证，也不能用界面自动输入终端命令。需要用户在已登录服务器的CMD手动运行准备好的后台启动命令，或提供已经配置好的非交互SSH访问。未成功启动时必须明确记录，不能承诺今晚已经开跑。

IR同配置10轮训练+逐轮验证实测约22分04秒；RGB预计同量级，含检查/保存/出图可先预留30–45分钟，另加结果取回时间。这是根据历史推算，实际耗时必须记录。Codex关闭、网络中断或服务器睡眠会影响监看/取回；仅在服务器后台进程持久性验证通过后，其自动训练/评估/保存才可不依赖原SSH窗口继续。

用户可一次回复“授权清单全部操作”，也可以指定不允许的条目；之后只在超出范围或遇到必须用户处理的认证/权限问题时停下。
