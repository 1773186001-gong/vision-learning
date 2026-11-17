# Day04 结果取回中断记录

记录时间：2026-10-30 16:32，中国标准时间。

## 已有证据

- 用户已授权 Day04 检查、上传、后台训练、结果取回及训练成功并取回核验后的服务器关机。
- 用户启动后台任务：`rgb_baseline_run01_20261030_125726_0d9fec62`，监控 PID 5972；后来启动关机守候器，终端返回 PID 6628。
- 训练期间取回的原始快照保存在 `checks/startup_downloads_20261030/`。其中 progress.json 记录 epoch=6、batch=700；早期 state.json 为 RUNNING，不能作为最终成功或最终退出码证据。
- 之前的远程目录观察出现 epoch_10.pth、completed.json、exit_status.json、loss_curve.png、预测图目录和自动生成交接报告。此次连接刷新前也能看到这些文件名，但刷新后连接失败，因此此次界面列表可能是缓存，不能代替读取最终文件。

## 本次实际操作与失败

1. 通过已授权的 PyCharm SFTP 选中预测图、best.pth、completed.json、config.json、console 日志及结果目录内的自动报告，执行下载。
2. PyCharm 返回 `Server closed connection during identification exchange`，下载失败。
3. 刷新远程目录后仍出现同一错误。
4. 普通本地端口检查受执行环境网络权限限制，未获得远程状态。随后经过工具审批的只读 TCP 检查得到 `SSH_PORT_22_OPEN`，但读取的 SSH greeting 为空。端口开放不证明目标设备身份、SSH 登录正常或训练成功。
5. 本地 `outputs/rgb_baseline_run01` 尚无成功下载的最终产物；未运行依赖这些文件的 `review_rgb_results.py`，没有最终 RGB 指标或 checkpoint 核验结论。

## 关机状态

没有创建或上传 `ready_to_shutdown.json`，没有发送新的关机命令，未确认服务器关机。现有守候器的代码设置两小时等待期限；按先前启动时间推测可能已超时，但没有读取其最新状态文件，不能将超时当成已验证事实。

## 恢复连接后的下一步

- 先只读取回任务 state、shutdown_watcher_status 和最终结果，确认 SUCCEEDED、退出码 0 与实际产物。不要重跑训练、测试或启动重复任务。
- 下载 best/last、日志、CSV、全部验证 JSON、测试指标及缓存、图片和完成记录；逐轮 epoch 权重可留在服务器。
- 运行本地 review_rgb_results.py，严格加载两份 checkpoint，对照 CSV、缓存复算指标及历史保护 SHA。保存真实核验输出。
- 完成正式交接及 RGB/IR 对照；如果守候器仍有效，再按已有授权提交取回核验标记。若已超时，应先明确其实际状态，再选择获准且可用的关机执行方式，不能声称原守候器仍会自动关机。

当前主交接文档仍包含原准备阶段说明，尚不是最终训练报告。以本记录为当前状态补充；所有既有代码、原始数据与历史实验结果均保留。
