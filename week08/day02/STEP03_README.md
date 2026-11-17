# Day02 Step3：DataLoader 批处理与设备传输

当前状态：本地 CPU 和服务器 RTX 3060 GPU 真实验证均已通过，GPU结果已获准取回本地。最新记录见 [DAY02_STEP03_GPU_GPT_HANDOFF.md](DAY02_STEP03_GPU_GPT_HANDOFF.md)。

新增 `experiment_batch_device.py`，复用 Day01 的 LLVIPDataset、detection_collate_fn，以及 Step2 的 PairedLetterboxAndFlip。
完整学习记录和远程准备方案见 [DAY02_STEP03_GPT_HANDOFF.md](DAY02_STEP03_GPT_HANDOFF.md)。

实际执行的本地命令：

```powershell
Set-Location D:\python\test\vision-learning
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day02\experiment_batch_device.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day02\outputs\step03_batch_device_cpu `
  --device cpu `
  --seed 42
```

本次输出为 `outputs/step03_batch_device_cpu/terminal_output.txt` 与 `results.json`。
重跑请改用新的输出目录；不覆盖既有日志。

脚本只在当前机器执行，不会建立 SSH/SFTP 连接。必须显式选 cpu 或 cuda:0；CUDA 模式不回退到 CPU，并要求服务器解释器 `D:\cv-server-env\Scripts\python.exe` 和 RTX 3060。

CUDA 分支已在服务器实际运行，train/val/test 全部通过，终端退出码为0。结果位于 `outputs/step03_batch_device_gpu/`；远程操作仍须按用户要求逐阶段获准。原CPU交接文档保留当时“GPU待验证”的历史状态，以最新GPU交接文档为准。
