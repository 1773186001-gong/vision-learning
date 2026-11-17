# 项目背景

- 我是软件工程硕士研一，正在从基础学习计算机视觉和深度学习。
- 已完成 Week01–Week07，包括 CNN、ResNet18、迁移学习、模型评估、注意力机制、FPN/PAN、目标检测框、IoU 和检测损失。
- 接下来准备开始 Week08，逐步学习红外图像、红外可见光融合及小目标检测。

# 工作方式

- 全程使用中文解释。
- 每次只推进一个明确步骤，等待我反馈后再继续。
- 优先提供完整可运行的代码、运行命令、预期输出和结果解释。
- 不主动安排测验。
- 不随意删除、重构或覆盖已经完成的实验代码。
- 修改代码前先阅读相关文件，尽量保持已有功能不变。
- 不编造实验结果、模型指标或论文内容。
- 编写科研周报时，结合实际代码、实验结果和论文阅读，语言自然，避免空泛总结。

# 本地开发环境

- 操作系统：Windows。
- 项目目录：`D:\python\test\vision-learning`
- 本地 Python：`D:\python\.venv\Scripts\python.exe`
- 本地显卡：AMD Radeon RX 6650 XT 8GB。
- 已完成的实验主要位于 `week01` 至 `week07`。

# RTX 3060 训练服务器

- 服务器为 Windows 10 笔记本。
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6GB 显存。
- 主机名：`DESKTOP-BFIO0RA`
- SSH 用户名：`gxbb`
- 当前局域网 IP：`192.168.0.149`，可能随网络变化。
- 服务器项目：`D:\vision-learning-server`
- 服务器虚拟环境：`D:\cv-server-env`
- 服务器 Python：`D:\cv-server-env\Scripts\python.exe`
- PyTorch：`2.14.1+cu126`
- torchvision：`0.29.1+cu126`
- CUDA 已测试可用。
- PyCharm Professional 已配置 SFTP 同步。
- 服务器有 `train-manager.ps1` 和 `train-monitor.ps1`，支持后台训练、查看状态、日志和停止任务。
- 不要未经我确认就连接服务器、启动训练、停止进程或修改服务器环境。
- 不要使用项目中同步过去的 `.venv` 作为服务器解释器。

# 安全与执行约束

- 默认先检查、解释，再修改。
- 不要擅自安装依赖或升级环境。
- 不要未经确认执行大规模训练。
- 不要删除已有数据、日志和模型权重。
- 不要将密码、密钥或其他凭据写入项目文件。
