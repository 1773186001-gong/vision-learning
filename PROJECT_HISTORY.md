# 项目历史交接文档

## 一、项目总体情况

本人为软件工程硕士研一，目前正在学习计算机视觉、深度学习、目标检测和相关论文。

- 本地项目路径：`D:\python\test\vision-learning`
- 本地 Python：`D:\python\.venv\Scripts\python.exe`
- 当前进度：已完成 Week01–Week07，后续准备进入 Week08。
- 主要研究兴趣：图像处理、特征提取、注意力机制、红外与可见光多模态融合、小目标检测等。

## 二、Week01–Week02：图像处理与 CNN 基础

### 学习内容

- 图像读取、尺寸调整、归一化。
- PyTorch Tensor 和图像数据格式。
- 卷积神经网络的基本结构。
- 猫狗图像二分类。

### 实验记录与认识

早期使用小规模猫狗数据集，训练集和测试集各 40 张图片。

基础 CNN 曾出现训练准确率达到 97.5%、测试准确率只有 52.5% 的情况。通过该实验初步认识到过拟合问题，以及训练准确率与测试准确率之间的区别。

## 三、Week03：ResNet18 与迁移学习

### 学习内容

- ResNet18 网络结构。
- 残差连接。
- 预训练权重。
- 冻结参数与微调。
- 单张图像预测。

### 实验记录

`week03/day01/train_resnet.py` 实现的是基于 MNIST 的简化残差网络：包含自定义残差块和两层残差结构，并非 torchvision 提供的标准 ResNet18。Week03 后续的 `day03` 至 `day06` 实验才涉及标准 ResNet18 的数据增强、预训练权重、迁移学习与微调。

以下三轮训练指标为历史运行记录；本次只读检查未重新运行代码或验证其数值：

- Loss：0.1715
- Train Accuracy：96.52%
- Test Accuracy：96.04%

迁移学习实验中，使用预训练权重的模型曾达到 82.5% 测试准确率。也尝试过只训练全连接层和微调不同网络层，发现不同策略的结果存在差异。

## 四、Week04：模型评价指标

### 学习内容

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 一次实验结果

- Accuracy：0.825
- Precision：0.760
- Recall：0.950
- F1：0.8444
- 混淆矩阵：`[[14,6],[1,19]]`

重点理解了不同评价指标的意义，尤其是误分类、漏检和类别不平衡对评价结果的影响。

## 五、Week05：特征图与模型微调

### 学习内容

- CNN 特征图可视化。
- 卷积层参数变化。
- 自定义 CNN 与 ResNet18 参数量比较。
- 冻结主干网络与微调 Layer4。

### 实验结果

自定义 CNN：

- 总参数量：93,954
- 一次测试准确率：47.5%
- 混淆矩阵：`[[7,13],[8,12]]`

ResNet18：

- 总参数量：11,177,538
- 仅训练 FC 时可训练参数：1,026
- 微调 Layer4+FC 时可训练参数：8,394,754

Layer4+FC 微调实验：

- 训练 5 轮。
- 训练准确率达到 100%。
- 测试准确率达到 90%，即 36/40。

还分析了错误预测样本，例如将猫识别为狗且预测概率很高的情况。由此认识到模型输出的预测概率并不等于模型整体准确率。

## 六、Week06：注意力机制与特征金字塔

### 学习内容

- Channel Attention
- SE Attention
- CBAM
- Coordinate Attention
- FPN
- PAN

### 实验记录

实验中使用形状为 `(1,64,56,56)` 的特征张量。

CBAM：

- 通道注意力权重形状：`(1,64,1,1)`
- 空间注意力权重形状：`(1,1,56,56)`

FPN 输入：

- C2：`(1,64,56,56)`
- C3：`(1,128,28,28)`
- C4：`(1,256,14,14)`
- C5：`(1,512,7,7)`

FPN 输出 P2–P5 使用 128 通道，实验参数量为 713,728。PAN 实验参数量为 1,599,232。

重点理解了多尺度特征融合的作用，为后续目标检测学习打基础。

## 七、Week07：目标检测基础

### 学习内容

- Bounding Box
- `xyxy` 和 `xywh` 坐标格式。
- 归一化坐标。
- IoU。
- Backbone、Neck、Detection Head。
- 分类损失与定位损失。

### 实验记录

Day01：

- 图像尺寸：320×240。
- 目标框：`(50,40,180,160)`。
- 面积：15,600。
- 保存了 `bbox_demo.png`。

Day02：

- 两个框的 IoU：0.517241。

Day03：

- 输入：`(1,3,224,224)`。
- P2 分类输出：`(1,2,56,56)`。
- P2 边界框输出：`(1,4,56,56)`。

Day04：

- Classification Loss：0.201413。
- IoU：0.517241。
- 示例 Box Loss：0.482759。
- 示例 Total Loss：0.684172。

以上用于理解检测损失的简化演示，不是完整目标检测器的训练结果。

## 八、RTX 3060 训练服务器搭建

### 服务器信息

- 操作系统：Windows 10。
- 主机名：`DESKTOP-BFIO0RA`。
- 用户：`gxbb`。
- 当前局域网 IP：`192.168.0.149`。
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6GB。
- 服务器项目：`D:\vision-learning-server`

### 服务器环境

- 虚拟环境：`D:\cv-server-env`
- Python：`D:\cv-server-env\Scripts\python.exe`
- PyTorch：`2.14.1+cu126`
- torchvision：`0.29.1+cu126`
- CUDA 已验证可用。

### 已完成配置与验证

- OpenSSH Server 安装与启动。
- SSH 远程登录。
- Windows 防火墙端口配置。
- 笔记本合盖后短时间 SSH 连通性测试。
- PyCharm Professional SFTP Deployment。
- 项目文件与数据集上传。
- GPU 张量计算验证。
- 后台训练与日志保存。
- PowerShell 训练管理器。

训练管理器文件：

- `train-manager.ps1`
- `train-monitor.ps1`

支持操作：`start`、`status`、`log`、`stop`。

### 2026-10-29 最终验证

- Status：SUCCEEDED。
- Exit code：0。
- Started：14:30:40。
- Finished：14:30:48。
- Monitor PID：5908。

训练结果：

- Epoch 1：Loss 0.7465，Train Acc 47.50%，Test Acc 80.00%。
- Epoch 2：Loss 0.5208，Train Acc 70.00%，Test Acc 87.50%。
- Epoch 3：Loss 0.4354，Train Acc 85.00%，Test Acc 90.00%。

相关产物：

- 日志：`D:\vision-learning-server\logs\train_demo_20261029_143043.log`
- 模型：`D:\vision-learning-server\checkpoints\resnet18_demo_20261029_143043.pth`

该验证实验仅基于 40 张训练图片和 40 张测试图片，用于服务器环境验证，不是正式科研实验。

## 九、当前进度与下一步

### 当前状态

- Week01–Week07 已完成。
- RTX 3060 服务器环境已经搭建。
- PyCharm SFTP 同步正常。
- 服务器后台训练管理器已经验证成功。

### 下一阶段计划

1. 开始 Week08，学习红外图像和红外可见光多模态基础。
2. 结合现有 PyTorch 知识完成可运行的小实验。
3. 逐步学习红外小目标检测与多尺度特征融合。
4. 选择相关论文进行阅读、复现和实验分析。
5. 根据真实实验记录撰写研究生科研周报。

## 十、Week08 Day01–Day03 实际完成状态（2026-10-30更新）

- Day01：完成LLVIP官方数据检查及场景隔离的1,000对子集，train/val/test为700/150/150；保留VOC XML，自定义PyTorch Dataset，不使用YOLO。
- Day02：完成640×640 Letterbox、RGB/IR同步水平翻转与RTX 3060数据管线实测。此前代码和结果保留。
- Day03：在RTX 3060 Laptop 6GB上完成IR单模态Faster R-CNN冒烟测试、10轮训练、验证集选模及一次最终测试，终端退出码0；本次无OOM或训练中断。
- 模型为`fasterrcnn_mobilenet_v3_large_320_fpn`，官方COCO预训练、背景/person两类；IR复制为3通道，输入640×640，内部resize也设640；batch=1、workers=0、seed=42、SGD lr=0.001、momentum=0.9、weight_decay=0.0005，无AMP及学习率调度。
- 最佳第9轮：val P=0.884319、R=0.826923、AP50=0.873139、mAP50:95=0.431061。
- 最终test：P=0.850416、R=0.882184、AP50=0.898907、mAP50:95=0.471626；TP/FP/FN=307/54/41。使用自定义单person类101点AP协议，不是官方COCO API完整指标。
- 逐轮训练及验证累计1324.373885秒，约22分04秒，不含checkpoint落盘、绘图和最终测试；逐轮记录的最大peak_allocated为375.313 MiB、peak_reserved为476 MiB，非整卡总占用。
- 用户授权后已通过现有PyCharm SFTP取回best/last、日志、CSV、10份验证JSON、测试缓存、曲线及9张预测图。两checkpoint严格加载、历史与CSV及缓存指标一致性检查通过；未再次训练或运行测试推理。断点恢复已实现，但本次未实测中断后续训。

本地结果：`week08/day03/outputs/ir_baseline_run01/`。

完整启动日志：`week08/day03/outputs/ir_baseline_run01_launch_1793326678503699800.log`。

当前学习交接：[DAY03_IR_TRAINING_GPT_HANDOFF.md](week08/day03/DAY03_IR_TRAINING_GPT_HANDOFF.md)。原准备文档归档为`week08/day03/DAY03_IR_PREPARATION_ARCHIVE.md`；原始服务器报告保留在结果目录。

当前下一步为向ChatGPT提交交接文档，理解四项Loss、验证选模、AP及真实预测案例。尚未进行RGB、RGB+IR或后续实验；不自动开始新的训练。


## 十一、Week08 Day04 RGB基线完成（2026-10-30）

- RTX 3060 Laptop完成10轮RGB Faster R-CNN训练、val选模、一次test和9张案例；后台SUCCEEDED、退出码0，无OOM或数值异常记录。模型/数据/优化器/评估沿用Day03条件，仅改模型输入为visible，从官方COCO预训练重新初始化。
- 最佳第4轮：val P=0.677184、R=0.670673、AP50=0.676349、mAP50:95=0.264635。
- test P=0.491776、R=0.859195、AP50=0.800126、mAP50:95=0.368304；TP/FP/FN=299/309/49。IR为307/54/41；RGB误检更多，不据单次结果推断融合效果。
- 训练+验证1341.709314秒，脚本1403.201208秒；peak allocated=395375104 bytes（377.059 MiB），peak reserved=486539264 bytes（464 MiB）。
- 用户用命令行下载后，本地严格加载best/last，核对10轮记录、150张缓存/348个GT、复算指标与40个历史保护哈希，全部通过；没有重跑训练或测试推理。
- 本地结果：`week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce/`。正式交接：`week08/day04/DAY04_RGB_TRAINING_GPT_HANDOFF.md`。早期准备版及取回中断记录保留。
- 服务器曾关闭再开机取回；原关机守候器已不再存活，本次重启后关机尚未确认。下一步仅进行ChatGPT原理学习与IR/RGB结果分析，不自动进入融合或新训练。
