# Day03 IR 训练准备记录（历史归档，非当前状态）

当前已完成本地完整流程实现和工程检查，未获得新一轮服务器SSH/上传/远程训练授权，尚未开始正式IR训练。不得将以下计划或合成断言视为实验指标。

模型为 fasterrcnn_mobilenet_v3_large_320_fpn + COCO预训练 + 两类预测头；输入IR三通道、640 Letterbox，内部min/max也设640。SGD lr0.001、momentum0.9、wd0.0005、batch1、workers0、seed42、10轮，无AMP/调度。验证mAP50:95选best，测试仅最终一次。

代码及复现命令、指标协议、续训和异常停止策略详见README.md。代码入口：ir_common.py、train_ir.py、evaluate_ir.py、predict_ir.py；preflight_ir.py为只读环境检查，check_day03.py为本地工程验证。

本地发现torch=2.14.1+cpu，matplotlib/Pillow可用，torchmetrics/pycocotools未装。采用不新增依赖的单类101点AP，不声称完全等同官方COCO评估。

预训练权重本地和服务器均已确认未缓存。服务器环境真实输出为torch=2.14.1+cu126、torchvision=0.29.1+cu126、CUDA=True、NVIDIA GeForce RTX 3060 Laptop GPU。Pillow可用，matplotlib/torchmetrics/pycocotools未安装。权重下载须另获许可，不得随机初始化训练。

已将绘图与训练分离：服务器每轮保存真实CSV，缺少matplotlib时明确记录CURVE DEFERRED；最终取回后使用新增plot_training.py在本地已有matplotlib的环境生成loss_curve.png。预测图仍由Pillow生成，无需在服务器安装任何新依赖。

本地真实检查：val/030001输入IR=(3,640,640)，模型transform保持(640,640)，框不变；train/100030空框=(0,4)。合成指标测试包含完美匹配、漏检、重复预测和空标注，均通过，但不是检测成绩。

以下均待实测：每轮四项Loss/总Loss、验证P/R/AP、best epoch、测试指标、显存/耗时和代表预测案例。本地没有生成best.pth或last.pth，未伪造权重。

完整训练成功后，服务器输出目录会自动生成同名真实结果交接Markdown，再获得下载许可后将日志/CSV/指标/图片/权重取回，整理为最终本地交接。

当前下一步：授权只读检查服务器环境、子集、代码和权重缓存；依检查结果确认上传、必要的权重下载和训练命令。完整基线仍是本任务目标，当前只是因远程授权边界暂留在准备阶段。
