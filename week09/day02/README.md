# Week09 Day02：RPN Proposal 定位误差可视化

本步骤只对按 Day01 CSV 选中的少量 validation 图片做原模型只读前向，不训练、不使用 test、不修改 Week08、Day01、模型结构、checkpoint 或 RPN 预算。

## 实际输入与选例

复用 week09/day01/rpn_diagnostic.py 导出的原恢复方法、paired loader、坐标检查、模型状态校验和 transform/RPN hooks。IR 使用 ir_common.checkpoint_model()；Fusion 使用 fusion_model.restore()，输入保持 visible/infrared 字典。

从正式 GPU Day01 的 rpn_per_gt.csv 按 stem + **0 起始 gt_index** 配对，使用 max_iou_top300。已检查其与 max_iou_top1000 完全一致，都是原最多150个后处理候选。

固定选择顺序：

1. IR 优势3个：IR≥0.75>Fusion。场景03优先，再按 IR−Fusion IoU 降序。
2. Fusion 优势2个：Fusion≥0.75>IR，按 Fusion−IR IoU 降序。
3. 临界1个：排除已选GT；至少一侧在[0.70,0.80]，优先两侧跨越0.75，再按任一侧距0.75的最小距离升序。

平局按 stem、gt_index 升序，条件不足如实输出实际数量。按该规则共有6个不同图片的GT。推理顺序为选中图片原 image_index 升序，模型先IR后Fusion；不遍历其余验证图像做前向。

## 提取与几何

启动时检查当前 TorchVision RegionProposalNetwork.filter_proposals 源码。原方法返回 (final_boxes, final_scores)，scores 为 sigmoid(objectness)，已对应原逐层筛选、裁边、小框/分数过滤、NMS 和 post-NMS 截断后的框。临时实例包装只调用原绑定方法、复制返回张量并原样返回；在 finally 中移除hooks并恢复原方法查找状态。

以同次 transform 输出的 GT 和真实 RPN 输出框计算最大IoU，所选框及分数必须和 filter_proposals 捕获内容对应。最佳框按 IoU 最大选取；若并列，torch.argmax 取原输出中第一个。objectness排名为原输出的1起始位置，不是最终person分类置信度或IoU概率。

沿用同步640 Letterbox和原内部transform。在底部两格使用相同640×640红外底图、相同坐标和放大窗口。Fusion仍输入原配对RGB+IR，红外仅作为一致的展示底图。顶部为原始RGB/IR，不叠加640框，因此无原图逆变换。

- 绿色实线：GT；蓝色虚线：最大IoU的原RPN候选。
- dx=proposal中心x−GT中心x，正值向右；dy同理，正值向下。
- dx_norm=dx/GT宽，dy_norm=dy/GT高。
- width_ratio=proposal宽/GT宽，height_ratio同理；边界误差均为proposal−GT。

GPU结果逐案例与Day01比较，容差1e-6并检查0.75覆盖判定；若不一致保留失败日志并停止，不重新定义优势案例。CPU可用于排错，但不要求跨设备逐位一致。

## 运行

使用新的输出目录，路径存在即拒绝执行。下列是运行示例；实际执行状态只以输出文件和真实日志为准。

~~~powershell
& D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week09\day02\visualize_proposals.py --device cuda:0 --seed 42 --threads 2 --output D:\vision-learning-server\week09\day02\outputs\proposal_visualization_run01
~~~

本地静态检查：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B D:\python\test\vision-learning\week09\day02\visualize_proposals.py --help
~~~

## 输出与核验

- selected_cases.csv：精确选例、Day01 IoU、原因及原索引。
- proposal_details.csv：每案例每模型GT/最佳框xyxy、IoU、objectness、分数排名、候选数量及Day01一致性。
- geometry_analysis.csv：以上信息及中心、dx/dy、归一化偏移、宽高、比例和四边误差。
- case_*.png：所有实际选中案例，每例一张2×2图，含相同区域的局部放大。
- day02_report.md：中文数值观察、逐图链接和Day03建议。
- run.log、execution_status.json：真实过程及正常/异常结束状态。
- rpn_proposals.json：两模型所有选中图片的全部原后处理框、对应objectness和GT，供独立复算与后续读取。
- run_metadata.json：命令、环境、固定顺序、原RPN配置、checkpoint与模型state、Day01文件及选中输入的前后SHA256。

整个Day01目录、相关Week08源码、checkpoint和六张图片输入在推理前后校验，原配置与Day01逐项比较。下载后还需独立核验服务器文件SHA256、从原候选重算最佳IoU/索引/分数与几何量，并实际打开全部图片检查。

本步骤是按差值与阈值挑选的个例分析，不能估计整个验证集的偏差比例，也不能证明某个融合模块或RPN机制是原因。Day03优先追踪原候选筛选与框回归，在有机制证据前不直接修改融合模块。

## 只重绘已保存结果

初次GPU运行后目视检查发现部分放大图标题/图例拥挤，当前脚本已调整图例位置并把放大标签限制在窗内。原始GPU输出和源版本校验保持原样。可在本地使用下面命令，从保存的CSV重新绘图到新目录，不恢复模型、不执行forward：

~~~powershell
& D:\python\.venv\Scripts\python.exe -B D:\python\test\vision-learning\week09\day02\visualize_proposals.py --render-from D:\python\test\vision-learning\week09\day02\outputs\proposal_visualization_run01 --output D:\python\test\vision-learning\week09\day02\outputs\proposal_visualization_run01\visual_review
~~~

重绘记录见 visual_review/render_metadata.json；图像只显示原GPU保存的坐标与数值。此输出目录也必须不存在。真实运行状态仍以日志为准。
