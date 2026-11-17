
import torch
import torch.nn.functional as F


def box_iou_xyxy(box_a, box_b):
    """计算单个 xyxy 边界框的 IoU。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


if __name__ == "__main__":
    # 假设类别编号：0=猫，1=狗
    true_class = torch.tensor([0])

    # 两个类别的原始分数（logits）
    predicted_logits = torch.tensor([[2.0, 0.5]])

    # Cross Entropy 会根据正确类别计算分类损失
    cls_loss = F.cross_entropy(predicted_logits, true_class)

    # 真实边界框与预测边界框，使用 xyxy 格式
    true_box = (50.0, 40.0, 180.0, 160.0)
    predicted_box = (80.0, 60.0, 200.0, 150.0)

    iou = box_iou_xyxy(true_box, predicted_box)

    # 这里用 1-IoU 演示定位误差的概念
    # 它不是完整检测器实际训练时使用的通用 Box Loss
    box_loss = 1.0 - iou

    total_loss = cls_loss.item() + box_loss

    print("========== Detection Loss Demo ==========")
    print(f"True class: {true_class.item()}")
    print(f"Predicted logits: {predicted_logits.tolist()}")
    print(f"Classification loss: {cls_loss.item():.6f}")
    print(f"Box IoU: {iou:.6f}")
    print(f"Illustrative box loss (1-IoU): {box_loss:.6f}")
    print(f"Illustrative total loss: {total_loss:.6f}")
