
def calculate_area(box):
    """
    计算边界框面积。
    box 格式：(x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = box

    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)

    return width * height


def calculate_iou(box_a, box_b):
    """
    计算两个 xyxy 格式边界框的 IoU。
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 1. 计算交集区域的边界
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # 2. 计算交集面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 3. 分别计算两个框的面积
    area_a = calculate_area(box_a)
    area_b = calculate_area(box_b)

    # 4. 计算并集面积
    union_area = area_a + area_b - inter_area

    # 避免并集面积为 0 时出现除零错误
    if union_area == 0:
        return 0.0

    # 5. 返回 IoU
    return inter_area / union_area


if __name__ == "__main__":
    # 真实目标框
    box_a = (50, 40, 180, 160)

    # 模拟模型预测框
    box_b = (80, 60, 200, 150)

    area_a = calculate_area(box_a)
    area_b = calculate_area(box_b)

    iou = calculate_iou(box_a, box_b)

    print("========== IoU Calculation ==========")
    print(f"Ground truth box A: {box_a}")
    print(f"Predicted box B: {box_b}")
    print(f"Area of A: {area_a}")
    print(f"Area of B: {area_b}")
    print(f"IoU: {iou:.6f}")