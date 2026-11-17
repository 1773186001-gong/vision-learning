import cv2

# =========================
# 1. 读取图片
# =========================
image = cv2.imread("test.png")

# 检查图片是否读取成功
if image is None:
    print("图片读取失败，请检查 test.png 是否和 day07.py 在同一个目录")
    exit()

print("图片读取成功")
print("图片尺寸：", image.shape)


# =========================
# 2. 灰度化
# =========================
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# =========================
# 3. 二值化
# =========================
_, binary = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY
)


# =========================
# 4. 创建 Kernel
# =========================
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (9, 9)
)


# =========================
# 5. 开运算
# 去除一些小噪声
# =========================
clean = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)


# =========================
# 6. 查找轮廓
# =========================
contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("检测到的轮廓数量：", len(contours))


# =========================
# 7. 复制原图
# =========================
result = image.copy()


# =========================
# 8. 遍历所有轮廓
# =========================
object_count = 0

for contour in contours:

    # 计算面积
    area = cv2.contourArea(contour)

    # 过滤小轮廓
    if area < 500:
        continue

    # 外接矩形
    x, y, w, h = cv2.boundingRect(contour)

    object_count += 1

    # 画矩形
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # 在框上显示编号
    cv2.putText(
        result,
        f"Object {object_count}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(
        f"目标 {object_count}: "
        f"面积={area:.1f}, "
        f"x={x}, y={y}, "
        f"宽={w}, 高={h}"
    )


print("最终保留目标数量：", object_count)


# =========================
# 9. 显示结果
# =========================
cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Binary", binary)
cv2.imshow("Clean", clean)
cv2.imshow("Result", result)


# =========================
# 10. 等待按键
# =========================
cv2.waitKey(0)
cv2.destroyAllWindows()