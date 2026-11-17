import cv2

# 1. 读取
image = cv2.imread("test.jpg")

# 2. 灰度
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# 3. 二值化
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# 4. 查找轮廓
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

# 5. 遍历轮廓
for contour in contours:

    # 面积
    area = cv2.contourArea(contour)

    # 过滤小轮廓
    if area < 500:
        continue

    # 外接矩形
    x, y, w, h = cv2.boundingRect(contour)

    # 画框
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # 输出信息
    print(
        "面积：", area,
        "位置：", x, y,
        "宽：", w,
        "高：", h
    )

cv2.imshow("Original", image)
cv2.imshow("Binary", binary)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()