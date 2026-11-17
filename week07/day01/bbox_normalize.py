
# week07/day01/bbox_normalize.py

# 图片尺寸
image_width = 320
image_height = 240

# 边界框格式：左上角 x、左上角 y、右下角 x、右下角 y
x_min, y_min, x_max, y_max = 50, 40, 180, 160

# 第一步：xyxy 转换为中心点 xywh
center_x = (x_min + x_max) / 2
center_y = (y_min + y_max) / 2
box_width = x_max - x_min
box_height = y_max - y_min

# 第二步：计算归一化坐标
norm_x = center_x / image_width
norm_y = center_y / image_height
norm_width = box_width / image_width
norm_height = box_height / image_height

print("========== Bounding Box Normalization ==========")
print(f"Image size: {image_width} x {image_height}")

print("\n[Pixel coordinates: xyxy]")
print(f"x_min={x_min}, y_min={y_min}, "
      f"x_max={x_max}, y_max={y_max}")

print("\n[Pixel coordinates: center xywh]")
print(f"center_x={center_x}, center_y={center_y}")
print(f"width={box_width}, height={box_height}")

print("\n[Normalized coordinates: xywh]")
print(f"x_center={norm_x:.6f}")
print(f"y_center={norm_y:.6f}")
print(f"width={norm_width:.6f}")
print(f"height={norm_height:.6f}")
