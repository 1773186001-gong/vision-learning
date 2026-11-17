
from pathlib import Path
from PIL import Image, ImageDraw

# 1. 创建输出目录
save_dir = Path(__file__).parent
save_dir.mkdir(parents=True, exist_ok=True)

# 2. 创建一张模拟图片：宽320像素，高240像素
image_width = 320
image_height = 240

image = Image.new(
    "RGB",
    (image_width, image_height),
    color=(235, 240, 245)
)

draw = ImageDraw.Draw(image)

# 3. 定义目标框：x_min, y_min, x_max, y_max
box = (50, 40, 180, 160)

x_min, y_min, x_max, y_max = box

# 4. 绘制目标框
draw.rectangle(
    box,
    outline=(255, 0, 0),
    width=3
)

# 5. 在框附近标记类别
draw.text(
    (x_min, y_min - 18),
    "target",
    fill=(255, 0, 0)
)

# 6. 计算目标框的宽、高和面积
box_width = x_max - x_min
box_height = y_max - y_min
box_area = box_width * box_height

# 7. 将图片保存到当前脚本所在目录
save_path = save_dir / "bbox_demo.png"
image.save(save_path)

# 8. 输出检测框信息
print("========== Bounding Box Demo ==========")
print(f"Image size: {image_width} x {image_height}")
print(f"Bounding box (xyxy): {box}")
print(f"Box width: {box_width}")
print(f"Box height: {box_height}")
print(f"Box area: {box_area}")
print(f"Image saved to: {save_path}")
