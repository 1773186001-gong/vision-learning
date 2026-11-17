model = "YOLO11"

images = [
    "001.jpg",
    "002.jpg",
    "003.jpg"
]

experiment = {
    "model": model,
    "image_size": 640,
    "batch_size": 16
}


if len(images) > 0:
    print("数据集共有", len(images), "张图片")


for image in images:
    print("正在处理：", image)


def show_experiment():
    print("模型：", experiment["model"])
    print("图片尺寸：", experiment["image_size"])
    print("Batch Size：", experiment["batch_size"])


show_experiment()