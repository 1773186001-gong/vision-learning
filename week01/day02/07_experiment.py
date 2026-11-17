experiment = {
    "model": "YOLO11",
    "dataset": "IR-UAV",
    "image_size": 640,
    "batch_size": 16,
    "epochs": 100
}

images = [
    "001.jpg",
    "002.jpg",
    "003.jpg",
    "004.jpg",
    "005.jpg"
]


def show_experiment():

    print("========== 实验配置 ==========")
    print("模型：", experiment["model"])
    print("数据集：", experiment["dataset"])
    print("图片尺寸：", experiment["image_size"])
    print("Batch Size：", experiment["batch_size"])
    print("Epoch：", experiment["epochs"])


def process_images():

    print("========== 数据处理 ==========")
    print("图片总数：", len(images))

    for image in images:
        print("正在处理：", image)


show_experiment()

print()

process_images()