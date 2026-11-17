experiment = {
    "model": "YOLO11",
    "dataset": "infrared_dataset",
    "image_size": 640,
    "batch_size": 8,
    "epochs": 100
}

print("模型：", experiment["model"])
print("数据集：", experiment["dataset"])
print("图片尺寸：", experiment["image_size"])
print("Batch Size：", experiment["batch_size"])
print("训练轮数：", experiment["epochs"])

print("新的 Batch Size：", experiment["batch_size"])