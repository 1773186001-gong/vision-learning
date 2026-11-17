images = [
    "image001.jpg",
    "image002.png",
    "image003.jpg",
    "label001.txt",
    "image004.jpg"
]

for image in images:

    if image.endswith(".jpg"):
        print("发现图片：", image)