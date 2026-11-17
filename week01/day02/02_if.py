map50 = 0.76

print("当前 mAP@0.5：", map50)

if map50 >= 0.8:
    print("模型效果很好")
elif map50 >= 0.7:
    print("模型效果一般，还有提升空间")
else:
    print("模型效果比较差，需要进一步优化")