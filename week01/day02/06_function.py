def check_map(map50):

    print("当前 mAP@0.5：", map50) 

    if map50 >= 0.8:
        print("效果很好")
    elif map50 >= 0.7:
        print("效果还可以")
    else:
        print("还有较大的提升空间")


check_map(0.85)
check_map(0.75)
check_map(0.62)