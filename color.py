import cv2
# 导入 OpenCV 库，用于图像处理和显示

def callback(x):
    pass
# 定义滑动条的回调函数，接收一个参数 x（当前滑动条值），但不执行任何操作

cv2.namedWindow("color", cv2.WINDOW_NORMAL)
# 创建一个名为 "color" 的窗口，支持调整大小

img = cv2.imread(r"./img/login.png")
# 读取本地图片文件 login.png，返回为 BGR 格式的图像矩阵

color_spaces = [
    cv2.COLOR_BGR2BGRA,
    cv2.COLOR_BGR2BGRA,  # 重复项，应为不同颜色空间
    cv2.COLOR_BGR2HSV,
    cv2.COLOR_BGR2YUV
]
# 定义一组颜色空间转换代码，用于后续调用 cv2.cvtColor

cv2.createTrackbar('curclolr', "color", 0, len(color_spaces), callback)
# 创建滑动条，名称为 'curclolr'，关联到 "color" 窗口，范围 0~len(color_spaces)-1，使用 callback 函数

while True:
    cur_color = cv2.getTrackbarPos('curclolr', "color")
    # 获取当前滑动条位置，对应 color_spaces 中的索引

    img = cv2.cvtColor(img, color_spaces[cur_color])
    # 将当前图像转换为目标颜色空间（注意：会改变通道数）

    cv2.imshow("color", img)
    # 在窗口中显示转换后的图像

    key = cv2.waitKey(10)
    # 等待 10ms 键盘输入

    if key & 0xFF == ord('q'):
        break
    # 检查是否按下 'q' 键，是则退出循环

cv2.destroyAllWindows()
# 销毁所有 OpenCV 窗口，释放资源
