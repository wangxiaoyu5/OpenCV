import cv2          # 导入OpenCV库，用于图像处理和计算机视觉功能
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

curshape = 0        # 全局变量，用于存储当前绘制的形状类型：0=直线，1=矩形，2=圆形
startpos = (0,0)    # 全局变量，存储鼠标点击的起始位置坐标

def callback(event, x, y, flags, param, ):  # 定义鼠标事件回调函数
    global curshape  # 声明使用全局变量curshape
    global startpos  # 声明使用全局变量startpos
    if event == cv2.EVENT_LBUTTONDOWN:       # 当鼠标左键被按下时
        startpos = (x, y)                    # 记录鼠标按下时的坐标作为起始位置
    elif event == cv2.EVENT_LBUTTONUP:       # 当鼠标左键被释放时
        endpos = (x, y)                      # 记录鼠标释放时的坐标作为结束位置
        if curshape == 0:                    # 如果当前形状为直线
            cv2.line(img, startpos, endpos, (255, 255, 255), 5, 4)  # 在图像上绘制白色直线
            # 参数：目标图像，起始点，结束点，颜色(BGR格式白色)，线宽，线型
        elif curshape == 1:                  # 如果当前形状为矩形
            cv2.rectangle(img, startpos, endpos, (255, 255, 255), -1)  # 绘制白色填充矩形
            # 参数：目标图像，左上角坐标，右下角坐标，颜色(BGR格式白色)，线宽(-1表示填充)
        elif curshape == 2:                  # 如果当前形状为圆形
            a = x - startpos[0]              # 计算x方向的距离差
            b = y - startpos[1]              # 计算y方向的距离差
            r = int ((a * a + b * b) ** 0.5) # 计算半径，使用勾股定理并转换为整数
            cv2.circle(img, startpos, r, (255, 255, 255), -1)  # 绘制白色填充圆形
            # 参数：目标图像，圆心坐标，半径，颜色(BGR格式白色)，线宽(-1表示填充)

cv2.namedWindow("drawshape", cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口，命名为"drawshape"

cv2.setMouseCallback("drawshape", callback)      # 为窗口设置鼠标事件回调函数

img = np.zeros((480, 640, 3), np.uint8)          # 创建一个黑色背景的图像，尺寸为480x640，3个颜色通道

while True:                                      # 主循环，持续运行
    cv2.imshow("drawshape", img)                 # 在窗口中显示当前图像
    key = cv2.waitKey(10) & 0xFF                 # 等待按键输入，每次循环等待10毫秒

    if key == ord('q'):                          # 如果按下'q'键
        break                                    # 退出循环
    elif key == ord('l'):                        # 如果按下'l'键，选择绘制直线
        curshape = 0                             # 设置当前形状为直线(0)
    elif key == ord('r'):                        # 如果按下'r'键，选择绘制矩形
        curshape = 1                             # 设置当前形状为矩形(1)
    elif key == ord('c'):                        # 如果按下'c'键，选择绘制圆形
        curshape = 2                             # 设置当前形状为圆形(2)

cv2.destroyAllWindows()                          # 销毁所有OpenCV创建的窗口
