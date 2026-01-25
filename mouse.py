import cv2
import numpy as np


# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    # 打印鼠标事件类型、坐标、标志位和参数
    print(event, x, y, flags, param)


# 创建一个名为 "mouse" 的窗口
cv2.namedWindow("mouse")

# 设置窗口大小为 640x480 像素
cv2.resizeWindow("mouse", 640, 480)

# 绑定鼠标事件回调函数到窗口
cv2.setMouseCallback("mouse", mouse_callback, param=None)

# 创建一个黑色的图像（3通道，尺寸 480x640）
img = np.zeros((480, 640, 3), np.uint8)

# 主循环：持续显示图像并监听键盘输入
while True:
    # 显示图像在窗口中
    cv2.imshow("mouse", img)

    # 等待按键，若按下 'q' 键则退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 销毁所有 OpenCV 窗口
cv2.destroyAllWindows()
