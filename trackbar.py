import cv2
import numpy as np

# 定义一个空的回调函数，用于 `createTrackbar` 的参数
def callback():
    pass

# 创建一个名为 "trackbar" 的窗口，支持调整大小
cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)

# 创建三个滑动条，分别控制 B、G、R 通道的值（0-255）
cv2.createTrackbar("R", "trackbar", 0, 255, callback)
cv2.createTrackbar("G", "trackbar", 0, 255, callback)
cv2.createTrackbar("B", "trackbar", 0, 255, callback)

# 创建一个黑色的图像（480x640，3通道），初始为全黑
img = np.zeros((480, 640, 3), np.uint8)

# 主循环：持续读取滑动条值并更新图像颜色
while True:
    # 获取当前滑动条的值
    r = cv2.getTrackbarPos("R", "trackbar")
    g = cv2.getTrackbarPos("G", "trackbar")
    b = cv2.getTrackbarPos("B", "trackbar")

    # 将图像所有像素设置为当前 RGB 值（注意：OpenCV 使用 BGR 格式）
    img[:] = [b, g, r]

    # 显示图像在窗口中
    cv2.imshow("trackbar", img)

    # 等待按键，若按下 'q' 键则退出循环
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

# 销毁所有 OpenCV 窗口
cv2.destroyAllWindows()
