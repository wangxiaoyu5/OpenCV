import cv2  # 导入 OpenCV 库，用于图像处理和显示功能

# cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)  # 被注释掉的代码：创建一个自适应大小的窗口
cv2.namedWindow("image",cv2.WINDOW_NORMAL)  # 创建一个名为 "image" 的可调整大小的窗口
cv2.resizeWindow("image", 640, 480)  # 将窗口 "image" 的大小设置为 640x480 像素
cv2.imshow("image", 0)  # 在 "image" 窗口中显示数值 0（由于显示参数错误，这会报错）

key = cv2.waitKey(0)  # 等待键盘按键输入，参数 0 表示无限等待，返回按键的 ASCII 码值
if key == 113:  # 判断按键是否为 'q' 键（'q' 的 ASCII 码值为 113）
    exit()  # 如果按下 'q' 键则退出程序

cv2.destroyAllWindows()  # 销毁所有已创建的 OpenCV 窗口
