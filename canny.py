import cv2  # 导入OpenCV库，用于计算机视觉和图像处理
import numpy as np  # 导入NumPy库，用于数值计算和数组操作


img = cv2.imread(r"./img/1.jpg")  # 读取指定路径下的图像文件，存储到img变量中

#  canny算子 图片 最小值 最大值
dst = cv2.Canny(img, 150, 200)



cv2.imshow("img", img)  # 显示原始图像，窗口标题为"img"
cv2.imshow("dst", dst)  # 显示滤波后图像，窗口标题为"dst"
cv2.waitKey(0)  # 等待按键事件，0表示无限等待直到有按键输入
cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
