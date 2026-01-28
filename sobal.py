import cv2  # 导入OpenCV库，用于计算机视觉和图像处理
import numpy as np  # 导入NumPy库，用于数值计算和数组操作


img = cv2.imread(r"./img/2.png")  # 读取指定路径下的图像文件，存储到img变量中

#  Sobel算子 y方向梯度 抗噪音强
# d1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# 沙尔 算子 3*3
# d1 = cv2.Scharr(img, cv2.CV_64F, 1, 0)
#  Sobel算子 x方向梯度
# d2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# 沙尔 算子
# d2 = cv2.Scharr(img, cv2.CV_64F, 0, 1)

#  拉普拉斯算子 对噪声敏感 一般先噪声处理
dst = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

# dst = d1 + d2



# cv2.imshow("d1", d1)
# cv2.imshow("d2", d2)
cv2.imshow("img", img)  # 显示原始图像，窗口标题为"img"
cv2.imshow("dst", dst)  # 显示滤波后图像，窗口标题为"dst"
cv2.waitKey(0)  # 等待按键事件，0表示无限等待直到有按键输入
cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
