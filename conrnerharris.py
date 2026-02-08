# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，用于数值计算
import numpy as np
# 导入pytesseract库，用于OCR文字识别（虽然在此代码中未使用）
import pytesseract

# 读取图像文件 'img/chessboard.jpg'，返回一个numpy数组表示的图像
img = cv2.imread('img/chessboard.jpg')
# 将彩色图像转换为灰度图像，减少计算复杂度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Harris角点检测算法检测图像中的角点
# 参数说明：
# - gray: 输入的灰度图像
# - blockSize: 角点检测中考虑的邻域大小，默认为2
# - ksize: Sobel算子的孔径大小，默认为3
# - k: Harris检测器的自由参数，默认为0.04
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 将检测到的角点标记为红色（BGR值为[0, 0, 255]）
# 条件是角点响应值大于最大响应值的1%
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# 显示处理后的图像
cv2.imshow('img', img)
# 等待用户按键，0表示无限等待
cv2.waitKey(0)
