import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取背景
# 1. 通过二值法得到黑白图片
# 2. 通过形态学获取背景

# 读取图像文件
img = cv2.imread('./img/water_coins.jpeg')
# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Otsu算法进行二值化处理，得到黑白图像
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 定义结构元素（卷积核）
kernel = np.ones((3,3), np.int8)
# 对二值图像进行开运算，去除噪声
open1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# 对开运算后的图像进行膨胀，填充前景区域
bg = cv2.dilate(open1, kernel, iterations = 1)

# 获取前景物体
# 计算距离变换，突出前景物体的中心区域
dist = cv2.distanceTransform(open1, cv2.DIST_L2, 5)

# 对距离变换结果进行阈值处理，得到前景区域
ret, fg = cv2.threshold(dist,0.7*dist.max(), 255, cv2.THRESH_BINARY)

# plt.imshow(dist, cmap='gray')
# plt.show()
# exit()

# 获取未知区域
# 将前景区域转换为uint8类型
fg = np.uint8(fg)
# 通过背景减去前景，得到未知区域
unknow = cv2.subtract(bg, fg)

# 创建连通域
# 对前景区域进行连通域分析，标记不同的前景物体
ret, marker = cv2.connectedComponents(fg)

# 将标记值加1，避免与背景冲突
marker = marker + 1
# 将未知区域标记为0
marker[unknow==255] = 0

# 进行图像分割
# 使用分水岭算法对图像进行分割
result = cv2.watershed(img, marker)

# 将分割边界标记为红色
img[result == -1] = [0, 0, 255]

# 显示处理结果
cv2.imshow("img", img)
cv2.imshow("unknow", unknow)
cv2.imshow("fg", fg)
cv2.imshow("bg", bg)
cv2.imshow("thresh", thresh)
cv2.waitKey()
