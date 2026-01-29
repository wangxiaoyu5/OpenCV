# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，用于数值计算
import numpy as np

# 读取图像文件 './img/5.png'，返回一个NumPy数组（即图像数据）
img = cv2.imread('./img/5.png')

# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行阈值处理，使用Otsu算法自动确定最优阈值
# THRESH_BINARY_INV表示反向二值化，THRESH_OTSU表示使用Otsu方法
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 查找图像中的轮廓
# 输入参数：
#   - thresh: 二值化后的图像
#   - cv2.RETR_TREE: 轮廓检索模式，返回完整的层级结构
#   - cv2.CHAIN_APPROX_SIMPLE: 轮廓近似方法，压缩水平、垂直和对角线方向的元素
# 返回值：
#   - contours: 检测到的轮廓列表
#   - hierarchy: 轮廓的层级信息
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 5)
# 打印检测到的所有轮廓信息
# print(contours)

# 计算面积
area = cv2.contourArea(contours[0])
print(area)


# 计算周长
len = cv2.arcLength(contours[0], True)
print(len)

# 显示原始图像 和处理后的图像
cv2.imshow('img', img)
cv2.imshow('thresh', thresh)

# 等待用户按键（0表示无限等待）
cv2.waitKey(0)
# 关闭所有OpenCV创建的窗口
cv2.destroyAllWindows()
