# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，用于数值计算
import numpy as np

# 定义一个函数 drawShape，用于在图像上绘制近似轮廓的边界线
def drawShape(img, approx):
    i = 0
    # 遍历所有近似轮廓点
    while i < len(approx):
        # 如果是最后一个点，则将其与第一个点连接，形成闭合图形
        if i == len(approx)-1:
            x, y = approx[i][0]
            x1, y1 = approx[0][0]
            cv2.line(img, (x, y), (x1, y1), (0, 255, 0), 1)
            break
        else:
            # 否则将当前点与下一个点连接
            x, y = approx[i][0]
            x1, y1 = approx[i+1][0]
            cv2.line(img, (x, y), (x1, y1), (0, 255, 0), 1)
        i += 1

# 读取图像文件 './img/hello.jpeg'，返回一个NumPy数组（即图像数据）
img = cv2.imread('./img/hello.jpeg')

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

# 绘制所有检测到的轮廓，颜色为绿色，线宽为1
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

# 计算轮廓的最小外接旋转矩形
r = cv2.minAreaRect(contours[1])
# 获取旋转矩形的四个顶点坐标
box = cv2.boxPoints(r)
# 将浮点型坐标转换为整数类型
box = np.int32(box)
# 绘制最小外接矩形，颜色为红色，线宽为2
cv2.drawContours(img, [box], 0, (0,0, 255), 2)

# 最大外接矩形
# 计算第二个轮廓的边界矩形（直立矩形）
x,y,w,h = cv2.boundingRect(contours[1])
# 绘制边界矩形，颜色为蓝色，线宽为2
cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 2)

# 显示处理后的图像
cv2.imshow('img', img)
# 等待用户按键（0表示无限等待）
cv2.waitKey(0)
