import cv2
import numpy as np

# 读取图像文件
img = cv2.imread(r"./img/flower.png")

# 使用均值漂移滤波对图像进行平滑处理，参数分别为空间窗口半径和色彩窗口半径
mean_img = cv2.pyrMeanShiftFiltering(img, 20, 30)

# 对平滑后的图像进行Canny边缘检测，参数为低阈值和高阈值
img_canny = cv2.Canny(mean_img, 50, 150)

# 查找图像中的轮廓，返回轮廓列表和层级结构
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制所有找到的轮廓，颜色为绿色，线宽为3
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# 显示Canny边缘检测结果
cv2.imshow("img_canny", img_canny)
# 显示绘制了轮廓的原图
cv2.imshow("img", img)
# 显示均值漂移滤波后的图像
cv2.imshow("mean_img", mean_img)
# 等待按键事件，0表示无限等待
cv2.waitKey(0)
