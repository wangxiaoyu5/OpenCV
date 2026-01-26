
import cv2
import numpy as np

# 读取图像文件 './img/login.png' 并存储到变量 img 中
img = cv2.imread('./img/login.png')

# 打印图像的形状信息（高度、宽度、通道数）
print(img.shape)

# 创建一个与目标尺寸相同的全1数组，然后乘以100创建常数矩阵
img2 = np.ones((997, 1912, 3), np.uint8) * 100

# 显示原始图像，窗口标题为 "img"
cv2.imshow("img", img)

# 对两张图像进行像素级加法运算（注意：这里可能会因尺寸不匹配出错）
# 两张图片大小需一样
result = cv2.add(img, img2)

# 显示加法运算后的结果图像
cv2.imshow("result", result)
# 对两张图像进行像素级减法运算（注意：这里可能会因尺寸不匹配出错）
sub = cv2.subtract(img, img2)
cv2.imshow("sub", sub)

# 对两张图像进行像素级乘法运算（注意：这里可能会因尺寸不匹配出错）
mul = cv2.multiply(img, img2)
cv2.imshow("mul", mul)

# 对两张图像进行像素级除法运算（注意：这里可能会因尺寸不匹配出错）
div = cv2.divide(img, img2)
cv2.imshow("div", div)

# 等待键盘按键事件（参数0表示无限期等待）
cv2.waitKey(0)

# 销毁所有OpenCV创建的窗口
cv2.destroyAllWindows()