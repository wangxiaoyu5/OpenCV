import numpy as np
import cv2

# 开
# img = cv2.imread('./img/dotj.png')
# 闭
img = cv2.imread('./img/dotinj.png')
# 梯度
# img = cv2.imread('./img/tophat.png')

# kernel = np.ones((5,5),np.uint8)

# 卷积核
# MORPH_RECT 矩形
# MORPH_CROSS 十字形
# MORPH_ELLIPSE 椭圆
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
# 腐蚀
# dst = cv2.erode(img,kernel,iterations = 1)
# 膨胀
# dst = cv2.dilate(img,kernel,iterations = 1)

# 开操作 先腐蚀后膨胀 去除大图形外的小图形
# dst = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# 闭操作 先膨胀后腐蚀 去除大图形里的小图形
# dst = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
# 礼帽 原图减开操作 得到大图形外的小图形
# dst = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
# 黑帽 原图减闭操作 得到大图形里的小图形
# dst = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
# 形态学梯度 原图减腐蚀操作 求边缘
dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('dst', dst)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
