
import cv2
import numpy as np

# 导入图片
dog = cv2.imread('./img/1.jpg')

# 创建logo
logo = np.zeros((200, 200, 3), np.uint8)
mask  = np.zeros((200, 200), np.uint8)

# 绘制logo
logo[20:120, 20:120] = [0,0,255]
logo[80:180, 80:180] = [0,255,0]
mask[20:120, 20:120] = 255
mask[80:180, 80:180] = 255
# mask 取反
m =  cv2.bitwise_not( mask)
# 选择dog 添加logo的位置
roi = dog[0:200, 0:200]
# 与m 进行与运算
tmp = cv2.bitwise_and(roi, roi, mask=m)

dst = cv2.add(tmp, logo)

dog[0:200, 0:200] =  dst
cv2.imshow("dog", dog)
# cv2.imshow("dst", dst)
# cv2.imshow("tmp", tmp)
# cv2.imshow("logo", logo)
# cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()








