import  cv2
import numpy as np


img = cv2.imread('./img/1.jpg')
# disze (x,y)
# new_img  = cv2.resize(img, (400, 400))

# INTER_NEAREST: 邻近插值算法，速度最快但效果最差
# 适用于对图像质量要求不高的场景，如快速预览

# INTER_LINEAR: 双线性插值算法，使用原图中的4个邻近点进行插值
# 效果优于INTER_NEAREST，是常用的平衡选择

# INTER_CUBIC: 三次插值算法，使用原图中的16个邻近点进行插值
# 效果好但计算量大，适合高质量图像处理

# INTER_AREA: 区域插值算法，效果最好
# 特别适合图像缩小操作，能有效减少锯齿和模糊现象
new_img  = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
print(img.shape)


cv2.imshow("img", img)
cv2.imshow("new_img", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()









