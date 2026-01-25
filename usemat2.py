import cv2
import numpy as np


img = cv2.imread(r"./img/login.png")
# 高度 长度 通道数
print(img.shape)
# 占用空间 = 高度 * 长度 * 通道数
print(img.size)
# 位深
print(img.dtype)

img[0:100, 0:100] = [255, 0, 0]
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()