import cv2
import numpy as np

img = cv2.imread(r"./img/1.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 解包返回值，并使用灰度图像
ret, binary_img = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("img", img)
cv2.imshow("img1", img1)
cv2.imshow("dst", binary_img)  # 显示二值化图像
cv2.waitKey(0)
cv2.destroyAllWindows()
