import cv2
import numpy as np

img = cv2.imread(r"./img/3.png")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 自适应二值化
dst = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 0)

cv2.imshow("img", img)
cv2.imshow("img1", img1)
cv2.imshow("dst", dst)  # 显示二值化图像
cv2.waitKey(0)
cv2.destroyAllWindows()
