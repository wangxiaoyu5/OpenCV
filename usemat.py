import cv2
import numpy as np


img = cv2.imread(r"./img/login.png")
# 浅拷贝
img2 = img
# 深拷贝
img3 = img.copy()

img[0:100, 0:100] = [255, 0, 0]
cv2.imshow("img", img)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()