import cv2
import numpy as np

img = cv2.imread('./img/1.jpg')
# 上下翻转
# new_img = cv2.flip(img, 0)
# 左右翻转
# new_img = cv2.flip(img, 1)
# 左右上下翻转
new_img = cv2.flip(img, -1)

cv2.imshow("img", img)
cv2.imshow("new_img", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
