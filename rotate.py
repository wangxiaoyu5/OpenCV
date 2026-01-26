import cv2
import numpy as np

img = cv2.imread('./img/1.jpg')
# 旋转90 度
new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# 旋转180 度
new_img2 = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow("img", img)
cv2.imshow("new_img", new_img)
cv2.imshow("new_img2", new_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()












