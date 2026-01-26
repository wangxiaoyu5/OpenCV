import cv2
import numpy as np

img = np.zeros((200, 200), np.uint8)
img2 = np.zeros((200, 200), np.uint8)

img[50:100, 50:100] = 255
img2[80:130, 80:130] = 255
# 非运算
new_img = cv2.bitwise_not(img)
# 与运算 取图片交集
new_img2 = cv2.bitwise_and(img, img2)
# 或运算 取图片并集
new_img3 = cv2.bitwise_or(img, img2)
# 异或运算 取图片异集
new_img4 = cv2.bitwise_xor(img, img2)


cv2.imshow("new_img4", new_img4)
cv2.imshow("new_img3", new_img3)
cv2.imshow("new_img2", new_img2)
cv2.imshow("new_img", new_img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
