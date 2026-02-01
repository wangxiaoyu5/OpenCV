import cv2
import numpy as np

img = cv2.imread(r"./img/flower.png")


mean_img = cv2.pyrMeanShiftFiltering(img, 20, 30)

img_canny = cv2.Canny(mean_img, 50, 150)

contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)








cv2.imshow("img_canny", img_canny)
cv2.imshow("img", img)
cv2.imshow("mean_img", mean_img)
cv2.waitKey(0)




















