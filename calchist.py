import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread(r"./img/cat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# 均衡化
equ = cv2.equalizeHist(gray)

# 自适应均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(gray)


plt.hist(equ.ravel(), 256)

plt.hist(gray.ravel(), 256, [0, 256])
plt.show()

cv2.imshow("img", img)
cv2.imshow("equ", equ)
cv2.imshow("cl1", cl1)
cv2.waitKey(0)




