import cv2
import numpy as np


# img = cv2.imread(r"./img/chess.png")
img1 = cv2.imread(r"./img/opencv_search.png")
img2 = cv2.imread(r"./img/opencv_orig.png")
# 灰度化
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



# 创建SIFT对象
# sift = cv2.SIFT_create()
# sift = cv2.xfeatures2d.SIFT_create()
#  创建SURF对象
# suft = cv2.xfeatures2d.SURF_create()

#  创建SURF对象
orb = cv2.ORB_create()
# 进行SIFT特征检测
# kp = sift.detect(gray, None)
# kp , des = sift.detectAndCompute(gray, None)

# 进行SURF特征检测
# kp , des = suft.detectAndCompute(gray, None)

# 进行ORB特征检测
# kp , des = orb.detectAndCompute(gray, None)
kp1, des1 = orb.detectAndCompute(g1, None)
kp2, des2 = orb.detectAndCompute(g2, None)
# 创建暴力匹配对象
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
match = bf.match(des1, des2)


#  绘制特征点
# img = cv2.drawKeypoints(gray, kp, img)

img = cv2.drawMatches(img1, kp1, img2, kp2, match, None)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()















