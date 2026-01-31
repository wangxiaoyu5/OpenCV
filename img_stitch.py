import cv2
import numpy as np

# 第一步，读取文件，将图片设置成一样大小640x480
# 第二步，找特征点，描述子，计算单应性矩阵
# 第三步，根据单应性矩阵对图像进行变换，然后平移
# 第四步，拼接并输出最终结果

# 读取两张图片
img1 = cv2.imread(r"./img/map1.png")
img2 = cv2.imread(r"./img/map2.png")
# 将两张图片设置成同样大小
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

inputs = np.hstack((img1, img2))

cv2.imshow("inputs", inputs)
cv2.waitKey(0)
cv2.destroyAllWindows()
