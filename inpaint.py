# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，用于数值计算和数组操作
import numpy as np

# 读取原始图像文件，路径为"./img/inpaint.png"
img = cv2.imread(r"./img/inpaint.png")
# 读取掩码图像文件，路径为"./img/inpaint_mask.png"，以灰度模式加载（0表示灰度）
mask = cv2.imread(r"./img/inpaint_mask.png", 0)

# 使用Telea算法对图像进行修复，修复半径为5
dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
# 使用Navier-Stokes算法对图像进行修复，修复半径为5
dst2 = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

# 显示使用Telea算法修复后的图像，窗口名为"dst"
cv2.imshow("dst", dst)
# 显示使用Navier-Stokes算法修复后的图像，窗口名为"dst2"
cv2.imshow("dst2", dst2)
# 等待用户按键，0表示无限等待
cv2.waitKey(0)
# 关闭所有OpenCV创建的窗口
cv2.destroyAllWindows()
