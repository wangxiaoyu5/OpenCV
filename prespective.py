import cv2  # 导入OpenCV库，用于计算机视觉和图像处理
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

img = cv2.imread(r"./img/login.png")  # 读取指定路径下的图像文件，存储到img变量中

# 定义源图像上的四个角点坐标，用于计算透视变换矩阵
src = np.float32([[400, 300], [800, 300], [400, 1000], [800, 1000]])

# 定义目标图像上对应的四个角点坐标，与源图像的点一一对应
dst = np.float32([[200, 400], [600, 500], [150, 1100], [650, 1200]])

# 计算透视变换矩阵，根据源点和目标点的对应关系生成3x3变换矩阵
M = cv2.getPerspectiveTransform(src, dst)

# 执行透视变换：使用变换矩阵M对原图进行变换，输出尺寸为(2300, 3000)
new_img = cv2.warpPerspective(img, M, (2300, 3000))

cv2.imshow("img", img)  # 显示原始图像，窗口标题为"img"
cv2.imshow("new_img", new_img)  # 显示经过透视变换后的图像
cv2.waitKey(0)  # 等待按键事件，0表示无限等待直到有按键输入
cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
