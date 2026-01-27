import cv2  # 导入OpenCV库，用于计算机视觉和图像处理
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

img = cv2.imread(r"./img/login.png")  # 读取指定路径下的图像文件，存储到img变量中

# 定义2x3的仿射变换矩阵M，其中前两列代表旋转缩放，第三列代表平移
# M = np.float32([[1,0,100],[0,1,0]])   # 变换矩阵：[[1,0,100],[0,1,0]]表示向右平移100像素
# 旋转角度逆时针
# 中心的为（x,y）
# M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 24, 0.3)

# 定义源图像上的三个点坐标，用于计算仿射变换矩阵
src = np.float32([[400, 300], [800, 300], [400, 1000]])
# 定义目标图像上对应的三个点坐标，与源图像的点一一对应
dst = np.float32([[200, 400], [600, 500], [150, 1100]])
# 计算仿射变换矩阵，根据源点和目标点的对应关系生成2x3变换矩阵
M = cv2.getAffineTransform(src, dst)

# 执行仿射变换：使用变换矩阵M对原图进行变换，输出尺寸与原图相同
new_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

cv2.imshow("img", img)  # 显示原始图像，窗口标题为"img"
cv2.imshow("new_img", new_img)  # 显示经过仿射变换后的图像，窗口标题为"new_img"
cv2.waitKey(0)  # 等待按键事件，0表示无限等待直到有按键输入
cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
