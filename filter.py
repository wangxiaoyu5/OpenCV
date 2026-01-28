import cv2  # 导入OpenCV库，用于计算机视觉和图像处理
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from numpy.ma.extras import dstack

img = cv2.imread(r"./img/login.png")  # 读取指定路径下的图像文件，存储到img变量中

# 创建5x5的均值滤波核，每个元素值为1/25=0.04，用于平滑图像
# kernel = np.ones((5,5), np.float32)/25

# 使用自定义卷积核对图像进行二维滤波操作
# 参数1：输入图像
# 参数2：输出图像深度，-1表示与输入图像深度相同
# 参数3：卷积核
# dst = cv2.filter2D(img, -1, kernel)
# 使用均值滤波对图像进行滤波操作
# dst = cv2.blur(img, (5, 5))

# 使用高斯滤波对图像进行滤波操作
# dst = cv2.GaussianBlur(img, (5, 5), 1)
# 使用中值滤波对图像进行滤波操作  呼叫噪音
# dst = cv2.medianBlur(img, 5)

# 使用双边滤波对图像进行滤波操作  美颜
# dst = cv2.bilateralFilter(img, 9, 75, 75)





cv2.imshow("img", img)  # 显示原始图像，窗口标题为"img"
cv2.imshow("dst", dst)  # 显示滤波后图像，窗口标题为"dst"
cv2.waitKey(0)  # 等待按键事件，0表示无限等待直到有按键输入
cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
