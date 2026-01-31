import cv2
import numpy as np

# 读取图像文件，路径为 "./img/chess.png"
img = cv2.imread(r"./img/chess.png")

# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris角点检测（当前被注释掉）
# dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# blockSize: 邻域窗口大小；ksize: Sobel算子孔径大小；k: Harris检测器自由参数

# Shi-Tomasi角点检测
# maxCorners: 最大角点数量；qualityLevel: 角点质量阈值；minDistance: 角点间最小距离
conrnet = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)

# 将角点坐标转换为整数类型
conrnet = np.int32(conrnet)

# 角点展示（当前被注释掉）
# img[dst > 0.01 * dst.max()] = [0, 0, 255]

# 绘制Shi-Tomasi检测到的角点
for i in conrnet:
    x, y = i.ravel()  # 展平角点坐标
    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)  # 在图像上绘制紫色圆圈标记角点

# 显示图像并等待按键关闭窗口
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
