import cv2  # 导入 OpenCV 库，用于图像处理和计算机视觉任务
import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作

# 读取两张图像文件
img1 = cv2.imread(r"./img/opencv_search.png")  # 读取查询图像
img2 = cv2.imread(r"./img/opencv_orig.png")  # 读取目标图像
# 参数说明：
# filename: 图像文件路径，字符串类型。

# 将彩色图像转换为灰度图像
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 查询图像灰度化
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 目标图像灰度化
# 参数说明：
# src: 输入图像（通常是多通道图像）。
# code: 颜色空间转换标志，例如 `cv2.COLOR_BGR2GRAY`。

# 创建 SIFT 特征检测器对象
sift = cv2.SIFT_create()  # 初始化 SIFT 算法，用于提取图像的关键点和描述符
# 参数说明：
# 无参数，返回一个 SIFT 特征检测器对象。

# 检测关键点并计算描述符
kp1, des1 = sift.detectAndCompute(g1, None)  # 查询图像的关键点和描述符
kp2, des2 = sift.detectAndCompute(g2, None)  # 目标图像的关键点和描述符
# 参数说明：
# image: 输入图像（灰度图）。
# mask: 可选掩码，用于指定感兴趣区域。

# 配置 FLANN 匹配器的索引参数
index_params = dict(algorithm=1, trees=5)  # 使用 KD 树算法，设置树的数量为 5
search_params = dict(checks=50)  # 设置搜索过程中遍历的叶节点数量为 50
# 参数说明：
# index_params: 索引构建参数字典，例如 `{algorithm: 1, trees: 5}`。
# search_params: 搜索参数字典，例如 `{checks: 50}`。

# 创建 FLANN 匹配器对象
flann = cv2.FlannBasedMatcher(index_params, search_params)  # 初始化 FLANN 匹配器
# 参数说明：
# index_params: 索引构建参数字典。
# search_params: 搜索参数字典。

# 使用 KNN 方法进行特征匹配（第一次调用，结果未保存）
flann.knnMatch(des1, des2, k=2)  # 对描述符进行 KNN 匹配，k=2 表示每个特征点找两个最近邻
# 参数说明：
# queryDescriptors: 查询图像的描述符。
# trainDescriptors: 目标图像的描述符。
# k: 每个特征点查找的最近邻数量。

# 再次使用 KNN 方法进行特征匹配，并保存结果
matches = flann.knnMatch(des1, des2, k=2)  # 重新执行匹配，结果存储在 matches 中
# 参数说明：
# queryDescriptors: 查询图像的描述符。
# trainDescriptors: 目标图像的描述符。
# k: 每个特征点查找的最近邻数量。

# 筛选出优质匹配点
good = []  # 存储筛选后的匹配点
for i, (m, n) in enumerate(matches):  # 遍历所有匹配点对
    if m.distance < 0.7 * n.distance:  # 应用 Lowe's ratio test 过滤低质量匹配
        good.append(m)  # 符合条件的匹配点加入 good 列表
# 参数说明：
# matches: 匹配点对列表，每个元素包含两个最近邻的距离信息。

# 检查匹配点数量是否足够
if len(good) < 4:
    print("Not enough matches are found - %d/%d" % (len(good), 4))
    exit()
# 参数说明：
# good: 筛选后的优质匹配点列表。
# 4: 最少需要的匹配点数量，用于后续计算单应性矩阵。

# 提取匹配点的坐标
srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
# 参数说明：
# kp1, kp2: 查询图像和目标图像的关键点列表。
# m.queryIdx, m.trainIdx: 匹配点在各自关键点列表中的索引。
# reshape(-1, 1, 2): 将坐标重塑为 OpenCV 所需的格式。

# 计算单应性矩阵
H, status = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
# 参数说明：
# srcPts: 查询图像中匹配点的坐标。
# dstPts: 目标图像中匹配点的坐标。
# method: 用于估计单应性矩阵的方法，例如 `cv2.RANSAC`。
# ransacReprojThreshold: RANSAC 算法的最大重投影误差阈值。

# 获取目标图像的尺寸
h, w = img2.shape[:2]
# 参数说明：
# img2: 目标图像。
# shape[:2]: 图像的高度和宽度。

# 定义目标图像的四个角点
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# 参数说明：
# [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]: 图像四个角点的坐标。
# reshape(-1, 1, 2): 将坐标重塑为 OpenCV 所需的格式。

# 对角点进行透视变换
dst = cv2.perspectiveTransform(pts, H)
# 参数说明：
# pts: 输入的点坐标。
# H: 单应性矩阵。

# 在目标图像上绘制变换后的边界框
cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)
# 参数说明：
# img2: 目标图像。
# [np.int32(dst)]: 变换后的点坐标，转换为整数类型。
# isClosed: 是否闭合多边形，`True` 表示闭合。
# color: 边界框颜色，例如 [(255, 0, 0)](file://D:\PyCharmwork\tuxiangxuexi\add2.py#L5-L5) 表示蓝色。
# thickness: 边界框线条粗细。

# 绘制匹配结果
ret = cv2.drawMatches(img1, kp1, img2, kp2, good, None)  # 将匹配点绘制在图像上
# 参数说明：
# img1, img2: 两张输入图像。
# keypoints1, keypoints2: 两幅图像的关键点列表。
# matches: 匹配点对列表。
# outImg: 输出图像（可为 `None`）。
# flags: 绘制选项，例如 `cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS`。


# 显示匹配结果
cv2.imshow("img", ret)  # 显示绘图结果
cv2.waitKey(0)  # 等待用户按键关闭窗口
# 参数说明：
# winname: 窗口名称。
# mat: 要显示的图像矩阵。
# delay: 等待时间（毫秒），`0` 表示无限等待。
