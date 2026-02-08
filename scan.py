# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
import pytesseract  # Tesseract OCR库，用于文字识别
from PIL import Image  # PIL库，用于图像操作

# 定义函数：order_points(pts)
# 功能：将四个点按顺序排列为左上、右上、右下、左下
def order_points(pts):
    # 创建一个4x2的零矩阵，用于存储排序后的坐标
    rect = np.zeros((4, 2), dtype="float32")

    # 计算左上角和右下角点
    s = pts.sum(axis=1)  # 对每个点的x和y坐标求和
    rect[0] = pts[np.argmin(s)]  # 左上角点：坐标和最小
    rect[2] = pts[np.argmax(s)]  # 右下角点：坐标和最大

    # 计算右上角和左下角点
    diff = np.diff(pts, axis=1)  # 计算每个点的x和y坐标差值
    rect[1] = pts[np.argmin(diff)]  # 右上角点：差值最小
    rect[3] = pts[np.argmax(diff)]  # 左下角点：差值最大

    return rect  # 返回排序后的坐标

# 定义函数：four_point_transform(image, pts)
# 功能：对图像进行四点透视变换
def four_point_transform(image, pts):
    # 获取输入坐标点并排序
    rect = order_points(pts)
    (tl, tr, br, bl) = rect  # 分别为左上、右上、右下、左下点

    # 计算变换后的宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # 底边长度
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # 顶边长度
    maxWidth = max(int(widthA), int(widthB))  # 最大宽度

    # 计算变换后的高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # 右侧边长度
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # 左侧边长度
    maxHeight = max(int(heightA), int(heightB))  # 最大高度

    # 定义目标坐标点（变换后的图像四个角点）
    dst = np.array([
        [0, 0],  # 左上角
        [maxWidth - 1, 0],  # 右上角
        [maxWidth - 1, maxHeight - 1],  # 右下角
        [0, maxHeight - 1]], dtype="float32")  # 左下角

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped  # 返回变换后的图像

# 定义函数：resize(image, width=None, height=None, inter=cv2.INTER_AREA)
# 功能：调整图像尺寸
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None  # 初始化目标尺寸
    (h, w) = image.shape[:2]  # 获取原始图像的高度和宽度

    # 如果未指定宽高，则返回原图
    if width is None and height is None:
        return image

    # 如果只指定了高度，则按比例计算宽度
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # 如果只指定了宽度，则按比例计算高度
        r = width / float(w)
        dim = (width, int(h * r))

    # 调整图像尺寸
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized  # 返回调整后的图像

# 主程序部分
# 读取图像文件
image = cv2.imread('./img/page.jpg')
ratio = image.shape[0] / 500.0  # 计算缩放比例
orig = image.copy()  # 保存原始图像副本

# 缩放图像并进行预处理
image = resize(orig, height=500)  # 将图像高度缩放到500像素
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊降噪
edged = cv2.Canny(blur, 75, 200)  # Canny边缘检测

# 查找轮廓
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # 按面积排序，取前5个最大的轮廓

# 遍历轮廓，寻找四边形
for c in contours:
    peri = cv2.arcLength(c, True)  # 计算轮廓周长
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 多边形逼近
    if len(approx) == 4:  # 如果是四边形
        screenCnt = approx  # 保存该轮廓
        break

# 绘制轮廓
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)  # 绿色绘制轮廓

# # 显示边缘检测结果和轮廓图
# cv2.imshow('edged', edged)
# cv2.imshow('image', image)


# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)  # 对原始图像进行透视变换

# 二值化处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]  # 二值化
cv2.imwrite('scan.jpg', ref)  # 保存扫描结果

# 显示原始图像和扫描结果
cv2.imshow("Original", resize(orig, height=650))  # 显示原始图像
cv2.imshow("Scanned", resize(ref, height=650))  # 显示扫描结果
cv2.waitKey(0)
