import cv2
import numpy as np
from matplotlib import pyplot as plt

import myutils

# 读取参考图像（包含数字模板）
img = cv2.imread('./img/ocr_a_reference.png')
# 将图像转换为灰度图
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对灰度图进行二值化处理，阈值为10，背景为白色，前景为黑色
ret, thresh = cv2.threshold(img_color, 10, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 复制原图用于绘制轮廓
img_draw = img.copy()
# 绘制所有轮廓，红色线条，线宽为2
cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 2)
# 打印找到的轮廓数量
print(len(contours))
# 创建一个字典存储每个数字的ROI区域
digits = {}
# 按照从左到右的顺序对轮廓进行排序
contours = myutils.sort_contours(contours, method="left-to-right")[0]

# 遍历排序后的轮廓，提取每个数字的ROI并保存
for i, c in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)  # 获取轮廓的边界框
    roi = thresh[y:y + h, x:x + w]      # 提取ROI区域
    roi = cv2.resize(roi, (57, 88), interpolation=cv2.INTER_AREA)  # 调整大小为57x88
    digits[i] = roi  # 存储到字典中

# 定义结构元素
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # 矩形核
squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 正方形核

# 读取信用卡图像
image = cv2.imread('./img/credit_card_01.png')
# 调整图像宽度为300像素
image = myutils.resize(image, width=300)
# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 使用顶帽变换增强图像中的亮区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# 计算X方向的梯度
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)  # 取绝对值
(minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 获取最小和最大值
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化到0-255范围
gradX = gradX.astype("uint8")  # 转换为无符号8位整数

# 使用闭运算连接断裂的部分
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

# 对梯度图像进行二值化处理
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 再次使用闭运算填充小孔洞
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, squareKernel)
# 查找新的轮廓
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = contours
# 复制图像用于绘制轮廓
image_draw = image.copy()
# 绘制所有轮廓
cv2.drawContours(image_draw, conts, -1, (0, 0, 255), 2)
cv2.imshow('image_draw', image_draw)
# 初始化位置列表
locs = []

# 遍历轮廓，筛选出符合要求的数字区域
for (i, c) in enumerate(conts):
    (x, y, w, h) = cv2.boundingRect(c)  # 获取边界框
    ar = w / float(h)  # 计算宽高比
    # 判断是否满足条件：宽高比在2.5~4之间，宽度在40~55之间，高度在10~20之间
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))  # 添加符合条件的位置信息

# 按照x坐标排序位置信息
locs = sorted(locs, key=lambda x: x[0])
output = []  # 初始化输出列表
# 遍历每个数字组的位置信息
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    # 提取当前数字组的图像区域
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # 进行二值化处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 查找内部轮廓
    contours, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按照从左到右排序
    contours = myutils.sort_contours(contours, method="left-to-right")[0]
    # 遍历每个子轮廓
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)  # 获取边界框
        roi = group[y:y + h, x:x + w]      # 提取ROI
        roi = cv2.resize(roi, (57, 88), interpolation=cv2.INTER_AREA)  # 调整大小
        cv2.imshow("ROI", roi)  # 显示ROI图像
        scores = []  # 初始化匹配得分列表
        # 遍历模板数字，计算匹配得分
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)  # 模板匹配
            (_, score, _, _) = cv2.minMaxLoc(result)  # 获取最大匹配得分
            scores.append(score)
        # 找到得分最高的数字作为识别结果
        groupOutput.append(str(np.argmax(scores)))

    # 在原始图像上绘制矩形框和识别结果
    cv2.rectangle(image, (gX, gY), (gX + gW, gY + gH), (0, 0, 255), 2)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


    output.extend(groupOutput)

# 打印最终识别结果
print("识别结果:", "".join(output))

# 显示中间处理步骤的图像（可选）
# cv2.imshow('gradX', gradX)
# cv2.imshow('image', image)
# cv2.imshow('tophat', tophat)
# cv2.imshow('contours', img_draw)
# cv2.imshow('thresh', thresh)
# cv2.imshow('img_color', img_color)
# cv2.imshow('img', img)

# 等待按键退出
cv2.waitKey(0)
