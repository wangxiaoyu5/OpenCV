# -*- coding: utf-8 -*-
# 导入依赖库
from utils import FPS          # 自定义帧率统计类（需确保 utils.py 中定义了 class FPS）
import numpy as np            # 数值计算
import argparse               # 命令行参数解析（本文件未实际使用 args，但保留兼容性）
import dlib                   # 目标跟踪核心库（含 dlib.correlation_tracker）
import cv2                    # OpenCV 计算机视觉库

# SSD 模型输出的 21 类标签（索引 0 为 background）
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# 加载预训练 Caffe SSD 检测模型
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",   # 网络结构文件
    "MobileNetSSD_deploy.caffemodel"  # 权重文件
)

# 打开输入视频流
cap = cv2.VideoCapture("race.mp4")
if not cap.isOpened():
    raise FileNotFoundError("无法打开视频文件: race.mp4")

# 视频写入器（初始化为 None，后续按需创建）
writer = None

# 追踪器与标签列表：每个目标对应一个 tracker + label
trackers = []   # 存储 dlib.correlation_tracker 实例
labels = []     # 存储对应类别名（如 "person"）

# 初始化 FPS 计时器
fps = FPS().start()

# 主处理循环：逐帧读取并处理
while True:
    ret, frame = cap.read()
    if not ret or frame is None:  # 视频结束或读取失败
        break

    # 图像预处理：缩放至固定宽度 600px（保持宽高比）
    (h, w) = frame.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  # 抗锯齿缩放
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转 RGB（dlib 要求）

    # 【首次】初始化视频写入器（仅执行一次）
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Motion JPEG 编码
        writer = cv2.VideoWriter(
            "output.avi",                # 输出路径
            fourcc,                      # 编码器
            30,                          # 帧率（fps）
            (frame.shape[1], frame.shape[0]),  # 分辨率 (宽, 高)
            True                         # 彩色视频
        )

    # 阶段一：初始检测（仅当无追踪器时执行）
    if len(trackers) == 0:
        (h, w) = frame.shape[:2]
        # 构建网络输入 blob（归一化、减均值）
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(blob)
        detections = net.forward()  # 输出 shape=(1, 1, N, 7)

        # 遍历所有检测结果（N 个候选框）
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # 置信度（第 2 列）
            if confidence > 0.2:                 # 过滤弱检测
                idx = int(detections[0, 0, i, 1])   # 类别索引（第 1 列）
                label = CLASSES[idx]
                if label != "person":               # 仅追踪人
                    continue

                # 解析边界框：归一化坐标 → 像素坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # [x_min, y_min, x_max, y_max]
                (startX, startY, endX, endY) = box.astype("int")

                # 创建并初始化 dlib 追踪器
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)  # 构造 bbox
                t.start_track(rgb, rect)                           # 用当前帧初始化


                # 存储追踪器与标签
                labels.append(label)
                trackers.append(t)

                # 绘制检测框与标签
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 阶段二：持续跟踪（已有追踪器时执行）
    else:
        for (t, l) in zip(trackers, labels):
            t.update(rgb)                        # 更新追踪器到当前帧
            pos = t.get_position()               # 获取新位置（dlib.rectangle）
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # 绘制跟踪框与标签
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, l, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    if writer is not None:
        writer.write(frame)  # 写入当前帧到输出视频

    cv2.imshow("Frame", frame)           # 显示当前帧
    key = cv2.waitKey(1) & 0xFF         # 等待 1ms，获取按键
    if key == 27:                       # 按 Esc 退出
        break

    fps.update()  # 更新帧计数（注意：应在每帧处理后调用）


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# 释放资源
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
cap.release()
