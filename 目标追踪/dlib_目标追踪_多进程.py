# 导入自定义FPS计算模块，用于计算处理帧率
from utils import FPS
# 导入多进程库，用于并行运行多个追踪器
import multiprocessing
# 导入NumPy库，用于数组操作和处理检测框坐标
import numpy as np
# 导入argparse库，用于命令行参数解析
import argparse
# 导入dlib库，提供correlation_tracker相关滤波追踪器
import dlib
# 导入OpenCV库，用于视频读写、图像处理和DNN检测
import cv2

# 存放所有追踪进程的输入队列和输出队列
inputQueues = []
outputQueues = []


# 定义start_tracker函数，用于启动和管理单个目标的追踪器
def start_tracker(box, label, rgb, inputQueue, outputQueue):
    # 初始化dlib追踪器
    t = dlib.correlation_tracker()
    # 将边界框转换为dlib.rectangle格式
    rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    # 在第一帧中开始追踪
    t.start_track(rgb, rect)

    # 进入无限循环，持续处理后续帧
    while True:
        # 从输入队列获取下一帧（阻塞等待）
        rgb = inputQueue.get()

        # 非空帧则进行追踪
        if rgb is not None:
            # 更新追踪器，寻找目标新位置
            t.update(rgb)
            pos = t.get_position()  # 返回dlib.rectangle对象

            # 提取边界框坐标
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # 将标签和新框放入输出队列
            outputQueue.put((label, (startX, startY, endX, endY)))


# SSD模型输出的21类标签（索引0为background）
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# 加载预训练Caffe SSD检测模型
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",  # 网络结构文件
    "MobileNetSSD_deploy.caffemodel"  # 权重文件
)

# 打开输入视频流
cap = cv2.VideoCapture("race.mp4")
if not cap.isOpened():
    raise FileNotFoundError("无法打开视频文件: race.mp4")

# 视频写入器（初始化为None，后续按需创建）
writer = None

# 追踪器与标签列表：每个目标对应一个tracker + label
trackers = []  # 存储dlib.correlation_tracker实例
labels = []  # 存储对应类别名（如"person"）

# 初始化FPS计时器
fps = FPS().start()

# 主处理循环：逐帧读取并处理
while True:
    ret, frame = cap.read()
    if not ret or frame is None:  # 视频结束或读取失败
        break

    # 图像预处理：缩放至固定宽度600px（保持宽高比）
    (h, w) = frame.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  # 抗锯齿缩放
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转RGB（dlib要求）

    # 【首次】初始化视频写入器（仅执行一次）
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Motion JPEG编码
        writer = cv2.VideoWriter(
            "output.avi",  # 输出路径
            fourcc,  # 编码器
            30,  # 帧率（fps）
            (frame.shape[1], frame.shape[0]),  # 分辨率(宽, 高)
            True  # 彩色视频
        )

    # 阶段一：初始检测（仅当无追踪器时执行）
    if len(trackers) == 0:
        (h, w) = frame.shape[:2]
        # 构建网络输入blob（归一化、减均值）
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(blob)
        detections = net.forward()  # 输出shape=(1, 1, N, 7)

        # 遍历所有检测结果（N个候选框）
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # 置信度（第2列）
            if confidence > 0.2:  # 过滤弱检测
                idx = int(detections[0, 0, i, 1])  # 类别索引（第1列）
                label = CLASSES[idx]
                if label != "person":  # 仅追踪人
                    continue

                # 解析边界框：归一化坐标→像素坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # [x_min, y_min, x_max, y_max]
                (startX, startY, endX, endY) = box.astype("int")
                bbox = (startX, startY, endX, endY)

                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()

                inputQueues.append(iq)
                outputQueues.append(oq)

                p = multiprocessing.Process(target=start_tracker, args=(bbox, label, rgb, iq, oq))
                p.daemon = True
                p.start()

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


            else:
                # 多个追踪器处理的都是相同输入
                for iq in inputQueues:
                    iq.put(rgb)  # 将当前帧图像放入每个追踪器的输入队列中
                for oq in outputQueues:
                    # 从每个追踪器的输出队列中获取更新后的结果
                    (label, (startX, startY, endX, endY)) = oq.get()
                    # 在当前帧上绘制更新后的边界框和标签
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)  # 绘制绿色矩形框
                    cv2.putText(frame, label, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)  # 添加目标类别标签

            # 如果视频写入器已初始化，则将当前帧写入输出视频文件
            if writer is not None:
                writer.write(frame)
            # 显示当前帧图像
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF  # 等待按键输入（1ms延迟）
            # 如果按下ESC键（ASCII码为27），则退出循环
            if key == 27:
                break
            # 更新FPS计时器
            fps.update()
            # 停止FPS计时器并打印统计信息
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))  # 打印总耗时
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))  # 打印平均帧率
            # 释放资源：关闭视频写入器、销毁所有OpenCV窗口、释放视频捕获对象
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            cap.release()
