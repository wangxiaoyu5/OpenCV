import argparse  # 导入argparse模块，用于解析命令行参数
import time      # 导入time模块，用于时间相关操作
import cv2       # 导入OpenCV库，用于图像和视频处理
import numpy as np  # 导入numpy库，用于数值计算


# 配置参数
ap = argparse.ArgumentParser()  # 创建ArgumentParser对象，用于处理命令行参数
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")  # 添加视频文件路径参数
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")  # 添加跟踪器类型参数，默认为"kcf"
args = vars(ap.parse_args())  # 解析命令行参数并转换为字典

# opencv已经实现了的追踪算法
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,  # CSRT跟踪器
	"kcf": cv2.legacy.TrackerKCF_create,    # KCF跟踪器
	"boosting": cv2.legacy.TrackerBoosting_create,  # Boosting跟踪器
	"mil": cv2.legacy.TrackerMIL_create,    # MIL跟踪器
	"tld": cv2.legacy.TrackerTLD_create,    # TLD跟踪器
	"medianflow": cv2.legacy.TrackerMedianFlow_create,  # MedianFlow跟踪器
	"mosse": cv2.legacy.TrackerMOSSE_create  # MOSSE跟踪器
}

# 实例化OpenCV's multi-object tracker
trackers = cv2.legacy.MultiTracker_create()  # 创建多目标跟踪器实例
# vs = cv2.VideoCapture(args["video"])  # 打开视频文件
vs = cv2.VideoCapture('los_angeles.mp4')
# 视频流
while True:
	# 取当前帧
	frame = vs.read()  # 读取视频的下一帧
	# (true, data)
	frame = frame[1]  # 获取帧数据
	# 到头了就结束
	if frame is None:  # 如果没有更多帧，则退出循环
		break

	# resize每一帧
	(h, w) = frame.shape[:2]  # 获取帧的高度和宽度
	width=600  # 设置目标宽度
	r = width / float(w)  # 计算缩放比例
	dim = (width, int(h * r))  # 计算新的尺寸
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  # 调整帧大小

	# 追踪结果
	(success, boxes) = trackers.update(frame)  # 更新跟踪器并获取跟踪框

	# 绘制区域
	for box in boxes:  # 遍历所有跟踪框
		(x, y, w, h) = [int(v) for v in box]  # 将跟踪框坐标转换为整数
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 在帧上绘制矩形框

	# 显示
	cv2.imshow("Frame", frame)  # 显示当前帧
	key = cv2.waitKey(100) & 0xFF  # 等待按键输入

	if key == ord("s"):  # 如果按下"s"键
		# 选择一个区域，按s
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)  # 用户选择一个ROI区域

		# 创建一个新的追踪器
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()  # 根据参数创建指定类型的跟踪器
		trackers.add(tracker, frame, box)  # 将新跟踪器添加到多目标跟踪器中

	# 退出
	elif key == 27:  # 如果按下"Esc"键
		break  # 退出循环
vs.release()  # 释放视频捕获对象
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
