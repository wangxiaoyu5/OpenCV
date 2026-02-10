import numpy as np  # 导入NumPy库，用于数值计算
import cv2  # 导入OpenCV库，用于计算机视觉任务

# 打开视频文件
cap = cv2.VideoCapture('./img/test.avi')

# 角点检测所需参数
feature_params = dict(
    maxCorners=100,  # 最多检测100个角点
    qualityLevel=0.3,  # 角点质量等级，值越高角点质量越好
    minDistance=7  # 角点之间的最小距离
)

# Lucas-Kanade光流法参数
lk_params = dict(
    winSize=(15, 15),  # 搜索窗口大小
    maxLevel=2  # 金字塔最大层数
)

# 随机生成100种颜色，用于绘制轨迹
color = np.random.randint(0, 255, (100, 3))

# 读取第一帧图像
ret, old_frame = cap.read()
# 将第一帧转换为灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# 使用Shi-Tomasi算法检测角点，返回检测到的角点坐标
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个与原图大小相同的mask，用于绘制轨迹
mask = np.zeros_like(old_frame)

# 进入主循环
while True:
    # 读取下一帧图像
    ret, frame = cap.read()
    # 将当前帧转换为灰度图
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用Lucas-Kanade光流法计算角点在当前帧中的新位置
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1表示角点被成功跟踪，提取这些角点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制角点轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)  # 新角点坐标
        c, d = old.ravel().astype(int)  # 旧角点坐标
        # 在mask上绘制轨迹线
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # 在当前帧上绘制角点
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        # 显示mask和frame
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    # 将mask和当前帧叠加
    img = cv2.add(frame, mask)
    # 显示叠加后的图像
    cv2.imshow('img', img)

    # 等待按键，按ESC键退出
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # 更新前一帧和前一帧的角点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# 释放资源
cv2.destroyAllWindows()
cap.release()
