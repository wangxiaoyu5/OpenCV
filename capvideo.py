import cv2  # 导入 OpenCV 库，用于视频捕获和图像处理

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 创建视频编码器，使用 XVID 编码格式
VM = cv2.VideoWriter('./out.avi', fourcc, 25.0, (640, 480))  # 创建视频写入对象，输出文件为 output.mp4，帧率为 25fps，分辨率为 1280x720

cv2.namedWindow("video", cv2.WINDOW_NORMAL)  # 创建一个名为 "video" 的可调整大小的窗口
cv2.resizeWindow("video", 640, 480)  # 将窗口大小设置为 640x480 像素
cap = cv2.VideoCapture(0)  # 打开默认摄像头设备（索引为0）
# cap = cv2.VideoCapture("./img/output.mp4")  # 从视频文件中读取视频帧（被注释）


while cap.isOpened():  # 当摄像头正常打开时循环执行
    ret, frame = cap.read()  # 从摄像头读取一帧图像，ret 表示是否成功读取，frame 是图像数据
    if not ret:  # 如果读取失败则退出循环
        break
    cv2.imshow("video", frame)  # 在 "video" 窗口中显示当前帧
    cv2.resizeWindow("video", 640, 480)  # 重新设置窗口大小

    VM.write(frame)  # 将当前帧写入输出视频文件

    key = cv2.waitKey(10)  # 等待10毫秒的键盘输入，返回按键的ASCII码值

    if key & 0xFF == ord('q'):  # 检查是否按下 'q' 键（使用位运算处理不同平台的键值）
        break  # 如果按下 'q' 键则退出循环

cap.release()  # 释放摄像头资源
VM.release()  # 释放视频写入对象
cv2.destroyAllWindows()  # 销毁所有OpenCV窗口
