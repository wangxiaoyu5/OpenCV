import cv2  # 导入 OpenCV 库，用于图像处理和显示功能

cv2.namedWindow("image")  # 创建一个名为 "image" 的窗口，用于显示图像

img = cv2.imread(r"./img/login.png")  # 读取指定路径的图片文件，r"" 表示原始字符串，避免转义字符问题
while True:

    cv2.imshow("image", img)  # 在 "image" 窗口中显示加载的图片

    key = cv2.waitKey(0)  # 等待键盘按键输入，参数 0 表示无限等待，返回按键的 ASCII 码值

    if key & 0xFF== ord('q'):  # 判断按键是否为 'q' 键（'q' 的 ASCII 码值为 113）
        break # 如果按下 'q' 键则退出程序
    elif key & 0xFF== ord('s'):  # 判断按键是否为 's' 键（'s' 的 ASCII 码值为 115）
        cv2.imwrite("./img/login_save.png", img)  # 保存图片到指定路径，r"" 表示原始字符串，避免转义字符问题
        print("保存成功")
    else:
        print("请输入正确的按键")

cv2.destroyAllWindows()  # 销毁所有已创建的 OpenCV 窗口，释放资源
