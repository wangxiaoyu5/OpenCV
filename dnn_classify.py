import cv2  # 导入OpenCV库，用于图像处理和计算机视觉任务
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

config = "./model/bvlc_googlenet.prototxt"  # 定义Caffe模型的配置文件路径
model = "./model/bvlc_googlenet.caffemodel"  # 定义Caffe模型的权重文件路径
net = cv2.dnn.readNetFromCaffe(config, model)  # 从Caffe模型加载网络结构和权重

img = cv2.imread(r"./img/1.jpg")  # 读取待分类的图像文件（猫的图片）
blob = cv2.dnn.blobFromImage(img, 1,  # 将图像转换为模型输入所需的blob格式
                         (224, 224), (104, 117, 123), swapRB=True, crop=True)  # 设置输入尺寸、均值、通道交换和裁剪参数

net.setInput(blob)  # 将处理后的图像数据设置为神经网络的输入
out = net.forward()  # 运行前向传播，获取模型输出结果

classes = []  # 初始化一个空列表，用于存储类别名称
with open("./model/synset_words.txt", "rt") as f:  # 打开类别标签文件
    classes = [x[x.find(" ") + 1:] for x in f]  # 解析每一行，提取类别名称并存入列表

order = sorted(out[0], reverse=True)  # 对模型输出的概率值进行降序排序
z = list(range(3))  # 创建一个长度为3的列表，用于存储前三名的索引
for i in range(0, 3):  # 遍历前三名结果
    z[i] = np.where(out[0] == order[i])[0][0]  # 查找当前概率值对应的类别索引
    print("第", i + 1, "名：", classes[z[i]], "  概率：", order[i])  # 打印排名、类别名称和概率值
    print("类所在行", z[i] + 1)  # 打印该类别在synset_words.txt中的行号
