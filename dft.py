import numpy as np          # 导入NumPy库，用于数值计算
import cv2                 # 导入OpenCV库，用于图像处理
from matplotlib import pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘图

img = cv2.imread('./img/cat.jpg', 0)  # 读取灰度图像（0表示以灰度模式读取）

img_float32 = np.float32(img)         # 将图像数据转换为32位浮点型，以便进行傅里叶变换

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # 对图像进行离散傅里叶变换（DFT），输出复数形式
dft_shift = np.fft.fftshift(dft)      # 将频谱中心移动到图像中心，便于后续处理

rows, cols = img.shape                # 获取图像的高度和宽度
crow, ccol = int(rows / 2), int(cols / 2)  # 计算图像中心点坐标

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)  # 创建一个全为1的掩膜（高通滤波用）
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # 将中心区域设为0，保留高频成分

# # 低通滤波
# mask = np.zeros((rows, cols, 2), np.uint8)  # 创建一个全为0的掩膜（低通滤波用）
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  # 将中心区域设为1，保留低频成分

fshift = dft_shift * mask              # 应用掩膜，对频谱进行滤波
f_ishift = np.fft.ifftshift(fshift)    # 将频谱移回原始位置
img_back = cv2.idft(f_ishift)          # 进行逆离散傅里叶变换（IDFT），恢复图像
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算复数结果的幅度值，得到最终图像

plt.subplot(121), plt.imshow(img, cmap='gray')  # 显示原始图像
plt.title('Input Image'), plt.xticks([]), plt.yticks([])  # 设置标题并隐藏坐标轴刻度
plt.subplot(122), plt.imshow(img_back, cmap='gray')  # 显示处理后的图像
plt.title('Result'), plt.xticks([]), plt.yticks([])  # 设置标题并隐藏坐标轴刻度

plt.show()  # 显示图像
