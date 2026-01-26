import cv2
import numpy as np

# 读取图片
img = cv2.imread('./img/1.jpg')
img_original = cv2.imread('./img/login.png')

# 获取第一张图片的尺寸并调整第二张图片
height, width = img.shape[:2]
img2 = cv2.resize(img_original, (width, height))

print(f"img shape: {img.shape}")
print(f"img2 shape: {img2.shape}")

# 确保尺寸一致后再进行混合
weighted_result = cv2.addWeighted(img, 0.5, img2, 0.5, 0)

cv2.imshow("result", weighted_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
