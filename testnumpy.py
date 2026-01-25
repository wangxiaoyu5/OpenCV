import numpy as np
import cv2

# a = np.array([1, 2, 3])
# print(a)
# b = np.array([[1, 2, 3], [4, 5, 6]])
# print(b)
#
# c = np.zeros((480, 640), np.uint8)
c = np.zeros((480, 640, 3), np.uint8)
# print( c)
#
# d = np.ones((8, 8, 3), np.uint8) * 255
# print( d)
#
# e = np.full((8, 8, 3), 255, np.uint8)
# print( e)
#
# f = np.identity(4)
# print( f)
#
# g = np.eye(4, 4, k=3)
# print( g)
# 从矩阵中获取元素的值 [y,x]
# print(c[100, 100])
# count = 0
# # 向矩阵中某个元素赋值
# while count < 100:
#     # RGB
#     c[count, 100, 0] = 255
#     # c[count, 100] = [0,0,255]
#     count += 1

roi = c[100:400, 100:600]
roi[:, :] = [0, 0, 255]
roi[10:200, 10:200] = [0, 255, 0]


cv2.imshow("img", roi)
key = cv2.waitKey(0)
if key & 0xFF == ord('q'):
    cv2.destroyAllWindows()



















