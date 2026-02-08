import cv2
import numpy as np

cap = cv2.VideoCapture('./img/vtest.avi')
# mog = cv2.bgsegm.createBackgroundSubtractorMOG()
# 可以计算出阴影 会产生噪点
# mog = cv2.createBackgroundSubtractorMOG2()
# 可以计算出阴影 减少产生噪点 如果采用默认值好长时间无显示 要更改默认值
mog  = cv2.bgsegm.createBackgroundSubtractorGMG(10)


while True:
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    cv2.imshow('frame', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()






















