import cv2
import numpy as np

cap = cv2.VideoCapture('./img/vtest.avi')
# mog = cv2.bgsegm.createBackgroundSubtractorMOG()
# 可以计算出阴影 会产生噪点
mog = cv2.createBackgroundSubtractorMOG2()
# 可以计算出阴影 减少产生噪点 如果采用默认值好长时间无显示 要更改默认值
# mog  = cv2.bgsegm.createBackgroundSubtractorMOG2BackgroundSubtractorGMG(10)
#形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    ret, frame = cap.read()
    fgmask = mog.apply(frame)

    # 形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的轮廓
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)
        if perimeter > 188:
            # 找到一个直矩形（不会旋转）
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()






















