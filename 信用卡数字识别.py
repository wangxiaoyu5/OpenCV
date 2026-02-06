
import cv2
import numpy as np
from matplotlib import pyplot as plt

import myutils

img = cv2.imread('./img/ocr_a_reference.png')
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_color, 10, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_draw = img.copy()
cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 2)  # 红色轮廓，线宽2
print(len(contours))
digits = {}
contours = myutils.sort_contours(contours, method="left-to-right")[0]

for i, c in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88), interpolation=cv2.INTER_AREA)
    digits[i] = roi

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

image = cv2.imread('./img/credit_card_01.png')
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, squareKernel)
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = contours
image_draw = image.copy()
cv2.drawContours(image_draw, conts, -1, (0, 0, 255), 2)
cv2.imshow('image_draw', image_draw)
locs = []

for (i, c) in enumerate(conts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])
output = []
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]

    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = myutils.sort_contours(contours, method="left-to-right")[0]
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88), interpolation=cv2.INTER_AREA)

        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)











cv2.imshow('gradX', gradX)
cv2.imshow('image', image)
cv2.imshow('tophat', tophat)
# cv2.imshow('contours', img_draw)
#
# cv2.imshow('thresh', thresh)
# cv2.imshow('img_color', img_color)
# cv2.imshow('img', img)

cv2.waitKey(0)
