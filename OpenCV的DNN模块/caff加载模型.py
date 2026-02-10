import os

import cv2
import numpy as np
import os


def img_path(path):
    img_paths = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # 支持的图像格式
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(image_extensions):  # 只处理图像文件
                full_path = os.path.join(root, file)
                img_paths.append(full_path)
    return img_paths


rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
                               "bvlc_googlenet.caffemodel")

img_path_list =img_path(r".\images")
img = cv2.imread(img_path_list[0])
resized = cv2.resize(img, (224, 224))

blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

net.setInput(blob)
preds = net.forward()

idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx],
                                   preds[0][idx] * 100)
cv2.putText(img, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)

image_list = []
for p in img_path_list[1:]:
    img = cv2.imread(p)
    resized = cv2.resize(img, (224, 224))
    image_list.append(resized)



blob = cv2.dnn.blobFromImages(image_list, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

net.setInput(blob)
preds = net.forward()


for i, p in enumerate(img_path_list[1:]):
    img = cv2.imread(p)

    idx = np.argsort(preds[i])[::-1][0]
    text = "Label: {}, {:.2f}%".format(classes[idx],
                                       preds[i][idx] * 100)
    cv2.putText(img, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)









