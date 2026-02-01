import cv2
import numpy as np


class APP:
    startX = 0
    startY = 0
    rect = (0, 0, 0, 0)
    flag_rect = False

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.startX = x
            self.startY = y
            self.flag_rect = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_rect = False
            cv2.rectangle(self.img, (self.startX, self.startY), (x, y), (0, 255, 0), 3)
            self.rect = (min(self.startX, x), min(self.startY, y),
                         abs(self.startX - x), abs(self.startY - y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_rect:
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.startX, self.startY), (x, y), (255, 255, 0), 2)

    def run(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.onmouse)

        self.img = cv2.imread(r"./img/flower.png")
        self.img2 = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)
        while (1):
            cv2.imshow('input', self.img)
            cv2.imshow('output', self.output)
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
            if key == ord('g'):
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1,
                            cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((self.mask == 1) | (self.mask == 3), 255, 0).astype('uint8')
            self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)


APP().run()
