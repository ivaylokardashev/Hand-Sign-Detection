from typing import List
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from matplotlib import pyplot as plt
from math import ceil
import time


def center_cropped_image_in_matrix(imgWhite: List[int], h: int, w: int, imgSize):
    if aspectRatio > 1:
        k = imgSize / h
        wCal = ceil(k * w)

        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        imgResizeShape = imgResize.shape
        # to center imgCrop in imgWhite
        wGap = ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap: wCal + wGap] = imgResize
    else:
        k = imgSize / w
        hCal = ceil(k * h)

        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        imgResizeShape = imgResize.shape
        # to center imgCrop in imgWhite
        hGap = ceil((imgSize - hCal) / 2)
        imgWhite[hGap: hCal + hGap, :] = imgResize


def save_images(directory, imgCounter, imgWhite):
    cv2.imwrite(f'{directory}/Image_{time.time()}.jpg', imgWhite)
    print(imgCounter)


ESC = 27
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 15
imgSize = 300
whiteColorCode = 255


directory = "Data/Z"
imgCounter = 0

while True:
    success, img = cap.read()
    # this code mirroring the video
    img = cv2.flip(img, 1)
    # flipType parameter is necessary to be set to False because compliance in hands aka (right=right and left=left)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * whiteColorCode
        # use 'abs' function because values can't be negative
        # it has some errors without function that are thrown from imshow
        # abs resolve the problem
        imgCrop = img[abs(y-offset): y + h+offset, abs(x-offset): x + w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        center_cropped_image_in_matrix(imgWhite, h, w, imgSize)

        try:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        except cv2.error:
            pass

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ESC:
        exit()
    elif key == ord("s"):
        imgCounter += 1
        save_images(directory, imgCounter, imgWhite)

