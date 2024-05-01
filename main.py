from typing import List
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from math import ceil
import numpy as np


def center_cropped_image_in_matrix(imgWhite: List[float], h: float, w: float, imgSize, classifier):
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


def classifie_images(imgWhite, classifier):
    prediction, index = classifier.getPrediction(imgWhite)
    print(prediction, index)

def get_classifie_index(imgWhite, classifier):
    _, index = classifier.getPrediction(imgWhite)
    return index



ESC = 27
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 15
imgSize = 300
whiteColorCode = 255


directory = "Data/C"
imgCounter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    # this code mirroring the video
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
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

        # this center imgCrop in imgWhite by height and width
        center_cropped_image_in_matrix(imgWhite,h, w, imgSize, classifier)

        classifie_images(imgWhite, classifier)

        index = get_classifie_index(imgWhite, classifier)

        # rectangle that helps a letter to be more visible
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)

        # a letter
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # rectangle around hand
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        try:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        except cv2.error:
            pass

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    if key == ESC:
        exit()

