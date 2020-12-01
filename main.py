import cv2
import time
import threading
import numpy as np
import math
from PIL import Image
from skimage.measure import compare_ssim

def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(image1, image2, multichannel=True, full=True)
    return score


def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


def detec():
    video = cv2.VideoCapture('20191107-1446.MP4')
    history = 20  # 训练帧数

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)

    frames = 0
    HEIGHT = 720
    WIDTH = 1280
    fg_mask = None
    persons = []
    last = None
    s = 0
    while True:
        res, frame = video.read()
        base = frame.copy()
        out = cv2.VideoWriter('project_output_haar_and_svm1.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                              (WIDTH, HEIGHT))
        if type(frame) == type(None):
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        fg_mask = bs.apply(frame)  # 获取 foreground mask

        if frames < history:
            frames += 1
            continue
            # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 100, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # 获取所有检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        prev = None
        for c in contours:

            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if area > 10:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
            cv2.imshow("detection", frame)
            cv2.imshow("back", dilated)
            cur = base[x:x + w, y:y + h]
            if prev is not None:
                sim = classify_hist_with_split(cur, prev)
                print("similar %s", sim)
            prev = cur

        if cv2.waitKey(33) == 27:
            break


if __name__ == '__main__':
    detec()
