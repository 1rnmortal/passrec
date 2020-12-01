import cv2
import time
import threading
import numpy as np
import math

def detec():
    video = cv2.VideoCapture('20191107-1446.MP4')
    history = 20  # 训练帧数

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)

    frames = 0
    HEIGHT = 720
    WIDTH = 1280
    fg_mask = None
    while True:
        res, frame = video.read()
        out = cv2.VideoWriter('project_output_haar_and_svm1.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                              (WIDTH, HEIGHT))
        if type(frame) == type(None):
            break

        frame = cv2.resize(frame,(WIDTH, HEIGHT))

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
        for c in contours:
                    # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if area > 10:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
            cv2.imshow("detection", frame)
            cv2.imshow("back", dilated)
        if cv2.waitKey(33) == 27:
            break
if __name__ == '__main__':
    detec()