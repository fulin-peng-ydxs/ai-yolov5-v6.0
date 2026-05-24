# -*- coding: utf-8 -*-
import time
import argparse

import cv2
import numpy as np
tpPointsChoose = []
drawing = False
tempFlag = False


def draw_ROI(event, x, y, flags, param):
    global point1, tpPointsChoose,pts,drawing, tempFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        tempFlag = True
        drawing = False
        point1 = (x, y)
        tpPointsChoose.append((x, y))  # 用于画点
    if event == cv2.EVENT_RBUTTONDOWN:
        tempFlag = True
        drawing = True
        pts = np.array([tpPointsChoose], np.int32)
        pts1 = tpPointsChoose[1:len(tpPointsChoose)]
        # print(pts1)

        mask = np.array(pts1, dtype=np.int32).reshape(-1).tolist()
        # print(mask)
        print('mask:  poly,'+','.join(list(map(str, mask))))
        roi_list = []
        for i in range(0, len(mask)):
            if i % 2 == 0:
                roi_list.append([mask[i], mask[i+1]])
        print(roi_list)

    if event == cv2.EVENT_MBUTTONDOWN:
        tempFlag = False
        drawing = True
        tpPointsChoose = []


winName = 'draw mask'
# mp4_path = '/home/nano/mqc-work/video-service/video.avi'

parser = argparse.ArgumentParser()
parser.add_argument('-mp','--mp4_path', type=str, default=r"I:\Myproject\test\test.mp4")
args = parser.parse_args()
mp4_path = args.mp4_path


cv2.namedWindow(winName,0)
cv2.setMouseCallback(winName, draw_ROI)
cap = cv2.VideoCapture(mp4_path)  # 文件名及格式
# cap = cv2.VideoCapture(0)

fps=cap.get(cv2.CAP_PROP_FPS)
size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps: {}\nsize: {}".format(fps,size))

# vfps = 0.7/fps  #延迟播放用，根据运算能力调整
while (True):
    # capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(mp4_path)
        continue
        #print('can not read frame')
        #break 

    # display the resulting frame
    if (tempFlag == True and drawing == False) :  # 鼠标点击
        cv2.circle(frame, point1, 5, (0, 255, 0), 2)
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i + 1], (255, 0, 0), 2)
    if (tempFlag == True and drawing == True):  #鼠标右击
        cv2.polylines(frame, [pts], True, (0, 0, 255), thickness=2)
    if (tempFlag == False and drawing == True):  # 鼠标中键
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
    # time.sleep(vfps)
    cv2.imshow(winName, frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
    #     break
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q') or c == 27:  # 按q键退出
        break
# when everything done , release the capture

cap.release()
cv2.destroyAllWindows()
