# coding:utf-8

import numpy as np
import argparse
import cv2
import base64
import torch
import time
import os
from pathlib import Path
import yaml
import requests
from shapely.geometry import Polygon  # 多边形 pip install Shapely

from models.experimental import attempt_load
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.augmentations import letterbox


basedir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(basedir, "config.yaml")

# 加载 YAML 文件
with open(config_path, 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)


# 禁止区域非法侵入
def Cal_area_2poly(data1,data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """
    poly1 = Polygon(data1).convex_hull      # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积

    # print(inter_area)
    return inter_area


def judge_illegal_entry(boxes, roi_area):

    entry_objects_filter = [False]*len(boxes)
    for i, box in enumerate(boxes):
        vertices_4points = [[box[0],box[1]], [box[0],box[1]+box[3]], [box[0]+box[2],box[1]+box[3]], [box[0]+box[2],box[1]]]
        inter_area = Cal_area_2poly(vertices_4points, roi_area)
        entry_objects_filter[i] = True if inter_area > 0 else False
    return entry_objects_filter


def load_model(weights, imgsz=(640, 640), device="0", half=False):
    """
    :param weights: 推理模型路径
    :param imgsz: 输入图片的大小 默认640(pixels)
    :param device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
    :param half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    :return:
    """
    # 导入模型
    device1 = select_device(device)
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device1)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device1)['model']).to(device1).eval()
    if pt and device1.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device1).type_as(next(model.parameters())))  # run once

    return model, device1, names


def detect(img, model, device, names, conf_thres=0.6, iou_thres=0.45, max_det=1000, classes=None,
           agnostic_nms=False, augment=False, visualize=False, half=False, dnn=False):
    """
    :param img: 要推理的图片数据
    :param model: 推理模型
    :param names: 标签名字
    :param conf_thres: object置信度阈值 默认0.25  用在nms中
    :param iou_thres: 做nms的iou阈值 默认0.45   用在nms中
    :param max_det: 每张图片最多的目标数量  用在nms中
    :param device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
    :param classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
    :param agnostic_nms: 进行nms是否也除去不同类别之间的框 默认False
    :param augment: 预测是否也要采用数据增强 TTA 默认False
    :param visualize: 特征图可视化 默认FALSE
    :param half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    :param dnn: 是否使用OpenCV DNN进行ONNX推理,默认是False
    :return:
    """

    # model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    im0 = img
    im = letterbox(im0, 640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # www 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # 预测中
    pred = model(im, augment=augment, visualize=visualize)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # 返回结果
    detections = []
    im0r = im0c = img.copy()

    # 电子围栏区域
    rq_roi_area = config["video0"]["roi_areas"]
    rq_roi_area = np.array(rq_roi_area)
    rq_roi_area = rq_roi_area.astype('int')

    lg_roi_area = config["video0"]["roi_areas"]
    lg_roi_area = np.array(lg_roi_area)
    lg_roi_area = lg_roi_area.astype('int')

    for i, det in enumerate(pred):  # per image 每张图片
        annotator = Annotator(im0c, line_width=3, example=str(names))
        if len(det):
            # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # result
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if names[c] == "person":
                    rq_result = judge_illegal_entry([xyxy], rq_roi_area)
                    lg_result = judge_illegal_entry([xyxy], lg_roi_area)
                    # print(result)
                    label = f'{names[c]} {conf:.2f}'
                    if True in rq_result:
                        cv2.polylines(im0c, [np.array(rq_roi_area)], True, (0, 0, 255), 3)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        detections.append(names[c])
                    # if True not in lg_result:
                    #     cv2.polylines(im0c, [np.array(lg_roi_area)], True, (0, 0, 255), 3)
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    #     detections.append(names[c])
                else:
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detections.append(names[c])
        im0r = annotator.result()
    return detections, im0r

# def base64_input(base64_txt):
#     img_str = base64.b64decode(base64_txt)
#     im_ndarray = np.frombuffer(img_str, np.uint8)
#     image = cv2.imdecode(im_ndarray, cv2.IMREAD_COLOR)  # BGR
#     model_path = 'best.pt'
#     model, device, names = load_model(weights=model_path)
#     result, img = detect(img=image, model=model, device=device, names=names)
#     print(result)
#     return result
#
#
# def image_input(image, img_save_path):
#     model_path = r'I:\Myproject\tuwei_yolov5-v6.0\runs\train\all\weights\best.pt'
#     model, device, names = load_model(weights=model_path)
#     start_time = time.time()
#     result, img = detect(img=image, model=model, device=device, names=names)
#     save_path = os.path.join(img_save_path, "output.jpg")
#     cv2.imwrite(save_path, img)
#     print(result)
#     print("预测时间：", time.time()-start_time)
#     # return result


def extract_frames(video_path, mode, value, post_url, thresh=0.6, detect_goal=None):
    """
    video_path: 视频文件路径
    mode: 'seconds' 或 'frames'
    value:如果 mode 是 'seconds'，表示每秒抽取几帧；如果 mode 是 'frames'，表示间隔多少帧抽取一帧
    thresh: 置信度阈值，默认为0.6
    detect_goal: 检测目标，默认为None，表示全目标检测。格式为[0, 1, 2, 3]
    """

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    saved_count = 0

    # model_path = '/usr/src/app/runs/train/all/weights/best.pt'
    model_path = '/Users/pengshuaifeng/PycharmProjects/gzzn/yolov5-v6.0/runs/train/all/weights/best.pt'

    model, device, names = load_model(weights=model_path, device=select_device('cpu'))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'seconds':
            interval = int(fps / value)  # 每秒抽取几帧
        elif mode == 'frames':
            interval = value  # 间隔多少帧抽取一帧
        else:
            raise ValueError("Mode should be 'seconds' or 'frames'")

        if frame_count % interval == 0:
            time_num = int(time.time() * 1000)
            saved_count += 1
            result, img = detect(img=frame, model=model, device=device, names=names, conf_thres=thresh, classes=detect_goal)

            result_dic = {}
            for item_str in result:
                if item_str not in result_dic:
                    result_dic[item_str] = 1
                else:
                    result_dic[item_str] += 1

            if len(result):
                # 将 img 转换为 base64 编码
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                data = {
                    "category":"alarm",
                    "info": {
                        "alarmDate":int(time.time()), 
                        "deviceName":"A21黄阁大道北61号化工中心门口", 
                        "alarmGrade":"2", 
                        "alarmSource":"2",
                        "alarmType":"1430", 
                        "alarmPicture": img_base64  # 将 base64 编码的图片添加到数据中
                    }
                }
                print(f"调用远程接口，推送预警信息：开始,url:{post_url},frame:{frame_count}")
                headers={}
                headers['Content-Type'] = 'application/json'
                resp = requests.post(post_url, json=data,headers=headers)
                print(f"调用远程接口，推送预警信息：结束,url:{post_url},frame:{frame_count},resp:{resp.json()}")

            # resp = requests.post(post_url, data={'result': result_dic}, files={'file': open(img_save_path, 'rb'), })
            # os.remove(img_save_path)

        frame_count += 1
    print(f"Total saved frames: {saved_count}")


def main():
    parser = argparse.ArgumentParser()
    # 添加模型编码参数，类型为字符串，必填
    parser.add_argument('--token',type=str,required=True, help='视频流令牌')
    params = parser.parse_args()

    video = config["video0"]
    skip_frames = video["skip_frames"]
    detect_goal = video["detect_goal"]
    warning_url = video["warning_url"]
    thresh = video["conf_thresh"]

    extract_frames(video_path=video["input"]+params.token, mode="frames", value=skip_frames, post_url=warning_url, thresh=thresh, detect_goal=detect_goal)

def local():
    # 获取参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, help='类型：检测类型')
    parser.add_argument('--path', type=str, required=True, help='视频文件路径')
    params=parser.parse_args()
    video = config[params.type]
    skip_frames = video["skip_frames"]
    detect_goal = video["detect_goal"]
    warning_url = video["warning_url"]
    thresh = video["conf_thresh"]

    extract_frames(video_path=params.path, mode="frames", value=skip_frames, post_url=warning_url, thresh=thresh, detect_goal=detect_goal)

# if __name__ == '__main__':
#      main()

if __name__ == '__main__':
     local()

