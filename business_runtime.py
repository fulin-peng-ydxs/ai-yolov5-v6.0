# coding:utf-8
"""
run.py / run_local.py 的共享业务运行时。

职责边界：
- 统一命令行参数、日志格式、模型选择、视频采集、YOLO 推理、ROI 判断、去抖和告警冷却。
- online 模式：由 run.py 进入，真实上报任务状态和告警接口。
- local 模式：由 run_local.py 进入，不发真实接口，只保存告警截图到 outputs/alarm_images/。

设计约定：
- 两个入口脚本不能复制业务流程；差异只能通过 config.mode 在本文件内集中处理。
- OpenCV、Torch、YOLO 相关模块按需导入，保证 `python run.py --help` 不会触发重型依赖初始化。
- 默认模型路径面向 Docker 部署；本地调试可通过 --weights / --fire_weights 覆盖。
"""

import argparse
import base64
import hashlib
import json
import logging
import queue
import random
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALARM_IMAGE_DIR = OUTPUT_DIR / "alarm_images"

# Docker 部署默认权重路径。本地调试建议显式传 --weights。
DEFAULT_WEIGHTS = "/usr/src/app/runs/train/all/weights/best.pt"
DEFAULT_FIRE_WEIGHTS = "/usr/src/app/runs/models/fire/best.pt"
ALARM_QUEUE_MAXSIZE = 10
RTSP_RECONNECT_BASE_DELAY = 2
RTSP_RECONNECT_MAX_DELAY = 30
NETWORK_ERROR_LOG_INTERVAL = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)
logger = logging.getLogger("yolov5-business")

config = None
# 只保留最新帧，避免 RTSP/推理速度不匹配导致延迟持续堆积。
frame_queue = queue.Queue(maxsize=1)
# 告警上报异步执行，避免 HTTP 或本地写图阻塞推理线程。
alarm_queue = queue.Queue(maxsize=ALARM_QUEUE_MAXSIZE)


# 模型类别到中文业务名称的映射，同时用于构造告警标题和告警正文。
LABEL_MAP = {
    "person": "人员",
    "advertising": "广告",
    "no_helmet": "未戴安全帽",
    "helmet": "佩戴安全帽",
    "ladder": "移动的梯子",
    "fire": "火焰",
    "full_trash_can": "满溢的垃圾桶",
    "rubbish_bag": "塑料垃圾袋",
    "not_full_trash_can": "不满的垃圾桶",
    "muck_pile": "渣土堆",
    "car": "小汽车",
    "bike": "自行车",
    "moto": "摩托车",
    "tricycle": "三轮车",
    "bus": "公共汽车",
    "truck": "卡车",
    "box": "箱子",
    "sleep": "睡岗-趴着",
    "lying": "睡岗-平躺",
    "billbard": "广告牌",
    "railing": "栏杆",
    "awning": "遮阳伞伞蓬",
    "frame": "方形塑料筐",
    "phone_call": "打电话",
    "playing_phone": "玩手机",
    "plastic_bottle": "塑料瓶",
    "Brick_pile": "砖头堆",
    "table": "桌子",
    "smoke": "吸烟",
    "hang_clothes": "悬挂衣服",
    "banner": "横幅广告",
    "cyjc": "超员检测",
    "lgjc": "离岗检测",
    "wgtc": "违规停车",
    "wxjc": "危险检测",
}


def str_to_bool(value):
    """兼容命令行中 true/false、1/0、yes/no 等布尔写法。"""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def parse_args(mode):
    """解析 run.py/run_local.py 共用参数，并注入入口模式。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--msn", type=str, required=True, help="模型编码")
    parser.add_argument("--ins", type=str, required=True, help="模型实例编码")
    parser.add_argument("--secret", type=str, required=True, help="模型密钥")
    parser.add_argument("--taskId", type=str, required=True, help="任务id")
    parser.add_argument("--videoId", type=str, required=True, help="视频id")
    parser.add_argument("--videoUrl", type=str, required=True, help="视频地址")
    parser.add_argument("--server", type=str, required=True, help="服务器地址")
    parser.add_argument("--report_time", type=int, default=300, help="状态上报时间间隔，单位秒")

    parser.add_argument("--detect_model", type=str, default="frames", help="检测模式")
    parser.add_argument("--detect_area", type=str, required=False, help="检测区域")
    parser.add_argument("--detect_target", type=str, required=True, help="检测目标")
    parser.add_argument("--detect_thresh", type=float, default=0.66, help="置信度")
    parser.add_argument("--detect_frames", type=int, default=15, help="抽帧数量")

    parser.add_argument("--rqjc", type=str_to_bool, default=False, help="入侵检测")
    parser.add_argument("--cyjc", type=str_to_bool, default=False, help="超员检测")
    parser.add_argument("--cysl", type=int, default=10, help="超员数量")
    parser.add_argument("--lgjc", type=str_to_bool, default=False, help="离岗检测")
    parser.add_argument("--sgxw", type=str_to_bool, default=False, help="施工行为")
    parser.add_argument("--wxjc", type=str_to_bool, default=False, help="危险监测")
    parser.add_argument("--fire", type=str_to_bool, default=False, help="火情监测")

    parser.add_argument("--detect_fps", type=float, default=1.0, help="每秒检测次数")
    parser.add_argument("--alarm_cooldown", type=int, default=10, help="同一告警最小间隔，单位秒")
    parser.add_argument("--min_hit_count", type=int, default=2, help="连续命中才算告警")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="常规模型权重路径")
    parser.add_argument("--fire_weights", type=str, default=DEFAULT_FIRE_WEIGHTS, help="火情模型权重路径")

    args = parser.parse_args()
    args.mode = mode
    args.detect_target = json.loads(args.detect_target)
    args.detect_area = json.loads(args.detect_area) if args.detect_area else []
    # 传入 ROI 区域时默认开启入侵检测，兼容旧调用方式。
    args.rqjc = bool(args.detect_area) or args.rqjc
    try:
        validate_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_config(args):
    """启动前校验关键参数，避免运行中才暴露除零、空 ROI 等错误。"""
    if args.detect_fps <= 0:
        raise ValueError("--detect_fps 必须大于 0")
    if args.min_hit_count < 1:
        raise ValueError("--min_hit_count 必须大于等于 1")
    if args.alarm_cooldown < 0:
        raise ValueError("--alarm_cooldown 必须大于等于 0")
    if not isinstance(args.detect_target, list):
        raise ValueError("--detect_target 必须是 JSON 数组，例如 '[0]'")
    if (args.rqjc or args.lgjc) and len(args.detect_area) < 3:
        raise ValueError("--rqjc/--lgjc 需要至少 3 个 ROI 点，例如 '[[0,0],[100,0],[100,100]]'")
    for point in args.detect_area:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("--detect_area 中每个点必须是 [x,y] 格式")


def cal_area_2poly(data1, data2):
    """计算目标框多边形与 ROI 多边形的相交面积。"""
    from shapely.geometry import Polygon

    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull
    if not poly1.intersects(poly2):
        return 0
    return poly1.intersection(poly2).area


def judge_illegal_entry(boxes, roi_area):
    """判断每个检测框是否与 ROI 区域相交。"""
    entry_objects_filter = [False] * len(boxes)
    for i, box in enumerate(boxes):
        vertices_4points = [
            [box[0], box[1]],
            [box[0], box[1] + box[3]],
            [box[0] + box[2], box[1] + box[3]],
            [box[0] + box[2], box[1]],
        ]
        inter_area = cal_area_2poly(vertices_4points, roi_area)
        entry_objects_filter[i] = inter_area > 0
    return entry_objects_filter


def load_model(weights, imgsz=(640, 640), device="", half=False):
    """加载 PyTorch/导出格式模型，并返回模型、设备和类别名称。"""
    import torch

    from models.experimental import attempt_load
    from utils.general import check_suffix
    from utils.torch_utils import load_classifier, select_device

    device1 = select_device(device)
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix = False, Path(w).suffix.lower()
    suffixes = [".pt", ".onnx", ".tflite", ".pb", ""]
    check_suffix(w, suffixes)
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)
    stride, names = 64, [f"class{i}" for i in range(1000)]
    if pt:
        model = torch.jit.load(w) if "torchscript" in w else attempt_load(weights, map_location=device1)
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, "module") else model.names
        if half:
            model.half()
        if classify:
            modelc = load_classifier(name="resnet50", n=2)
            modelc.load_state_dict(torch.load("resnet50.pt", map_location=device1)["model"]).to(device1).eval()
    if pt and device1.type != "cpu":
        model(torch.zeros(1, 3, *imgsz).to(device1).type_as(next(model.parameters())))
    return model, device1, names


def detect(img, model, device, names, conf_thres=0.6, iou_thres=0.45, max_det=1000, classes=None,
           agnostic_nms=False, augment=False, visualize=False, half=False, dnn=False):
    """执行单帧 YOLO 推理，并套用业务检测规则生成告警标签。"""
    import cv2
    import numpy as np
    import torch

    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords
    from utils.plots import Annotator, colors

    im0 = img
    im = letterbox(im0, 640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]

    pred = model(im, augment=augment, visualize=visualize)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    detections = []
    im0r = im0c = img.copy()
    # 入侵和离岗当前共用同一组 ROI；后续如需拆分，应从参数层明确新增字段。
    rq_roi_area = np.array(config.detect_area).astype("int")
    lg_roi_area = np.array(config.detect_area).astype("int")
    person_count = 0

    for _, det in enumerate(pred):
        annotator = Annotator(im0c, line_width=3, example=str(names))
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"

                if config.rqjc and not config.lgjc:
                    rq_result = judge_illegal_entry([xyxy], rq_roi_area)
                    if True in rq_result:
                        cv2.polylines(im0c, [np.array(rq_roi_area)], True, (0, 0, 255), 3)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        detections.insert(0, f"rqjc:{names[c]}")
                    else:
                        continue

                if names[c] == "person":
                    if config.cyjc:
                        person_count += 1
                        label = f"{names[c]} [{person_count}] {conf:.2f}"
                    elif config.lgjc:
                        lg_result = judge_illegal_entry([xyxy], lg_roi_area)
                        if True not in lg_result:
                            cv2.polylines(im0c, [np.array(lg_roi_area)], True, (0, 0, 255), 3)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            detections.insert(0, "lgjc")
                        else:
                            continue

                annotator.box_label(xyxy, label, color=colors(c, True))
                detections.append(names[c])

            if config.cyjc:
                if person_count >= config.cysl:
                    detections.insert(0, f"cyjc:{person_count}")
                else:
                    return [], annotator.result()

            if config.sgxw:
                sgxw_helmet = "helmet" in detections
                sgxw_other = any(dete in ("muck_pile", "railing", "Brick_pile") for dete in detections)
                if sgxw_helmet and sgxw_other:
                    detections.insert(0, "sgxw")
                    return list(set(detections)), annotator.result()
                return [], annotator.result()

            if config.wxjc:
                if "lying" in detections:
                    detections.insert(0, "wxjc")
                    return list(set(detections)), annotator.result()
                return [], annotator.result()

        im0r = annotator.result()
    return list(set(detections)), im0r


def build_warn_info(result):
    """将检测结果转换为后端告警接口需要的字段。"""
    info = []
    wtype = result[0]
    wtitle = "预警:未知类型"
    if wtype in LABEL_MAP:
        wtitle = "预警:" + LABEL_MAP[wtype]
    if config.fire:
        if wtype == "fire":
            wtitle = "预警:火灾报警"
        elif wtype == "smoke":
            wtitle = "预警:烟雾报警"

    for i in result:
        if i.startswith("rqjc"):
            wtype = "rqjc"
            cname = i.split(":")[1]
            if cname == "person":
                wtype = "ryrq"
                wtitle = "人员入侵"
                info = ["人员入侵"]
                break
            if cname in ("car", "bike", "moto", "bus", "truck", "tricycle"):
                wtype = "wgtc"
                wtitle = "违规停车"
                info = ["违规停车"]
                break
            info.append(LABEL_MAP.get(i, i))
        elif i.startswith("cyjc"):
            wtype = "cyjc"
            wtitle = "超员检测"
            pcount = i.split(":")[1]
            info = [f"人员超员,限定人数{config.cysl}，当前人数{pcount}"]
            break
        elif i == "lgjc":
            wtype = "lgjc"
            wtitle = "离岗检测"
            info = ["人员离岗"]
            break
        elif i == "sgxw":
            wtype = "sgxw"
            wtitle = "施工行为"
            info = ["施工行为"]
            break
        elif i == "wxjc":
            wtype = "wxjc"
            wtitle = "危险检测"
            info = ["危险行为"]
            break
        elif config.fire and i in ("fire", "smoke"):
            info.append("火灾" if i == "fire" else "烟雾")
            break
        else:
            info.append(LABEL_MAP.get(i, i))

    return {
        "ins": config.ins,
        "msn": config.msn,
        "taskId": config.taskId,
        "wtype": wtype,
        "wtitle": wtitle,
        "wlevel": 1,
        "wtext": f"预警消息，识别到:{','.join(info)}",
        "wtags": f"{','.join(result)}",
    }


def api_signature():
    """生成后端接口签名头：MD5(模型编码+实例编码+模型密钥+时间戳+随机数)。"""
    timestamp = str(int(time.time()))
    random_num = str(random.randint(100000, 999999))
    sign_str = f"{config.msn}{config.ins}{config.secret}{timestamp}{random_num}"
    sign = base64.b64encode(hashlib.md5(sign_str.encode("utf-8")).digest()).decode("utf-8")
    return {
        "signature": sign,
        "timestamp": timestamp,
        "nonce": random_num,
    }


def report_task(status, data):
    """上报任务状态；local 模式只记录日志，不发真实请求。"""
    url = f"{config.server}/aimp-edge/oapi/task/report/{config.ins}"
    payload = {
        "taskId": config.taskId,
        "status": status,
        "data": f"{data}",
    }
    if config.mode == "local":
        logger.info("本地模式跳过任务状态上报，url=%s,data=%s", url, payload)
        return
    import requests

    try:
        headers = api_signature()
        headers["Content-Type"] = "application/json"
        logger.info("任务状态上报开始，url=%s,headers=%s,data=%s", url, headers, payload)
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        logger.info("任务状态上报结束，url=%s,result=%s", url, resp.json())
    except Exception as e:
        log_network_exception("task_report", "任务状态上报失败，url=%s,data=%s,异常=%s", url, payload, e)


def capture_loop(cap_url):
    """视频采集线程：持续读取视频源，只把最新帧放入队列。"""
    import cv2

    logger.info("启动视频采集线程，地址: %s", cap_url)
    cap = cv2.VideoCapture(cap_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fail_count = 0
    reconnect_delay = RTSP_RECONNECT_BASE_DELAY
    reconnect_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            logger.warning("读取失败 %s 次", fail_count)
            if fail_count >= 10:
                reconnect_count += 1
                logger.warning(
                    "读取视频流连续失败，开始第 %s 次重连，等待 %s 秒",
                    reconnect_count,
                    reconnect_delay,
                )
                cap.release()
                time.sleep(reconnect_delay)
                cap = cv2.VideoCapture(cap_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fail_count = 0
                if cap.isOpened():
                    logger.info("视频流重连成功")
                    reconnect_delay = RTSP_RECONNECT_BASE_DELAY
                else:
                    logger.warning("视频流重连后仍未打开")
                    reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)
            continue

        fail_count = 0
        reconnect_delay = RTSP_RECONNECT_BASE_DELAY
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)


def detect_loop(model, device, names, thresh, detect_goal):
    """推理线程：按 detect_fps 节流，执行去抖和告警冷却。"""
    last_detect_time = 0
    hit_counter = 0
    last_alarm_time = 0

    while True:
        frame = frame_queue.get()
        now = time.time()
        if now - last_detect_time < 1.0 / config.detect_fps:
            continue
        last_detect_time = now
        logger.info("获取帧，开始检测")

        result, img = detect(
            img=frame,
            model=model,
            device=device,
            names=names,
            conf_thres=thresh,
            classes=detect_goal,
        )

        if result:
            hit_counter += 1
        else:
            hit_counter = 0

        if hit_counter >= config.min_hit_count:
            if now - last_alarm_time >= config.alarm_cooldown:
                put_alarm_latest(result, img)
                last_alarm_time = now
                hit_counter = 0
            else:
                logger.info("不满足告警最小间隔")
        elif result:
            logger.info("检测命中但未达到连续命中阈值，去抖")


def alarm_loop():
    """告警处理线程：根据运行模式选择真实上报或本地保存。"""
    while True:
        result, img = alarm_queue.get()
        if config.mode == "local":
            save_local_alarm(result, img)
        else:
            post_alarm(result, img)


def put_alarm_latest(result, img):
    """非阻塞写入告警队列；队列满时丢弃最旧告警，避免网络异常拖慢推理。"""
    item = (result, img)
    try:
        alarm_queue.put_nowait(item)
        return
    except queue.Full:
        try:
            alarm_queue.get_nowait()
            logger.warning("告警队列已满，丢弃最旧告警后写入新告警")
        except queue.Empty:
            pass
    try:
        alarm_queue.put_nowait(item)
    except queue.Full:
        logger.warning("告警队列仍然满，丢弃当前告警")


def save_local_alarm(result, img):
    """local 模式告警处理：保存带标注的告警图片。"""
    import cv2

    logger.info("获取告警结果，开始本地处理，result=%s", result)
    ALARM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = ALARM_IMAGE_DIR / f"{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(file_path), img)
    logger.info("本地告警图片已保存，path=%s", file_path)


def post_alarm(result, img):
    """online 模式告警处理：向后端提交告警字段和图片。"""
    import cv2
    import requests

    logger.info("获取告警结果，开始上报，result=%s", result)
    if not config.server:
        logger.warning("server 为空，跳过告警上报")
        return

    url = f"{config.server}/aimp-edge/oapi/warn/{config.ins}"
    data = build_warn_info(result)
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            with open(tmp.name, "rb") as image_file:
                files = {"images": image_file}
                headers = api_signature()
                resp = requests.post(url, data=data, files=files, headers=headers, timeout=5)
        logger.info("告警上报成功，status_code=%s", resp.status_code)
    except Exception as e:
        log_network_exception("alarm_post", "告警上报失败: %s", e)


_last_network_error_log_time = {}


def log_network_exception(key, message, *args):
    """网络异常日志降频，避免后端不可用时刷屏。"""
    now = time.time()
    last_time = _last_network_error_log_time.get(key, 0)
    if now - last_time >= NETWORK_ERROR_LOG_INTERVAL:
        _last_network_error_log_time[key] = now
        logger.exception(message, *args)
    else:
        logger.warning("网络异常仍在持续，已降频输出，key=%s", key)


def select_model_path():
    """根据是否火情检测选择权重路径。"""
    return config.fire_weights if config.fire else config.weights


def run_video():
    """启动模型加载、采集线程、推理线程、告警线程和任务心跳。"""
    model_path = select_model_path()
    logger.info("加载模型，path=%s", model_path)
    model, device, names = load_model(weights=model_path)

    report_task(1, "任务启动")
    threading.Thread(target=capture_loop, args=(config.videoUrl,), daemon=True, name="capture").start()
    threading.Thread(
        target=detect_loop,
        args=(model, device, names, config.detect_thresh, config.detect_target),
        daemon=True,
        name="detect",
    ).start()
    threading.Thread(target=alarm_loop, daemon=True, name="alarm").start()

    while True:
        report_task(1, "任务运行中")
        time.sleep(config.report_time)


def main(mode):
    """入口函数，由 run.py/run_local.py 传入 online 或 local。"""
    global config, frame_queue, alarm_queue
    frame_queue = queue.Queue(maxsize=1)
    alarm_queue = queue.Queue(maxsize=10)
    config = parse_args(mode)
    logger.info("运行模式: %s", config.mode)
    logger.info("视频地址: %s", config.videoUrl)
    logger.info("服务地址: %s", config.server)
    logger.info("参数信息: %s", config)
    logger.info("检测目标: %s", config.detect_target)
    run_video()
