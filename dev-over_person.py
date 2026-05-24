import cv2
import torch
import numpy as np

# ------------------ 配置区域与阈值 ------------------ #
# 定义一个四边形区域（以像素坐标为准）
REGION_POLYGON = np.array([[100, 200], [400, 200], [400, 600], [100, 600]])

# 超员阈值：超过此值报警
THRESHOLD = 3

# ------------------ 判断是否在区域内 ------------------ #
def is_inside_region(bbox, region_polygon):
    """
    判断目标框的中心点是否在指定区域内
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    # OpenCV 的函数判断点是否在多边形内
    return cv2.pointPolygonTest(region_polygon, (cx, cy), False) >= 0

# ------------------ 人员计数逻辑 ------------------ #
def count_people_in_region(detections, region_polygon, threshold):
    """
    统计区域内的人数，并判断是否超员
    """
    count = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # YOLOv5 中 class=0 通常为人
            if is_inside_region([x1, y1, x2, y2], region_polygon):
                count += 1
    return count, count > threshold

# ------------------ 主流程 ------------------ #
def main(video_path):
    # 加载模型
    model = torch.hub.load('./', 'custom',path='runs/train/all/weights/best.pt' ,source='local')  # 也可换成 yolov5m, yolov5l
    model.conf = 0.25  # 置信度阈值
    model.iou = 0.45   # NMS 阈值
    model.classes = [0]  # 只检测人类

    cap = cv2.VideoCapture(video_path)

    frame_index = 0  # 当前帧数
    skip_frames = 5  # 每 5 帧检测一次
    last_detections = []  # 上次的检测结果

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每 N 帧运行一次检测
        if frame_index % skip_frames == 0:
            results = model(frame)
            last_detections = results.xyxy[0].cpu().numpy()

        detections = last_detections  # 使用当前或上一帧的检测结果
        count, overloaded = count_people_in_region(detections, REGION_POLYGON, THRESHOLD)
        frame_index += 1  # 帧编号加一

        # 可视化检测框
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                color = (0, 0, 255) if is_inside_region([x1, y1, x2, y2], REGION_POLYGON) else (255, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)

        # 可视化区域和人数
        cv2.polylines(frame, [REGION_POLYGON], isClosed=True, color=(0, 255, 255), thickness=2)
        color = (0, 0, 255) if overloaded else (0, 255, 0)
        status = "ALERT" if overloaded else "SAFE"
        cv2.putText(frame, f'Count: {count} | {status}', (REGION_POLYGON[0][0], REGION_POLYGON[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Overcrowd Detector", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ 执行入口 ------------------ #
if __name__ == "__main__":
    # 可替换为你的本地视频路径或摄像头索引 0
    video_source = "录屏2025-06-20 15.40.13.mov"
    main(video_source)