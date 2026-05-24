import argparse
from pathlib import Path

# OpenCV 库，用于视频处理
import cv2


# 抽取视频帧方法
def extract_frames(video_path, output_dir, frame_interval):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise SystemExit(f"无法打开视频文件: {video_path}")

    frame_count = 0  # 帧计数器
    saved_count = 0  # 保存帧计数器

    while True:
        ret, frame = cap.read()
        if not ret:  # 视频读取结束
            break

        # 判断是否为需要保存的帧
        if frame_count % frame_interval == 0:
            output_file = output_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(output_file), frame)
            saved_count += 1
            print(f"保存帧: {output_file}")

        frame_count += 1

    # 释放资源
    cap.release()
    print(f"抽帧完成，共保存了 {saved_count} 帧到 {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video by fixed frame interval.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output image directory.")
    parser.add_argument("--interval", type=int, default=20, help="Save one frame every N frames.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.output, args.interval)
