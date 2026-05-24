import argparse
from pathlib import Path

import cv2


def extract_one_frame(video_path, output_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read frame {frame_index} from: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Saved frame {frame_index} to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract one frame from a video.")
    parser.add_argument("--video", required=True, help="Input video path or stream URL.")
    parser.add_argument("--output", default="sample_frame.jpg", help="Output image path.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to extract, default 0.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_one_frame(Path(args.video), Path(args.output), args.frame)
