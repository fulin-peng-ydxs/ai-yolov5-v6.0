import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def select_roi(image_path, output_path=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    points = []
    window_name = "select_roi"

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"add point: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.polylines(image, [np.array(points)], False, (0, 255, 0), 2)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)

    print("Click ROI points in order, then press q or Esc to finish.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    print("Selected ROI points:")
    print(points)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
        print(f"Saved ROI points to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Select polygon ROI points on an image.")
    parser.add_argument("--image", required=True, help="Image path used as ROI background.")
    parser.add_argument("--output", help="Optional JSON output path for selected points.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    select_roi(Path(args.image), Path(args.output) if args.output else None)
