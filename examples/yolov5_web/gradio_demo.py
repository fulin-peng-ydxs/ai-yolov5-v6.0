"""
    •	torch：PyTorch 深度学习框架，用于加载 YOLOv5 模型。
	•	gradio：一个快速构建交互式机器学习界面的 Python 库。
"""
import argparse
from pathlib import Path

import torch
import gradio as gr

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Gradio YOLOv5 demo.")
    parser.add_argument("--weights", default=str(ROOT / "yolov5s.pt"), help="Model weights path.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


args = parse_args()

'''	•	torch.hub.load：
	•	从本地目录加载自定义的 YOLOv5 模型。
	•	参数解释：
	•	"./"：表示当前目录。
	•	"custom"：指定自定义模型。
	•	path="runs/train/exp/weights/best.pt"：YOLOv5 模型权重文件路径。
	•	source="local"：表明从本地加载，而非从在线库加载。
'''
model = torch.hub.load(str(ROOT), "custom", path=args.weights, source="local")
'''
	•	title：界面标题。
	•	desc：界面描述，说明项目用途或功能。
'''
title = "基于Gradio的YOLOv5演示项目"
desc = "这是一个基于Gradio的YOLOv5演示项目，非常简洁，非常方便！"
'''
	•	base_conf：默认置信度阈值，控制检测框的可信度。
	•	base_iou：默认 IoU（交并比）阈值，控制边界框的重叠程度。
'''
base_conf, base_iou = 0.25, 0.45
'''
	•	det_image：处理图像的函数。
	•	输入参数：
	•	img：上传的图像。
	•	conf_thres：置信度阈值。
	•	iou_thres：IoU 阈值。
	•	内部逻辑：
	1.	设置模型的置信度和 IoU 阈值。
	2.	使用模型处理图像。
	3.	返回渲染后的检测结果。
'''
def det_image(img, conf_thres, iou_thres):
    model.conf = conf_thres
    model.iou = iou_thres
    return model(img).render()[0]
'''
	•	inputs：定义输入组件。
	1.	"image"：上传图像组件。
	2.	gr.Slider：滑块组件，允许用户动态调整 conf_thres 和 iou_thres。
	•	minimum：滑块最小值。
	•	maximum：滑块最大值。
	•	value：滑块默认值。
	•	outputs：定义输出组件。
	•	"image"：输出检测后的图像。
	•	fn：指定处理逻辑的函数，这里是 det_image。
	•	title 和 description：设置界面的标题和描述。
	•	live=True：开启实时模式，允许用户拖动滑块时实时更新结果。
	•	examples：
	•	提供预设的示例输入，包括：
	•	图像路径。
	•	默认置信度和 IoU 阈值。
	•	.launch(share=True)：
	•	启动 Gradio 界面。
	•	share=True：生成一个可以共享的临时公网链接。
'''
gr.Interface(inputs=["image", gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
             outputs=["image"],
             fn=det_image,
             title=title,
             description=desc,
             live=True,
             examples=[[str(ROOT / "data/train/bvn/images/train/frame_0000.jpg"), base_conf, base_iou],
                       [str(ROOT / "data/train/bvn/images/train/frame_0030.jpg"), 0.3, base_iou]]).launch(share=args.share)
