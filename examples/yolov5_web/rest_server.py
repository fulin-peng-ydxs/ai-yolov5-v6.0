# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

"""
	•	argparse：处理命令行参数。
	•	io：用于处理输入/输出字节流。
	•	numpy：处理数组操作（如图像解码）。
	•	cv2：OpenCV，用于图像处理。
	•	torch：PyTorch，用于加载 YOLOv5 模型和进行推理。
	•	flask：轻量级 Web 框架，用于实现 REST API。
	•	PIL.Image：用于处理图像文件的格式转换。
"""
import argparse
import io
from pathlib import Path

import numpy as np
import cv2
import torch
from flask import Flask, request
from PIL import Image

'''
	•	app = Flask(__name__)：初始化 Flask 应用。
	•	models = {}：用于存储加载的 YOLOv5 模型。
	•	DETECTION_URL：定义目标检测的 URL 路径，<model> 是动态部分，代表具体模型名称。
'''
app = Flask(__name__)
models = {}
DETECTION_URL = '/v1/object-detection/<model>'
ROOT = Path(__file__).resolve().parents[2]

'''
	•	绑定了一个动态 URL，例如 /v1/object-detection/yolov5s。
	•	限制请求方法为 POST，即只能通过 POST 请求发送数据。
'''
@app.route(DETECTION_URL, methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    if request.data: #读取数据流
        '''
        	•	request.data：从请求中读取二进制图像数据。
            •	cv2.imdecode()：将二进制数据解码为 OpenCV 图像。
            •	如果指定的模型 model 存在于加载的模型字典 models 中：
                1.	使用模型进行推理。
                2.	调用 results.render() 绘制预测结果。
                3.	使用 cv2.imencode() 将结果编码为 JPEG 格式并返回。
        '''
        img = cv2.imdecode(np.frombuffer(request.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if model in models:
            results = models[model](img)  # reduce size=320 for faster inference
            results = results.render()[0]
            return cv2.imencode(".jpg", results)[1].tobytes()

    if request.files.get('image'): #读取文件
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))
        '''
        	•	request.files['image']：从表单中获取上传的文件。
            •	Image.open()：使用 PIL 加载图像。
            •	如果模型存在：
                1.	使用模型进行推理（默认输入大小为 640）。
                2.	返回预测结果的边界框信息，格式为 JSON。
        '''
        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient='records')


if __name__ == '__main__':
    '''
    	•	命令行参数：
            •	--port：指定服务器运行端口，默认值为 5000。
            •	--model：指定加载的模型名称（可以是多个）。
    '''
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()
    '''
    	•	遍历用户指定的模型名称。
	    •	使用 torch.hub.load 从本地路径加载模型，并存储到 models 字典中，方便后续调用。
    '''
    for m in opt.model:
        models[m] = torch.hub.load(str(ROOT), m, source="local")
    '''
    	•	host='0.0.0.0'：让服务监听所有网络接口。
	    •	port=opt.port：服务端口由用户通过命令行参数指定。
    '''
    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat
