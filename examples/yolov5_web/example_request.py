# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

"""
Perform test request
"""
'''
	•	cv2: 用于图像处理（读取、编码、解码等）。
	•	numpy: 处理数组和缓冲区数据。
	•	matplotlib.pyplot: 用于可视化图像（将结果显示出来）。
	•	requests: 用于向 YOLOv5 Flask API 发送 HTTP 请求。
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
'''
	•	DETECTION_URL: Flask API 的端点 URL，用于接收 POST 请求，进行目标检测。
	•	IMAGE: 待检测图像的本地文件路径。
'''
DETECTION_URL = 'http://localhost:5110/v1/object-detection/yolov5s'
IMAGE = 'data/images/zidane.jpg'
# Read image
# with open(IMAGE, 'rb') as f:
#     image_data = f.read()
'''	•	使用 cv2.imread() 读取图像文件，返回一个 NumPy 数组，格式为 BGR。'''
img = cv2.imread(IMAGE)
'''	•	将图像从 BGR 转换为 RGB，因为 matplotlib 和一些视觉工具默认使用 RGB。'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
'''
	•	使用 cv2.imencode() 将图像编码为 .jpg 格式。
	•	[1] 获取编码后的字节流。
	•	.tobytes() 转换为字节形式，用于通过 HTTP POST 发送。
'''
img = cv2.imencode(".jpg", img)[1].tobytes()
'''
	•	发送 POST 请求到 Flask API：
	•	DETECTION_URL 是目标 URL。
	•	data=img 是请求体，包含字节形式的图像数据。
	•	response: Flask API 返回的响应，通常包含检测后的结果图像。
'''
response = requests.post(DETECTION_URL, data=img)
'''
	•	response.content: API 返回的二进制数据（JPEG 格式的图像）。
	•	np.frombuffer(): 将二进制数据转换为 NumPy 数组。
	•	cv2.imdecode(): 将数组解码为图像格式。
'''
img = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
'''
	•	使用 matplotlib 显示解码后的图像。
	•	plt.imshow(img): 将图像加载到图形窗口。
	•	plt.show(): 显示窗口。
'''
plt.imshow(img)
plt.show()