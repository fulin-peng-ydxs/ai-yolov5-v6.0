# YOLOv5 by Ultralytics, GPL-3.0 license

# 使用 NVIDIA PyTorch 基础镜像，镜像内已包含 Python、CUDA、PyTorch 等运行环境
FROM nvcr.io/nvidia/pytorch:21.05-py3
# 如需升级到新 CUDA/PyTorch 版本，可评估后切换到其他基础镜像
# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 设置 DEBIAN_FRONTEND 环境变量为 noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖：OpenCV 图像显示/解码、FFmpeg 视频流处理、常用排障工具
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx ffmpeg

# 设置容器时区为 Asia/Shanghai
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata
    
# 安装 Python 依赖
COPY requirements.txt .
RUN python -m pip install --upgrade pip
# NVIDIA 镜像自带的 tensorboard 插件可能与项目依赖冲突，先卸载再安装 requirements.txt
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 导出模型或实验追踪时才需要安装以下可选工具
# RUN pip install --no-cache coremltools onnx gsutil notebook "wandb>=0.12.2" -i https://pypi.tuna.tsinghua.edu.cn/simple
# PyTorch/torchvision 由基础镜像提供，默认不要在 requirements.txt 或 Dockerfile 中重复安装
# RUN pip install --no-cache -U torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install --no-cache torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 创建工作目录
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# 拷贝项目文件并授予启动脚本执行权限
COPY . /usr/src/app
RUN chmod 777 /usr/src/app/run.sh

# 下载 YOLOv5 绘图字体到用户配置目录
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/


ENTRYPOINT ["/usr/src/app/run.sh"]

# 如需指定 HOME，可按需打开
# ENV HOME=/usr/src/app


# 使用示例 -------------------------------------------------------------------------------------------------------------

# 构建并推送镜像
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# 拉取并运行镜像
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# 挂载本地数据集目录运行镜像
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# 停止所有运行中的容器
# sudo docker kill $(sudo docker ps -q)

# 停止基于指定镜像创建的所有容器
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# 进入运行中的容器
# sudo docker exec -it 5a9b5863d93d bash

# 启动并进入已停止的容器
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# 清理 Docker 缓存、镜像和卷
# docker system prune -a --volumes

# Ubuntu 驱动安装参考
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP 多卡训练测试命令
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3
