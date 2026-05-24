# YOLOv5 视觉检测项目

本项目基于 YOLOv5 v6.0，增加了业务检测、视频流推理、告警上报、ROI 区域判断、数据集配置和若干数据处理工具。

## 目录说明

```text
.
├── data/                    # 数据集配置、少量示例图、训练数据入口
│   ├── *.yaml               # YOLO 数据集配置，如 all.yaml、moke.yaml、bvn.yaml、fire_smoke.yaml
│   ├── hyps/                # 训练超参数配置，如 hyp.smoke.yaml
│   ├── images/              # YOLO 官方示例图，保留 bus.jpg、zidane.jpg
│   ├── raw/                 # 原始抽帧素材、未标注素材，默认不入库
│   └── train/               # YOLO 格式训练集目录，如 moke、bvn、fire_smoke、moke_black
├── docs/
│   └── emmx/                # 从 python-yolo 迁移的知识笔记/脑图文档
├── examples/                # 从 python-yolo 迁移的 Web/UI 示例，不作为主入口
│   ├── yolov5_web/          # Flask、Gradio、请求示例
│   └── yolov5_ui/           # PySide6 GUI 示例
├── legacy/                  # 历史脚本归档，非当前推荐入口
│   ├── config.yaml          # detect1106.py/detect_new.py 旧版配置
│   ├── detect1106.py        # 旧版多路视频抽帧检测脚本
│   └── detect_new.py        # 旧版单路/本地抽帧检测脚本
├── models/                  # YOLOv5 网络结构和项目定制模型 yaml
├── outputs/                 # 运行输出、调试图片、告警截图，默认不入库
│   ├── alarm_images/        # run_local.py 本地告警图片输出
│   └── debug/root_media/    # 从根目录归档的临时图片、视频、截图
├── runs/                    # 训练、推理输出目录，默认不再新增入库
│   ├── detect/              # detect.py 推理结果
│   ├── train/               # train.py 训练结果、权重和指标
│   └── models/              # 部署模型归档，建议只保留必要说明或少量交付权重
├── tools/                   # 数据处理和标注辅助工具
│   ├── extract_one_frame.py # 从视频抽一帧
│   ├── select_roi.py        # 鼠标点选 ROI 多边形
│   └── video_to_img.py      # 按固定帧间隔批量抽帧
├── utils/                   # YOLOv5 工具代码
├── business_runtime.py      # run.py/run_local.py 共享运行机制
├── detect.py                # YOLOv5 官方通用推理入口
├── run.py                   # 业务线上推理入口，含任务上报、告警上报
├── run_local.py             # 业务本地调试入口，上报逻辑做了本地化处理
├── train.py                 # YOLOv5 训练入口
├── val.py                   # 模型验证入口
└── export.py                # 模型导出入口
```

## 文件管理约定

已在 `.gitignore` 中补充以下规则：

- `runs/`：训练和推理输出默认不新增入库。
- `data/images/`：只保留已有官方示例图 `bus.jpg`、`zidane.jpg`；运行时告警截图统一写入 `outputs/alarm_images/`。
- `data/raw/`：原始抽帧素材、未标注素材默认不入库。
- `outputs/`：运行输出、调试图片、告警截图默认不入库。
- `examples/`：仅保存示例应用和实验入口，不作为线上推荐入口；生产推理仍以 `run.py`、`run_local.py`、`detect.py` 为准。
- `docs/emmx/`：保存迁移过来的知识笔记类文档，不参与训练和推理运行。
- `save_img/`、根目录图片、视频、`.vscode/`：视为历史或本地调试产物，不再新增使用。

根目录历史调试图片、视频已归档到 `outputs/debug/root_media/`；历史 `save_img` 图片已归档到 `outputs/alarm_images/legacy/`；原 `train_data_source` 已归档到 `data/raw/train_data_source/`。这些目录默认不入库。

如果确实需要提交样例图片，建议新建 `data/samples/`，并在 README 中说明用途。正式训练集、原始素材和权重建议放对象存储、NAS、制品库或 DVC，不建议直接放 Git。

`legacy/` 中的 `detect1106.py`、`detect_new.py`、`config.yaml` 只用于参考旧版抽帧、ROI 和接口上报逻辑，不作为当前推荐运行入口。

## 环境准备

推荐运行环境以 Docker/服务器为准：

- Docker 基础镜像：`nvcr.io/nvidia/pytorch:21.05-py3`
- Python：3.8，由基础镜像提供
- PyTorch：由基础镜像提供，项目 `requirements.txt` 不重复安装 `torch/torchvision`

Docker 构建：

```bash
docker build -t aimp.video.yolov5:latest .
```

Docker 运行示例：

```bash
docker run -it --gpus all --net=host aimp.video.yolov5:latest <run.py 参数>
```

本机调试建议使用 Python 3.8 虚拟环境。当前项目不以 Python 3.12 作为依赖基线：

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要本机 GPU 推理，请按本机 CUDA 版本单独安装匹配的 `torch/torchvision`。

版本说明：

- `numpy==1.20.3`、`Pillow==9.0.0` 与当前 Docker 镜像对齐。
- `protobuf==3.20.0` 用于规避高版本 protobuf 与旧训练/日志组件的兼容问题。
- `pandas` 是 YOLOv5 结果表格、训练结果读取和绘图逻辑的实际依赖。
- `coremltools`、`onnx`、`notebook`、`wandb` 等导出/实验工具不是默认运行依赖，需要时单独安装。

## 数据集结构

YOLO 数据集建议保持如下结构：

```text
data/train/<dataset_name>/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

当前已有和迁移后的数据集配置：

| 数据集 | 配置文件 | 数据目录 | 类别 |
|---|---|---|---|
| moke | `data/moke.yaml` | `data/train/moke` | `moke` |
| bvn | `data/bvn.yaml` | `data/train/bvn` | `daitu`、`minren` |
| fire_smoke | `data/fire_smoke.yaml` | `data/train/fire_smoke` | `fire`、`smoke`、`smoker` |
| moke_black | `data/moke_black.yaml` | `data/train/moke_black` | `moke` |

`moke_black` 来自迁移数据，暂不与当前 `moke` 数据集合并，避免不同来源数据和标签分布混在一起。

`moke` 数据集配置示例：

```yaml
path: data/train/moke
train: images/train
val: images/val
nc: 1
names: ['moke']
```

对应训练命令使用 `--data data/moke.yaml`。其他数据集按表格中的配置文件通过 `--data` 显式指定。

## 训练

通用训练入口是 `train.py`。常用命令：

```bash
python train.py \
  --img 640 \
  --batch-size 16 \
  --epochs 100 \
  --data data/moke.yaml \
  --cfg models/yolov5s.yaml \
  --weights yolov5s.pt \
  --name moke
```

说明：

- `--data`：数据集配置文件。
- `--cfg`：模型结构配置。小模型可用 `models/yolov5n.yaml` 或 `models/yolov5s.yaml`。
- `--weights`：预训练权重。首次训练常用 `yolov5s.pt`；从零训练可传空字符串。
- `--name`：输出目录名，结果默认写入 `runs/train/<name>/`。

恢复训练：

```bash
python train.py --resume runs/train/moke/weights/last.pt
```

验证模型：

```bash
python val.py \
  --data data/moke.yaml \
  --weights runs/train/moke/weights/best.pt \
  --img 640
```

迁移数据集训练示例：

```bash
python train.py \
  --img 640 \
  --batch-size 16 \
  --epochs 100 \
  --data data/bvn.yaml \
  --cfg models/yolov5s.yaml \
  --weights yolov5s.pt \
  --name bvn
```

```bash
python train.py \
  --img 640 \
  --batch-size 16 \
  --epochs 100 \
  --data data/fire_smoke.yaml \
  --hyp data/hyps/hyp.smoke.yaml \
  --cfg models/yolov5s.yaml \
  --weights yolov5s.pt \
  --name fire_smoke
```

```bash
python train.py \
  --img 640 \
  --batch-size 16 \
  --epochs 100 \
  --data data/moke_black.yaml \
  --cfg models/yolov5s.yaml \
  --weights yolov5s.pt \
  --name moke_black
```

建议不要为了某个数据集修改 `train.py` 默认参数，统一通过 `--data`、`--hyp`、`--weights` 显式传参。

## 通用推理

使用 YOLOv5 官方入口 `detect.py` 做图片、视频、摄像头或 RTSP 推理：

```bash
python detect.py \
  --weights runs/train/moke/weights/best.pt \
  --source data/images/bus.jpg \
  --conf-thres 0.5
```

常见 `--source`：

- 图片：`data/images/bus.jpg`
- 视频：`outputs/debug/root_media/test.mp4`
- 目录：`data/images/`
- 摄像头：`0`
- RTSP：`rtsp://user:password@host/stream`

结果默认保存到 `runs/detect/exp*`。

## 示例应用

`examples/` 目录保存从 `python-yolo` 迁移过来的示例应用，主要用于参考和本地验证，不作为当前项目的生产入口。

```text
examples/
├── yolov5_web/
│   ├── rest_server.py       # Flask REST 推理服务示例
│   ├── gradio_demo.py       # Gradio 可视化推理示例
│   └── example_request.py   # REST API 请求示例
└── yolov5_ui/
    ├── detect_pyside6.py    # PySide6 GUI 推理示例
    ├── main_window.ui
    └── ui_main_window.py
```

Flask 示例：

```bash
python examples/yolov5_web/rest_server.py --port 5110 --model yolov5s
```

请求示例：

```bash
python examples/yolov5_web/example_request.py
```

Gradio 示例：

```bash
python examples/yolov5_web/gradio_demo.py --weights yolov5s.pt
```

如需生成公网临时链接：

```bash
python examples/yolov5_web/gradio_demo.py --weights yolov5s.pt --share
```

PySide6 示例需要额外安装 GUI 依赖，适合本机调试，不建议放到服务器默认运行链路。

## 业务推理

### run.py

`run.py` 是面向线上任务的业务入口，包含：

- 视频流采集线程，只保留最新帧。
- YOLO 推理线程，支持检测频率控制、连续命中去抖、告警冷却。
- 告警处理线程，向 `server` 上报图片和结构化告警信息；网络异常不会阻塞推理线程。
- 任务心跳上报。

`run.py` 和 `run_local.py` 共用 `business_runtime.py`，采集、推理、ROI、去抖、告警冷却、模型加载、日志格式保持一致；区别只在 `run.py` 真实上报接口，`run_local.py` 离线保存告警图片。

`run.py` 和 `run_local.py` 使用同一套日志格式：

```text
时间 日志级别 [线程名] 日志内容
```

运行时健壮性策略：

- 视频采集队列只保留最新帧，避免推理慢时视频延迟持续堆积。
- 告警队列满时会丢弃最旧告警，再写入新告警，避免接口慢或网络异常拖住推理线程。
- RTSP 连续读取失败后会自动重连，并按 2s、4s、8s 逐步退避，最大 30s。
- 任务上报和告警上报都有 5 秒超时；同类网络异常默认 60 秒输出一次完整异常栈，其余降频记录。
- 启动阶段会校验关键参数，参数非法会直接退出，不进入推理循环。

示例：

```bash
python run.py \
  --msn model001 \
  --ins instance001 \
  --secret secret \
  --taskId task001 \
  --videoId video001 \
  --videoUrl rtsp://example.com/stream \
  --server http://127.0.0.1:8080 \
  --detect_target '[0]' \
  --detect_thresh 0.66 \
  --detect_fps 1 \
  --min_hit_count 2 \
  --alarm_cooldown 10 \
  --weights /usr/src/app/runs/train/all/weights/best.pt
```

区域检测示例：

```bash
python run.py \
  --msn model001 \
  --ins instance001 \
  --secret secret \
  --taskId task001 \
  --videoId video001 \
  --videoUrl rtsp://example.com/stream \
  --server http://127.0.0.1:8080 \
  --detect_target '[0]' \
  --detect_area '[[100,100],[500,100],[500,400],[100,400]]'
```

关键参数：

- `--detect_target`：JSON 数组，指定检测类别 ID，如 `'[0,5]'`。
- `--detect_area`：JSON 多边形点位数组，传入后会自动开启入侵检测；开启 `rqjc/lgjc` 时至少需要 3 个点。
- `--detect_fps`：每秒检测次数，必须大于 0。
- `--min_hit_count`：连续命中次数，必须大于等于 1，用于降低单帧误报。
- `--alarm_cooldown`：同类告警冷却时间，单位秒，必须大于等于 0。
- `--fire`：火情模型开关。开启后代码会使用 `--fire_weights` 指定的模型。
- `--weights`：常规模型权重路径，默认 `/usr/src/app/runs/train/all/weights/best.pt`。
- `--fire_weights`：火情模型权重路径，默认 `/usr/src/app/runs/models/fire/best.pt`。

默认部署模型路径为：

```text
/usr/src/app/runs/train/all/weights/best.pt
```

Docker 或服务器部署时需要确保该路径存在；本地调试通常通过 `--weights` 显式传入相对路径。

### run_local.py

`run_local.py` 用于本地调试，参数基本同 `run.py`，但任务上报/告警上报逻辑更适合离线验证。典型用法：

```bash
python run_local.py \
  --msn test \
  --ins test \
  --secret test \
  --taskId test \
  --videoId test \
  --videoUrl outputs/debug/root_media/test.mp4 \
  --server http://127.0.0.1:8080 \
  --detect_target '[0]' \
  --detect_fps 1 \
  --min_hit_count 2 \
  --alarm_cooldown 10 \
  --weights runs/train/all/weights/best.pt
```

本地告警截图统一写入：

```text
outputs/alarm_images/
```

## 工具脚本

从视频抽一帧：

```bash
python tools/extract_one_frame.py \
  --video test.mp4 \
  --output data/samples/sample_frame.jpg \
  --frame 0
```

鼠标点选 ROI 区域：

```bash
python tools/select_roi.py \
  --image data/samples/sample_frame.jpg \
  --output data/samples/roi.json
```

操作方式：按顺序点击多边形顶点，按 `q` 或 `Esc` 结束。输出 JSON 可直接整理为 `--detect_area` 参数或写入配置。

按固定帧间隔批量抽帧：

```bash
python tools/video_to_img.py \
  --video outputs/debug/root_media/test.mp4 \
  --output data/raw/train_data_source/demo \
  --interval 20
```

## 模型和产物建议

- 训练输出：保留在 `runs/train/<name>/`，不要直接提交全部图片、TensorBoard 日志和权重。
- 部署权重：建议只保留当前线上使用的 `best.pt`，并在 `runs/models/<model_name>/readme.md` 说明来源、类别、训练数据、指标。
- 实验记录：建议把关键训练指标从 `results.csv` 摘到 `docs/`，而不是提交完整 `runs/train/`。

## 常见问题

### 数据集路径找不到

检查 `data/*.yaml` 中的 `path`、`train`、`val` 是否能组合成真实路径。比如 `data/moke.yaml` 会解析到：

```text
data/train/moke/images/train
data/train/moke/images/val
```

迁移数据集路径示例：

```text
data/bvn.yaml          -> data/train/bvn/images/train
data/fire_smoke.yaml   -> data/train/fire_smoke/images/train
data/moke_black.yaml   -> data/train/moke_black/images/train
```

如果移动了数据目录，需要同步修改对应 yaml 中的 `path`。

### RTSP 卡住或延迟很高

`run.py` 使用单帧队列，只保留最新帧；RTSP 连续读取失败会自动重连并退避。如果仍然延迟，优先检查网络、RTSP 服务端、OpenCV FFmpeg 支持和 `--detect_fps`。

### 后端接口异常会不会影响检测

告警上报在线程中异步处理，且告警队列满时会丢弃最旧告警，不会反向阻塞推理线程。任务状态上报和告警上报都设置了 5 秒超时，并对持续网络异常做日志降频。

### ROI 参数错误

开启入侵检测或离岗检测时，`--detect_area` 必须是至少 3 个点的 JSON 数组，例如：

```bash
--detect_area '[[100,100],[500,100],[500,400],[100,400]]'
```

如果 `--detect_fps`、`--min_hit_count`、`--alarm_cooldown` 或 ROI 参数非法，程序会在启动阶段直接报错退出。

### 告警太频繁或误报

优先调整：

- `--detect_thresh`
- `--min_hit_count`
- `--alarm_cooldown`
- `--detect_area`

### Git 状态里出现大量图片或视频

这些通常是运行输出、原始素材或调试样例。默认不建议入库；如需保留，请放到明确的样例目录并在文档中说明用途。
