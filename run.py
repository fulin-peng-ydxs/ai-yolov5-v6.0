# coding:utf-8
"""
线上业务推理入口。

用途：
- 作为 Docker/服务器中的正式启动脚本，由 run.sh 默认调用。
- 读取命令行参数后进入 online 模式。
- online 模式会真实调用任务状态上报接口和告警上报接口。

注意：
- 采集、推理、ROI、去抖、告警冷却、模型加载等机制都在 business_runtime.py 中实现。
- 本文件只负责选择 online 模式，避免 run.py 和 run_local.py 复制业务逻辑后再次分叉。
"""

from business_runtime import main


if __name__ == "__main__":
    main("online")
