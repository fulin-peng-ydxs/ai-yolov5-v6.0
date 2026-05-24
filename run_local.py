# coding:utf-8
"""
本地业务调试入口。

用途：
- 用于开发机或离线环境验证视频采集、模型推理、ROI、去抖和告警冷却机制。
- 读取命令行参数后进入 local 模式。
- local 模式不调用真实任务状态上报接口，也不调用真实告警接口；命中告警时只保存图片。

输出：
- 本地告警截图保存到 outputs/alarm_images/。

注意：
- 除接口上报改为离线适配外，其余运行机制与 run.py 共用 business_runtime.py。
"""

from business_runtime import main


if __name__ == "__main__":
    main("local")
