#!/bin/bash

# 获取所有指令参数
args=("$@")

# 执行启动指令
python run.py ${args[@]}