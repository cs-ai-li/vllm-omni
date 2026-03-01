#!/bin/bash

# 获取模型路径，默认是一个占位符
MODEL=${1:-"/path/to/your/mova-weights"}
PORT=${2:-8000}

echo "正在启动 MOVA 720p API 服务器..."
echo "模型路径: $MODEL"
echo "端口: $PORT"

# 启动 vLLM Serve 模式
# --omni: 启用多模态/全方位模型支持
# --host 0.0.0.0: 允许外部访问
python -m vllm.entrypoints.openai.api_server 
    --model $MODEL 
    --omni 
    --port $PORT 
    --host 0.0.0.0 
    --max-model-len 4096
