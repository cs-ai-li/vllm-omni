# MOVA 720p 在线服务示例 (OpenAI 兼容)

本文件夹展示了如何将 MOVA 720p 作为一个标准 API 服务器运行。

### 1. 启动服务器

你可以一键启动符合 OpenAI 接口标准的 API 服务：

```bash
cd examples/online_serving/mova_720p
# 格式: bash run_server.sh <模型权重路径>
bash run_server.sh /path/to/your/mova-weights
```

默认服务会在 `http://localhost:8000` 运行。

### 2. 发送测试请求

启动服务器后，打开另一个终端运行客户端脚本进行测试：

```bash
cd examples/online_serving/mova_720p
python openai_chat_client.py --prompt "赛博朋克风格的繁华都市"
```

### 接口说明
- **Endpoint**: `POST /v1/chat/completions`
- **支持的功能**: 接收文本描述并生成图像。
- **自定义参数**: 可通过 `extra_body` 控制扩散步数 (`num_inference_steps`)。
