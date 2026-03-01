# MOVA 720p 离线推理示例

本文件夹提供了如何使用 vLLM-Omni 运行 MOVA 720p 模型的示例脚本。

### 准备工作
请确保你已经安装了 `vllm-omni` 并准备好了 MOVA 的模型权重。

### 快速开始

1. **进入目录**:
   ```bash
   cd examples/offline_inference/mova_720p
   ```

2. **运行推理**:
   ```bash
   python end2end.py --model /path/to/your/mova-weights --prompts "一只在草地上奔跑的小狗" --steps 50
   ```

### 参数说明
- `--model`: 必须，指定包含 `vae`, `transformer`, `scheduler` 等文件夹的模型根目录。
- `--prompts`: 可选，生成图像的描述。
- `--steps`: 可选，扩散步数（默认 50）。
- `--guidance-scale`: 可选，分类器自由引导系数（默认 5.0）。
