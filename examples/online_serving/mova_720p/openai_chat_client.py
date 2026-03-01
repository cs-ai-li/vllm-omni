import argparse
import requests
import json
import os

def main():
    parser = argparse.ArgumentParser(description="MOVA 720p OpenAI 兼容客户端示例")
    parser.add_argument("--prompt", type=str, default="A futuristic cityscape at night.", help="提示词")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/chat/completions", help="服务器地址")
    parser.add_argument("--model", type=str, default="mova_720p", help="模型名称")
    parser.add_argument("--output", type=str, default="server_output.png", help="输出文件名")
    
    args = parser.parse_args()

    # 1. 构造请求负载
    # 符合 OpenAI Chat API 格式
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": args.prompt
            }
        ],
        # 可以通过 extra_body 传递扩散模型特有的参数
        "extra_body": {
            "num_inference_steps": 50,
            "guidance_scale": 5.0
        }
    }

    # 2. 发送请求
    print(f"正在发送请求到 {args.url}...")
    try:
        response = requests.post(args.url, json=payload, stream=True)
        response.raise_for_status()

        # 3. 解析结果并保存图像
        # 备注：具体返回格式取决于 vLLM-Omni 的 API 实现（通常是二进制图片或包含 Base64 的 JSON）
        if response.headers.get("Content-Type", "").startswith("image/"):
            with open(args.output, "wb") as f:
                f.write(response.content)
            print(f"图像生成成功并已保存至: {args.output}")
        else:
            result = response.json()
            print("收到响应:", result)
            
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    main()
