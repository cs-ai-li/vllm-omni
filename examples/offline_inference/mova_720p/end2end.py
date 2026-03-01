import argparse
import os
from vllm_omni.entrypoints.omni import OmniDiffusion

def main():
    parser = argparse.ArgumentParser(description="MOVA 720p 离线推理示例")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--prompts", type=str, default="A beautiful sunset over the mountains.", help="提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG 系数")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出保存目录")
    
    args = parser.parse_args()

    # 1. 初始化推理引擎
    # OmniDiffusion 会自动从注册表中根据模型类型找到 Mova720PPipeline
    print(f"正在加载模型: {args.model}...")
    engine = OmniDiffusion(model=args.model)

    # 2. 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 执行推理
    print(f"正在生成图像，提示词: '{args.prompts}'")
    outputs = engine.generate(
        prompts=[args.prompts],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale
    )

    # 4. 保存结果
    for i, output in enumerate(outputs):
        output_path = os.path.join(args.output_dir, f"output_{i}.png")
        output.save(output_path)
        print(f"图像已保存至: {output_path}")

if __name__ == "__main__":
    main()
