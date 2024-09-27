import argparse
import os
import time
import torch
from transformers import pipeline
from PIL import Image

from diffusers import FluxControlNetPipeline, FluxControlNetModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='./outputs')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    os.makedirs(args.outdir, exist_ok=True)

    # Create the depth map out of input image 

    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=device)
    image = Image.open(args.image)
    depth = depth_pipe(image)["depth"]

    # Use the depth map to drive the image generation
    prompt = args.prompt
    # prompt = prompt.replace('{', '{{').replace('}', '}}')

    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    color_pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.bfloat16, device=device
    )

    width = depth.size[0]
    height = depth.size[1]

    color = color_pipe(prompt,
                control_image=depth,
                controlnet_conditioning_scale=0.5,
                width=width,
                height=height,
                num_inference_steps=24,
                guidance_scale=3.5,
    ).images[0]

    timestamp = int(time.time())

    depth_output = os.path.splitext(os.path.basename(args.image))[0] + f'_depth-{timestamp}.png'
    depth.save(os.path.join(args.outdir, depth_output))

    color_output = os.path.splitext(os.path.basename(args.image))[0] + f'_color-{timestamp}.png'
    color.save(os.path.join(args.outdir, color_output))