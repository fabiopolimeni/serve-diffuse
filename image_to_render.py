import argparse
import os
import time
import torch
from transformers import pipeline
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv(".env")

    # Get the HF_TOKEN from environment variables
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment variables. Add it to your environment variables or to an .env file."
        )

    # Set the HF_TOKEN environment variable
    os.environ["HF_TOKEN"] = hf_token

    parser = argparse.ArgumentParser(description="Depth Anything V2")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./outputs")

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Running pipelines on device: {device}")

    os.makedirs(args.outdir, exist_ok=True)

    timestamp = int(time.time())

    # Create the depth map out of the input image
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Base-hf",
        device=device,
    )
    image = Image.open(args.image)
    depth_image = depth_pipe(image)["depth"]

    depth_output = (
        os.path.splitext(os.path.basename(args.image))[0] + f"_depth-{timestamp}.png"
    )

    depth_image.save(os.path.join(args.outdir, depth_output))

    # Use the depth map to drive the image generation
    prompt = args.prompt

    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"

    controlnet_union = FluxControlNetModel.from_pretrained(
        controlnet_model, torch_dtype=torch.bfloat16
    )
    controlnet = FluxMultiControlNetModel([controlnet_union])

    color_pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    color_pipe.to(device=device)

    width = depth_image.size[0]
    height = depth_image.size[1]

    # The controlnet pipeline expects a control image in RGB format
    if depth_image.mode != "RGB":
        # Convert grayscale to RGB
        depth_image = depth_image.convert("RGB")

    color_image = color_pipe(
        prompt,
        control_image=[depth_image],
        control_mode=[2],
        controlnet_conditioning_scale=[0.5],
        width=width,
        height=height,
        num_inference_steps=6,
        guidance_scale=3.5,
        generator=torch.manual_seed(42),
    ).images[0]

    color_output = (
        os.path.splitext(os.path.basename(args.image))[0] + f"_color-{timestamp}.png"
    )

    color_image.save(os.path.join(args.outdir, color_output))
