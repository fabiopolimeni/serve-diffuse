import argparse
from email import generator
import os
import time
import torch
from transformers import pipeline
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from dotenv import load_dotenv
from utils import ranged_type

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Depth Anything V2")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=ranged_type(int, 1, 40), default=4)
    parser.add_argument("--guidance", type=ranged_type(float, 0, 10), default=3.5)
    parser.add_argument("--is_depth", type=bool, required=False, default=False)
    parser.add_argument("--depth_weight", type=ranged_type(float, 0, 1), default=0.5)
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv(".env")

    # Get the HF_TOKEN from environment variables.
    # This is necessary to load the FLUX.1-dev model from the Hugging Face Hub.
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment variables. Add it to your environment variables or to an .env file."
        )

    # Set the HF_TOKEN environment variable
    os.environ["HF_TOKEN"] = hf_token

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Pipelines will run on {device}")

    os.makedirs(args.outdir, exist_ok=True)

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

    timestamp = int(time.time())

    # Create or load the depth map
    if args.is_depth:
        # Load the provided depth image
        depth_image = Image.open(args.image)
    else:
        # Create the depth map from input image
        depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=device,
        )
        image = Image.open(args.image)
        depth_image = depth_pipe(image)["depth"]

        depth_output = (
            os.path.splitext(os.path.basename(args.image))[0]
            + f"_depth-{timestamp}.png"
        )

        depth_image.save(os.path.join(args.outdir, depth_output))

    # Use the depth map to drive the image generation
    prompt = args.prompt

    width = depth_image.size[0]
    height = depth_image.size[1]

    # The controlnet pipeline expects a control image in RGB format
    if depth_image.mode != "RGB":
        # Convert grayscale to RGB
        depth_image = depth_image.convert("RGB")

    # If seed is not provided, use a random seed
    seed = args.seed if not None else int.from_bytes(os.urandom(2), "big")

    generator = torch.manual_seed(seed)
    color_image = color_pipe(
        prompt,
        control_image=[depth_image],
        control_mode=[2],
        controlnet_conditioning_scale=[args.depth_weight],
        width=width,
        height=height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]

    color_output = (
        os.path.splitext(os.path.basename(args.image))[0] + f"_color-{timestamp}.png"
    )

    color_image.save(os.path.join(args.outdir, color_output))
