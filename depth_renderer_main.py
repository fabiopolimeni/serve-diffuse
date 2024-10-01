import argparse
import os
import time
from PIL import Image
from depth_renderer import DepthRenderer
from utils import ranged_type

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image to render")

    parser.add_argument(
        "--image", type=str, required=True, help="The base image that drives the render"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt to use for the render"
    )
    parser.add_argument(
        "--steps",
        type=ranged_type(int, 1, 40),
        default=4,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance",
        type=ranged_type(float, 0, 10),
        default=3.5,
        help="Scale for classifier-free guidance",
    )
    parser.add_argument(
        "--is_depth",
        type=bool,
        required=False,
        default=False,
        help="If true, the depth map will be extracred from the image",
    )
    parser.add_argument(
        "--depth_weight",
        type=ranged_type(float, 0, 1),
        default=0.5,
        help="Scale for depth conditioning",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs",
        help="Directory to save the output images. Defaults to ./outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the generator. If not provided, a random seed will be used",
    )
    parser.add_argument(
        "--save_depth",
        type=bool,
        default=False,
        help="If true, the depth map will be saved",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    renderer = DepthRenderer()
    renderer.load_pipelines()

    base_image = Image.open(args.image)

    timestamp = int(time.time())
    color_image, depth_image = renderer.render_image(
        base_image,
        args.prompt,
        args.steps,
        args.guidance,
        args.is_depth,
        args.depth_weight,
        args.seed,
    )

    print(f"Image generated in {time.time() - timestamp} seconds")

    # Save depth image
    if args.save_depth:
        depth_output = (
            f"{os.path.splitext(os.path.basename(args.image))[0]}_depth-{timestamp}.png"
        )
        depth_image.save(os.path.join(args.outdir, depth_output))

    # Save color image
    color_output = (
        f"{os.path.splitext(os.path.basename(args.image))[0]}_color-{timestamp}.png"
    )
    color_image.save(os.path.join(args.outdir, color_output))

    print(f"Color image saved to {color_output}")
