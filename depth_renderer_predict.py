# Prediction interface for Cog ⚙️
# https://cog.run/python

import time
from PIL import Image
from cog import BaseModel, BasePredictor, Input, Path
import os
import torch
import base64
import io
from depth_renderer import DepthRenderer


class Output(BaseModel):
    color_image: Path
    depth_image: Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.outdir = "results"
        os.makedirs(self.outdir, exist_ok=True)

        base_model = "black-forest-labs/FLUX.1-dev"
        controlnet_model = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
        self.renderer = DepthRenderer(base_model, controlnet_model)
        self.renderer.load_pipelines()

    @torch.inference_mode()
    def predict(
        self,
        base_image_url: Path = Input(
            description="Base, or depth if is_depth==True, image used to drive the final render",
            default=None,
        ),
        base_image_base64: str = Input(
            description="Base, or depth if is_depth==True, image in base64, used to drive the final render",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="",
        ),
        inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=40, default=10
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7
        ),
        depth_scale: float = Input(
            description="Scale for depth conditioning", ge=0, le=1, default=0.5
        ),
        is_depth: bool = Input(
            description="If true, the depth map will be extracted from the image",
            default=False,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Output:
        """Run a single prediction"""

        if prompt is None or prompt == "":
            raise Exception(
                "The prompt is required but was not set. Please set the prompt and try again."
            )

        # Check if at least one depth image input is provided
        if base_image_url is None and base_image_base64 is None:
            raise ValueError(
                "Either an image URL or file, or a base64 string, must be provided."
            )

        base_image = None
        if base_image_url:
            base_image = Image.open(base_image_url)
        elif base_image_base64:
            mime_type, base64_data = base_image_base64.split(";base64,")
            base_image = Image.open(io.BytesIO(base64.b64decode(base64_data)))

        timestamp = int(time.time())
        color_image, depth_image = self.renderer.render_image(
            base_image=base_image,
            prompt=prompt,
            steps=inference_steps,
            guidance=guidance_scale,
            is_depth=is_depth,
            depth_weight=depth_scale,
            seed=seed,
        )

        print(f"Image generated in {time.time() - timestamp} seconds")

        # Save depth image
        depth_output = f"depth-{timestamp}.png"
        depth_image.save(os.path.join(self.outdir, depth_output))

        # Save color image
        color_output = f"_color-{timestamp}.png"
        color_image.save(os.path.join(self.outdir, color_output))

        return Output(color_image=color_image, depth_image=depth_image)
