# Prediction interface for Cog ⚙️
# https://cog.run/python

import time
import os
import torch
import platform
import base64
import io
from PIL import Image
from cog import BaseModel, BasePredictor, Input, Path
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image

CONTROLNET_ID = "InstantX/SD3-Controlnet-Depth"
CONTROLNET_CACHE = "controlnet-cache"
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
MODEL_CACHE = "model-cache"

class Output(BaseModel):
    file: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # Determine the device based on the operating system
        if platform.system() == "Darwin" and torch.backends.mps.is_available():  # macOS
            self.device = "mps"
        elif platform.system() in ["Linux", "Windows"]:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        # Create the "outputs" folder if it doesn't exist
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        self.controlnet = SD3ControlNetModel.from_pretrained(
            CONTROLNET_ID,
            cache_dir=CONTROLNET_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        ).to(self.device, torch.float16)

        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            MODEL_ID,
            controlnet=self.controlnet,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True,
        ).to(self.device, torch.float16)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="blurry, NSFW, nude, naked, porn, ugly, bad quality, bad quality, worst quality",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=40,
            default=10
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            ge=1,
            le=20,
            default=7
        ),
        depth_image_base64: str = Input(
            description="Depth image, in base64 string, to drive the depth conditioning",
            default=None,
        ),
        depth_image_path: Path = Input(
            description="Depth image to drivee the depth conditioning",
            default=None,
        ),
        depth_conditioning_scale: float = Input(
            description="Scale for depth conditioning",
            ge=0,
            le=1,
            default=0.8
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if prompt is None or prompt == "":
            raise Exception(
                "The prompt is required but was not set. Please set the prompt and try again."
            )
        
        # Check if at least one depth image input is provided
        if depth_image_path is None and depth_image_base64 is None:
            raise ValueError("Either depth_image_path or depth_image_base64 must be provided.")
        
        # Process the depth image
        if depth_image_path:
            print(f"Using depth image from path: {depth_image_path}")
            depth_image = load_image(depth_image_path)
        elif depth_image_base64:
            depth_image = Image.open(io.BytesIO(base64.b64decode(depth_image_base64)))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")

        # First-time "warmup" pass
        if (self.device == "mps"):
            _ = self.pipe(prompt, num_inference_steps=1)

        generator = torch.Generator(self.device).manual_seed(seed)
        image = self.pipe(prompt=prompt, 
                          num_inference_steps=num_inference_steps, 
                          negative_prompt=negative_prompt,
                          guidance_scale=guidance_scale,
                          control_image=depth_image,
                          controlnet_conditioning_scale=depth_conditioning_scale,
                          width=width, 
                          height=height, 
                          generator=generator).images[0]
        
        timestamp = int(time.time())
        output_path = f"outputs/out_{timestamp}.png"

        print(f"Saving image to {output_path}")

        image.save(output_path)
        return Path(output_path)
