# Prediction interface for Cog ⚙️
# https://cog.run/python

import io
import time
from turtle import width
from typing import Any
from cog import BaseModel, BasePredictor, Input, Path
import os
import torch
import platform
from diffusers import StableDiffusionPipeline

class Output(BaseModel):
    file: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

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

        # Move the pipeline to the determined device
        self.pipe.to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=40, default=10
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Output:
        """Run a single prediction on the model"""

        if prompt is None or prompt == "":
            raise Exception(
                f"The prompt is required but was not set. Please set the prompt and try again."
            )

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")

        # First-time "warmup" pass
        if (self.device == "mps"):
            _ = self.pipe(prompt, num_inference_steps=1)

        # Results match those from the CPU device after the warmup pass.
        generator = torch.Generator(self.device).manual_seed(seed)
        image = self.pipe(prompt=prompt, 
                          num_inference_steps=num_inference_steps, 
                          negative_prompt=negative_prompt, 
                          height=512, 
                          width=512, 
                          generator=generator).images[0]
        
        timestamp = int(time.time())
        output_path = f"outputs/out_{timestamp}.png"
        image.save(output_path)
        return Output(file=output_path)
