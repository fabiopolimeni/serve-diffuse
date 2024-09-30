import os
import torch
from transformers import pipeline
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from dotenv import load_dotenv


class DepthRenderer:
    def __init__(self, base_model, controlnet_model):
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.device = self._get_device()
        self.color_pipe = None
        self.depth_pipe = None

    def _get_device(self):
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def load_pipelines(self):
        load_dotenv(".env")

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found in environment variables. Add it to your environment variables or to an .env file."
            )

        os.environ["HF_TOKEN"] = hf_token

        print(f"Pipelines will run on {self.device}")

        controlnet_union = FluxControlNetModel.from_pretrained(
            self.controlnet_model, torch_dtype=torch.bfloat16
        )
        controlnet = FluxMultiControlNetModel([controlnet_union])
        self.color_pipe = FluxControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
        )
        self.color_pipe.to(device=self.device)

        self.depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=self.device,
        )

    def render_image(
        self,
        base_image: Image,
        prompt: str,
        steps=4,
        guidance=3.5,
        is_depth=False,
        depth_weight=0.5,
        seed=None,
    ):

        depth_image = base_image

        if not is_depth:
            print("Converting image to depth")
            depth_image = self.depth_pipe(depth_image)["depth"]

        width, height = depth_image.size
        depth_image = depth_image.convert("RGB")

        seed = seed if seed is not None else int.from_bytes(os.urandom(2), "big")
        generator = torch.manual_seed(seed)

        print(f"Rendering image with seed {seed}")
        color_image = self.color_pipe(
            prompt=prompt,
            control_image=[depth_image],
            control_mode=[2],
            controlnet_conditioning_scale=[depth_weight],
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

        return color_image, depth_image
