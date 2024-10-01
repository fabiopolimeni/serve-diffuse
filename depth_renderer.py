import os
import torch
from transformers import pipeline
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel

# from dotenv import load_dotenv

MODEL_ID = "black-forest-labs/FLUX.1-dev"
CONTROLNET_ID = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
CACHE_DIR = "checkpoints"


class DepthRenderer:
    def __init__(self):
        self.base_model_id = MODEL_ID
        self.controlnet_model_id = CONTROLNET_ID
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

        # base_model = FluxPipeline.from_pretrained(
        #     self.base_model_id,
        #     cache_dir=CACHE_DIR,
        #     torch_dtype=torch.bfloat16,
        #     use_safetensors=True,
        #     local_files_only=True,
        # )
        # base_model.to(device=self.device)

        controlnet_model = FluxControlNetModel.from_pretrained(
            self.controlnet_model_id,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True,
        )
        controlnet_model.to(device=self.device)

        controlnet = FluxMultiControlNetModel([controlnet_model])

        self.color_pipe = FluxControlNetPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.color_pipe.to(device=self.device)

        self.depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=self.device,
        )

    @torch.inference_mode()
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
