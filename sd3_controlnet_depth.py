import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image

# load hf_token from environment file
import os
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv(".env.local")

# Get the HF_TOKEN from environment variables
hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")

# Set the HF_TOKEN environment variable
os.environ["HF_TOKEN"] = hf_token

# load pipeline
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet
)
pipe.to("mps")

# config
control_image = load_image("depth.jpeg")
prompt = "a panda cub, captured in a close-up, in forest, is perched on a tree trunk. good composition, Photography, the cub's ears, a fluffy black, are tucked behind its head, adding a touch of whimsy to its appearance. a lush tapestry of green leaves in the background. depth of field, National Geographic"
n_prompt = "bad hands, blurry, NSFW, nude, naked, porn, ugly, bad quality, worst quality"

# First-time "warmup" pass
_ = pipe(prompt, num_inference_steps=1)

# to reproduce result in our example
generator = torch.Generator(device="cpu").manual_seed(4000)
image = pipe(
    prompt, 
    negative_prompt=n_prompt, 
    control_image=control_image, 
    controlnet_conditioning_scale=0.5,
    guidance_scale=7.0,
    generator=generator,
    num_inference_steps=4
).images[0]
image.save('image.jpg')
