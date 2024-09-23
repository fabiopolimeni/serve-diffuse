# make sure you're logged in with `huggingface-cli login`
import os
import torch
import platform
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Determine the device based on the operating system
if platform.system() == "Darwin" and torch.backends.mps.is_available():  # macOS
    device = "mps"
elif platform.system() in ["Linux", "Windows"]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

print(f"Using device: {device} on platform: {platform.system()}")

# Create the "outputs" folder if it doesn't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Move the pipeline to the determined device
pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"

# First-time "warmup" pass
if (device == "mps"):
    _ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt, num_inference_steps=4).images[0]
image.save("outputs/astronaut_rides_horse.png")