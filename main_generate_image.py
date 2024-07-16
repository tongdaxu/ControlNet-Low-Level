import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save("generation_of_zyr_image_1.png")
