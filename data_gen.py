from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
import torch
import os
import numpy as np
from tqdm import tqdm

seed=64
torch.manual_seed(seed=seed)
torch.cuda.manual_seed_all(seed=seed)
np.random.seed(seed=seed)

model_id = "stabilityai/stable-diffusion-2-base"
OUT_ROOT = "./gen_512"
COUNT = 64000
os.makedirs(OUT_ROOT, exist_ok=True)

# Use the Euler scheduler here instead
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = ""

for i in tqdm(range(COUNT)):
    image = pipe(prompt).images[0]  
    image.save(os.path.join(OUT_ROOT, '{0:06d}.png'.format(i)))
