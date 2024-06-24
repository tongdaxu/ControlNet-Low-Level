import os
import torch
import argparse 

from pipe import EulerAncestralDiscreteInverse, StableDiffusionControlNetInverse
from dataset import ImageDataset
from op import SuperResolutionOperator
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer
from train_controlnet import import_model_class_from_model_name_or_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ControlNet for Low Level vision')
    parser.add_argument('--data', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--cnmodel', type=str)
    parser.add_argument('--scale', type=float, default=2.4)

    args = parser.parse_args()

    seed=64
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)

    DTYPE = torch.float32
    DATA_ROOT = args.data
    OUT_ROOT = args.out
    SCALE = args.scale

    out_dirs = ["source", "low_res", "recon", "recon_low_res"]
    out_dirs = [os.path.join(OUT_ROOT, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=DATA_ROOT, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    f = SuperResolutionOperator([1, 3, 512, 512], 8)
    f = f.to(dtype=DTYPE, device="cuda")

    model_id = "stabilityai/stable-diffusion-2-base"

    scheduler = EulerAncestralDiscreteInverse.from_pretrained(model_id, subfolder="scheduler")
    text_encoder_cls = import_model_class_from_model_name_or_path(model_id, None)
    text_encoder = text_encoder_cls.from_pretrained(
        model_id, subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet",
    )
    # controlnet = ControlNetModel.from_pretrained("/NEW_EDS/JJ_Group/xutd/diffusion-inversion/sd2cn_srx8_25/checkpoint-4000/controlnet")
    controlnet = ControlNetModel.from_pretrained(args.cnmodel)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        use_fast=False,
    )
    
    pipe = StableDiffusionControlNetInverse.from_pretrained(
        model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=DTYPE,
    )
    pipe.control_image_processor.config.do_normalize = True
    pipe.scheduler = scheduler
    pipe.to(device="cuda")

    for i, (x, x_path) in enumerate(dataloader):
        x_name = x_path[0].split('/')[-1]

        x = x.to(dtype=DTYPE, device="cuda")
        y1 = f(x)
        y2 = f(x, keep_shape=True)
        
        _, x_hat, _ = pipe(f=f,
                           y=y1,
                           scale=SCALE,
                           prompt="",
                           image=y2,
                           height=512,
                           width=512,
                           num_inference_steps=250,
                           guidance_scale=0.0)
        y_hat = f(x_hat)
        out_tensors = [x, y1, x_hat, y_hat]
        for i in range(4):
            save_image(out_tensors[i], os.path.join(out_dirs[i], x_name), normalize=True, value_range=(-1, 1))
