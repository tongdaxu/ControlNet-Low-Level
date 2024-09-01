import os
import torch
import argparse 

from pipe import StableDiffusionControlNetInverse, ControlNetLoraModel, EulerAncestralDSG
from dataset import ImageDataset
from op import SuperResolutionOperator, GaussialBlurOperator
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
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDPMScheduler, DDIMScheduler

# torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ControlNet for Low Level vision')
    parser.add_argument('--data', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--cnmodel', type=str)
    parser.add_argument('--scale', type=float, default=4.8)
    parser.add_argument('--disablecn', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--step', type=int, default=250)
    parser.add_argument('--operator', type=str, default="srx8")
    parser.add_argument('--mode', type=str, default="dps")
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    args = parser.parse_args()

    seed=64
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)

    DTYPE = torch.float32
    DATA_ROOT = args.data
    OUT_ROOT = args.out
    SCALE = args.scale
    USE_CN = (not args.disablecn)
    STEP = args.step
    USE_LORA = args.lora

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
    if args.operator == "srx8":
        f = SuperResolutionOperator([1, 3, 512, 512], 8)
    elif args.operator == "gdb":
        f = GaussialBlurOperator()
    else:
        raise ValueError

    f = f.to(dtype=DTYPE, device="cuda")

    model_id = "stabilityai/stable-diffusion-2-base"
    if args.mode == "dps" or args.mode == "cn":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif args.mode == "dsg" or args.mode == "max":
        scheduler = EulerAncestralDSG.from_pretrained(model_id, subfolder="scheduler")
    else:
        raise NotImplementedError
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

    if USE_LORA:
        controlnet = ControlNetLoraModel.from_pretrained(args.cnmodel)
    else:
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
    mses = []
    for i, (x, x_path) in enumerate(dataloader):

        if i % args.ngpu != args.rank:
            continue

        x_name = x_path[0].split('/')[-1]

        x = x.to(dtype=DTYPE, device="cuda")
        y1 = f(x)
        y2 = f(x, keep_shape=True)
        if args.mode == "dps":
            _, x_hat, _ = pipe(f=f,
                            y=y1,
                            scale=SCALE,
                            use_cn=USE_CN,
                            prompt="",
                            image=y2,
                            height=512,
                            width=512,
                            num_inference_steps=STEP,
                            guidance_scale=0.0)
        elif args.mode == "dsg":
            _, x_hat, _ = pipe.run_dsg(f=f,
                            y=y1,
                            scale=SCALE,
                            use_cn=USE_CN,
                            prompt="",
                            image=y2,
                            height=512,
                            width=512,
                            num_inference_steps=STEP,
                            guidance_scale=0.0)
        elif args.mode == "max":
            _, x_hat, _ = pipe.run_dsg_max(f=f,
                            y=y1,
                            scale=SCALE,
                            use_cn=USE_CN,
                            prompt="",
                            image=y2,
                            height=512,
                            width=512,
                            num_inference_steps=STEP,
                            guidance_scale=0.0)
        elif args.mode == "cn":
            _, x_hat, _ = pipe.run_cn(f=f,
                            y=y1,
                            scale=SCALE,
                            use_cn=USE_CN,
                            prompt="",
                            image=y2,
                            height=512,
                            width=512,
                            num_inference_steps=STEP,
                            guidance_scale=0.0)
        else:
            raise NotImplementedError
        y_hat = f(x_hat)

        out_tensors = [x, y1, x_hat, y_hat]

        mse = torch.mean((x - x_hat) ** 2).item()
        mses.append(mse)

        print("[{0}/1000], mse: {1:.6f}/avg mse: {2:.6f}".format(i, mse, np.mean(mses)))

        for i in range(4):
            save_image(out_tensors[i], os.path.join(out_dirs[i], x_name), normalize=True, value_range=(-1, 1))
