import os
import torch
from pipe import StableDiffusionInverse, EulerAncestralDiscreteInverse, StableSRPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDPMScheduler, DDIMScheduler
from dataset import ImageDataset
from op import SuperResolutionOperator, RealWorldOperator
from torchvision import transforms
import numpy as np
import argparse 
from torchvision.utils import save_image
from stablesr import UNet2DSDSR, UNet2DHalfSDSR
from diffusers import AutoencoderKL
from transformers import AutoTokenizer
from train_controlnet import import_model_class_from_model_name_or_path
import random

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ControlNet for Low Level vision')
    parser.add_argument('--model', type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument('--data', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--scale', type=float, default=4.8)
    parser.add_argument('--mode', type=str, default="dps")
    parser.add_argument('--step', type=int, default=500)
    args = parser.parse_args()

    def set_random(seed):
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        np.random.seed(seed=seed)
        random.seed(seed)

    DTYPE = torch.float32
    DATA_ROOT = args.data
    OUT_ROOT = args.out
    SCALE = args.scale
    STEP = args.step

    out_dirs = ["source", "low_res", "recon", "recon_low_res"]
    out_dirs = [os.path.join(OUT_ROOT, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.Resize(512),
         transforms.CenterCrop(512),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=DATA_ROOT, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    f = RealWorldOperator()
    f = f.to(dtype=DTYPE, device="cuda")

    ckpt = torch.load('/NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sdsr_reimpl/sdsr.ckpt')

    model = UNet2DHalfSDSR(
        in_channels=4,
        out_channels=256,
        attention_head_dim=[4,4,4,4],
        block_out_channels=[256,256,512,512],
        cross_attention_dim=1024,
        down_block_types=[
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ],
        downsample_padding=1,
        dual_cross_attention=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        layers_per_block=2,
        mid_block_scale_factor=1,
        norm_eps=1e-5,
        norm_num_groups=32,
        sample_size=96,
    )
    model.load_state_dict(ckpt["struct"])
    model.eval()

    model_unet = UNet2DSDSR(
        in_channels=4,
        out_channels=4,
        attention_head_dim=[5,10,20,20],
        block_out_channels=[320,640,1280,1280],
        cross_attention_dim=1024,
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ],
        downsample_padding=1,
        dual_cross_attention=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        layers_per_block=2,
        mid_block_scale_factor=1,
        norm_eps=1e-5,
        norm_num_groups=32,
        sample_size=96,
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ],
        use_linear_projection=True,
    )
    model_unet.load_state_dict(ckpt["unet"])
    model_unet.eval()

    model_id = args.model

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    train_inputs = tokenizer(
        [""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0].cuda()

    text_encoder_cls = import_model_class_from_model_name_or_path(model_id, revision=None)

    text_encoder = text_encoder_cls.from_pretrained(
        model_id, subfolder="text_encoder",
    ).cuda()
    train_inputs = train_inputs.unsqueeze(0)
    encoder_hidden_states = text_encoder(train_inputs, return_dict=False)[0]

    pipe = StableSRPipeline(
        vae = vae,
        text_encoder=None,
        tokenizer=None,
        unet=model_unet,
        controlnet=model,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
    pipe.to(device="cuda")

    mses = []
    for i, (x, x_path) in enumerate(dataloader):
        x_name = x_path[0].split('/')[-1]
        x_name = x_name[:-4] + ".png"
        x = x.to(dtype=DTYPE, device="cuda")

        # ensure same operator 
        set_random(i)
        y = f(x)
        _, x_hat, _ = pipe.run_dps_k(f=f,
                           y=y,
                           scale=SCALE,
                           prompt_embeds=encoder_hidden_states,
                           image=y,
                           height=512,
                           width=512,
                           num_inference_steps=STEP,
                           guidance_scale=0.0)
        # ensure same operator
        set_random(i)
        y_hat = f(x_hat)

        out_tensors = [x, y, x_hat, y_hat]

        mse = torch.mean((x - x_hat) ** 2)
        mses.append(mse.item())

        print("[{0}/1000], mse: {1:.6f}/avg mse: {2:.6f}".format(i, mse, np.mean(mses)))

        for i in range(4):
            save_image(out_tensors[i], os.path.join(out_dirs[i], x_name), normalize=True, value_range=(-1, 1))
