import os
import torch
from pipe import StableDiffusionInverse, EulerAncestralDiscreteInverse, EulerAncestralDSG, LCMScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDPMScheduler, DDIMScheduler
from dataset import ImageDataset
from op import SuperResolutionOperator, LayoutOperator
from torchvision import transforms
import numpy as np
import argparse 
from torchvision.utils import save_image
from diffusers import (
    UNet2DConditionModel,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ControlNet for Low Level vision')
    parser.add_argument('--model', type=str, default="Lykon/dreamshaper-7")
    parser.add_argument('--data', type=str, default="/NEW_EDS/JJ_Group/xutd/common_datasets/lsun_bedroom_256x256/lanczos")
    parser.add_argument('--out', type=str)
    parser.add_argument('--scale', type=float, default=4.8)
    parser.add_argument('--mode', type=str, default="dps")
    parser.add_argument('--stsl_k', type=int, default=2)
    parser.add_argument('--stsl_eta', type=int, default=0.1)
    parser.add_argument('--pt_lr', type=int, default=1e-5)
    parser.add_argument('--step', type=int, default=500)
    args = parser.parse_args()

    seed=64
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)

    DTYPE = torch.float32
    DATA_ROOT = args.data
    OUT_ROOT = args.out
    SCALE = args.scale
    STEP = args.step
    PROMPT = "A high quality photo of a bedroom."

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

    f = LayoutOperator()
    f = f.to(dtype=DTYPE, device="cuda")

    model_id = args.model

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionInverse.from_pretrained(model_id, scheduler=scheduler, torch_dtype=DTYPE)
    pipe = pipe.to("cuda")

    lcm_unet = UNet2DConditionModel.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", subfolder="unet",
    )
    lcm_unet = lcm_unet.to("cuda")

    lcm_scheduler = LCMScheduler.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler",
    )

    mses = []
    for i, (x, x_path) in enumerate(dataloader):
        if i >= 100:
            break
        x_name = x_path[0].split('/')[-1]
        x_name = x_name[:-4] + ".png"
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x, init=True)
        if args.mode == "dps":
            _, x_hat, _ = pipe.run_dps(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt=PROMPT,
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        if args.mode == "lgd":
            _, x_hat, _ = pipe.run_lgd(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt=PROMPT,
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "dpscm":
            _, x_hat, _ = pipe.run_dpscm(
                                    cm=lcm_unet,
                                    cms=lcm_scheduler,
                                    f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt=PROMPT,
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "lcm":
            image = pipe.run_lcm(
                                cm=lcm_unet,
                                cms=lcm_scheduler,
                                f=f,
                                y=y,
                                scale=SCALE,
                                prompt=PROMPT,
                                height=512,
                                width=512,
                                num_inference_steps=50,
                                guidance_scale=0.0,
                                )[0][0]
            image.save("example_lcm.png")
            assert(0)
        else:
            raise ValueError

        y = (y + 1.0) / 5.0
        y_hat = (f(x_hat, init=True) + 1.0) / 5.0
        out_tensors = [x, y, x_hat, y_hat]

        mse = torch.mean((x - x_hat) ** 2)
        mses.append(mse.item())

        print("[{0}/1000], mse: {1:.6f}/avg mse: {2:.6f}".format(i, mse, np.mean(mses)))

        for i in range(4):
            save_image(out_tensors[i], os.path.join(out_dirs[i], x_name), normalize=True, value_range=(-1, 1))
