import os
import torch
from pipe import StableDiffusionInverse, EulerAncestralDiscreteInverse, EulerAncestralDSG
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDPMScheduler, DDIMScheduler
from dataset import ImageDataset
from op import SuperResolutionOperator, GaussialBlurOperator, PhaseRetrievalOperator, NonlinearBlurOperator
from torchvision import transforms
import numpy as np
import argparse 
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ControlNet for Low Level vision')
    parser.add_argument('--model', type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument('--data', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--scale', type=float, default=4.8)
    parser.add_argument('--mode', type=str, default="dps")
    parser.add_argument('--operator', type=str, default="srx8")
    parser.add_argument('--stsl_k', type=int, default=2)
    parser.add_argument('--stsl_eta', type=int, default=0.1)
    parser.add_argument('--pt_lr', type=int, default=1e-5)
    parser.add_argument('--step', type=int, default=500)
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
    STEP = args.step

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
    elif args.operator == "pr":
        f = PhaseRetrievalOperator()
    elif args.operator == "nlb":
        f = NonlinearBlurOperator()
    else:
        raise NotImplementedError

    f = f.to(dtype=DTYPE, device="cuda")

    model_id = args.model
    if args.mode == "dsg" or args.mode == "max":
        scheduler = EulerAncestralDSG.from_pretrained(model_id, subfolder="scheduler")
    else:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionInverse.from_pretrained(model_id, scheduler=scheduler, torch_dtype=DTYPE)
    pipe = pipe.to("cuda")

    mses = []
    for i, (x, x_path) in enumerate(dataloader):
        if i % args.ngpu != args.rank:
            continue
        x_name = x_path[0].split('/')[-1]
        x_name = x_name[:-4] + ".png"
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x)
        if args.mode == "dps":
            _, x_hat, _ = pipe.run_dps(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)

        elif args.mode == "psld":
            _, x_hat, _ = pipe.run_psld(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "stsl":
            _, x_hat, _ = pipe.run_stsl(f=f,
                                    y=y,
                                    scale=SCALE,
                                    stsl_k=args.stsl_k,
                                    stsl_eta=args.stsl_eta,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "pt":
            _, x_hat, _ = pipe.run_pt(f=f,
                                    y=y,
                                    scale=SCALE,
                                    pt_lr=args.pt_lr,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "dsg":
            _, x_hat, _ = pipe.run_dsg(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "max":
            _, x_hat, _ = pipe.run_dsg_max(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "rsp":
            _, x_hat, _ = pipe.run_rsp(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        elif args.mode == "fdm":
            _, x_hat, _ = pipe.run_fdm(f=f,
                                    y=y,
                                    scale=SCALE,
                                    prompt="",
                                    height=512,
                                    width=512,
                                    num_inference_steps=STEP,
                                    guidance_scale=0.0)
        else:
            raise ValueError
        y_hat = f(x_hat)

        out_tensors = [x, y, x_hat, y_hat]

        mse = torch.mean((x - x_hat) ** 2)
        mses.append(mse.item())

        print("[{0}/1000], mse: {1:.6f}/avg mse: {2:.6f}".format(i, mse, np.mean(mses)))

        for i in range(4):
            save_image(out_tensors[i], os.path.join(out_dirs[i], x_name), normalize=True, value_range=(-1, 1))
